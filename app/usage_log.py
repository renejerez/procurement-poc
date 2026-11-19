"""Registo do uso do Gemini: cada pedido (tokens, latência, modelo) e cada pergunta, numa base DuckDB própria.

Porque não em procurement.duckdb: a aplicação abre essa base só para leitura e substitui-a por uma cópia nova
(notebook 02 no Drive, release «dados» no Streamlit Cloud). Um registo escrito lá perdia-se na atualização seguinte.
Este ficheiro, usage.duckdb, é escrito só pela aplicação e pelo notebook de avaliação.

Onde fica, por ordem:
1. USAGE_DB (variável de ambiente ou secret):
   - "md:<base>" -> MotherDuck (DuckDB na nuvem), com o token no secret MOTHERDUCK_TOKEN. É a mesma base para o
     Streamlit Cloud e para o Colab, e não se perde quando a aplicação reinicia. A base é criada se não existir.
     A ligação usa o endpoint Postgres do MotherDuck (psycopg), porque o MotherDuck só aceita clientes DuckDB
     1.4–1.5 e a aplicação usa o DuckDB 2.0. O SQL continua a ser DuckDB, executado no MotherDuck.
     MOTHERDUCK_HOST: anfitrião da região da conta (por omissão pg.us-east-1-aws.motherduck.com).
   - outro valor -> caminho de um ficheiro local, por exemplo na pasta do projeto no Drive
2. ao lado de PROCUREMENT_DB, se esta for um caminho local (Colab, computador)
3. pasta temporária (Streamlit Cloud sem MotherDuck: perde-se quando a aplicação reinicia)

No MotherDuck cada escrita é um pedido pela internet, por isso as escritas correm numa thread à parte, por ordem,
e o utilizador não espera por elas.

O custo é estimado com os preços da tabela gemini_prices (por 1 milhão de tokens), que começa vazia: no plano
gratuito o custo real é 0. Os tokens de raciocínio contam como saída, como na faturação da Google.
Uma falha a gravar o registo nunca interrompe o assistente: aparece só no log.
"""
import concurrent.futures
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import duckdb

QUESTION_MAX_CHARS = 1000
ERROR_MAX_CHARS = 300

SCHEMA = """
CREATE TABLE IF NOT EXISTS gemini_questions (
    question_id      VARCHAR,          -- UUID
    asked_at         TIMESTAMPTZ NOT NULL,
    session_id       VARCHAR,
    source           VARCHAR,          -- 'app' ou 'avaliacao'
    question         VARCHAR,          -- primeiros 1.000 caracteres
    contract_focus   BIGINT,
    scope            VARCHAR,          -- âmbito escolhido: concelho, ano e procedimento
    status           VARCHAR,          -- 'ok', 'sem_resposta', 'erro'
    answer_chars     INTEGER,
    n_requests       INTEGER,
    n_tool_calls     INTEGER,
    elapsed_s        DOUBLE,
    error            VARCHAR
);
CREATE TABLE IF NOT EXISTS gemini_calls (
    call_id          VARCHAR,          -- UUID
    question_id      VARCHAR,
    called_at        TIMESTAMPTZ NOT NULL,
    round            INTEGER,          -- 1 = pergunta; 2, 3... = depois dos resultados das ferramentas
    model            VARCHAR,
    ok               BOOLEAN,
    error            VARCHAR,
    prompt_tokens    INTEGER,
    output_tokens    INTEGER,
    thinking_tokens  INTEGER,
    cached_tokens    INTEGER,
    tool_prompt_tokens INTEGER,
    total_tokens     INTEGER,
    latency_s        DOUBLE,
    function_calls   VARCHAR           -- ferramentas pedidas nesta resposta, separadas por vírgulas
);
CREATE TABLE IF NOT EXISTS gemini_prices (
    model            VARCHAR,
    valid_from       DATE,
    input_per_1m     DOUBLE,           -- preço por 1 milhão de tokens de entrada
    output_per_1m    DOUBLE,           -- preço por 1 milhão de tokens de saída (inclui raciocínio)
    cached_per_1m    DOUBLE,           -- tokens em cache; se NULL, usa input_per_1m
    currency         VARCHAR DEFAULT 'USD'
);
CREATE OR REPLACE VIEW v_gemini_calls_cost AS
SELECT c.*,
       p.currency,
       (  (coalesce(c.prompt_tokens, 0) - coalesce(c.cached_tokens, 0)) * p.input_per_1m
        + coalesce(c.cached_tokens, 0) * coalesce(p.cached_per_1m, p.input_per_1m)
        + (coalesce(c.output_tokens, 0) + coalesce(c.thinking_tokens, 0)) * p.output_per_1m
       ) / 1e6 AS cost
FROM gemini_calls c
LEFT JOIN LATERAL (
    SELECT * FROM gemini_prices p
    WHERE p.model = c.model AND p.valid_from <= CAST(c.called_at AS DATE)
    ORDER BY p.valid_from DESC LIMIT 1
) p ON TRUE;
CREATE OR REPLACE VIEW v_gemini_usage_daily AS
SELECT CAST(called_at AS DATE) AS dia, model,
       count(*) AS pedidos,
       count(*) FILTER (WHERE NOT ok) AS pedidos_com_erro,
       count(DISTINCT question_id) AS perguntas,
       sum(prompt_tokens) AS tokens_entrada,
       sum(output_tokens) AS tokens_saida,
       sum(thinking_tokens) AS tokens_raciocinio,
       sum(total_tokens) AS tokens_totais,
       round(median(latency_s), 2) AS latencia_mediana_s,
       sum(cost) AS custo_estimado,
       any_value(currency) AS moeda
FROM v_gemini_calls_cost
GROUP BY ALL;
"""

_LOCK = threading.Lock()
_CON = {}
_WRITER = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="usage_log")   # escritas por ordem
_PENDING = []


def _setting(name):
    value = os.environ.get(name)
    if not value:
        try:
            import streamlit as st
            value = st.secrets.get(name)
        except Exception:
            value = None
    return value


def usage_db_path():
    """Caminho do ficheiro local, ou "md:<base>" para o MotherDuck."""
    explicit = _setting("USAGE_DB")
    if explicit:
        return explicit if explicit.startswith("md:") else Path(explicit)
    main = os.environ.get("PROCUREMENT_DB")
    if main and not main.startswith("github-release://"):
        return Path(main).with_name("usage.duckdb")
    return Path(tempfile.gettempdir()) / "usage.duckdb"


def is_remote():
    return str(usage_db_path()).startswith("md:")


MD_DEFAULT_HOST = "pg.us-east-1-aws.motherduck.com"


class _PgConnection:
    """Ligação ao MotherDuck pelo endpoint Postgres, com a mesma interface que o DuckDB usa aqui:
    execute(sql, params) com marcadores ?, e .df() para ler o resultado."""

    def __init__(self, conn):
        self.conn, self._cur = conn, None

    def execute(self, sql, params=None):
        if params is None:                 # vários comandos (o esquema): tirar comentários e separar por ;
            plain = "\n".join(line.split("--", 1)[0] for line in sql.splitlines())
            statements = [x.strip() for x in plain.split(";") if x.strip()]
        else:
            statements = [sql]
        for stmt in statements:
            self._cur = self.conn.cursor()
            self._cur.execute(stmt.replace("?", "%s") if params is not None else stmt, params)
        return self

    def df(self):
        import pandas as pd
        cols = [d.name for d in self._cur.description] if self._cur.description else []
        return pd.DataFrame(self._cur.fetchall(), columns=cols)

    def close(self):
        self.conn.close()


def _ca_bundle():
    """Certificados de confiança para verificar o MotherDuck. O psycopg[binary] traz a sua própria libpq, que não
    encontra os certificados do sistema (Colab: «certificate verify failed»); o certifi vem com o requests."""
    custom = _setting("MOTHERDUCK_SSLROOTCERT")
    if custom:
        return custom
    try:
        import certifi
        return certifi.where()
    except ImportError:
        return "system"


def _pg_connect(dbname, token, host):
    import psycopg
    return psycopg.connect(host=host, port=5432, dbname=dbname, user="postgres", password=token,
                           sslmode="verify-full", sslrootcert=_ca_bundle(), autocommit=True, connect_timeout=15)


def _open(target):
    if str(target).startswith("md:"):
        name = str(target)[3:].split("?")[0] or "procurement_usage"
        token = _setting("MOTHERDUCK_TOKEN") or _setting("motherduck_token")
        if not token:
            raise RuntimeError("falta o secret MOTHERDUCK_TOKEN")
        host = _setting("MOTHERDUCK_HOST") or MD_DEFAULT_HOST
        try:                                           # criar a base na primeira vez
            admin = _pg_connect("md:", token, host)
            admin.execute(f'CREATE DATABASE IF NOT EXISTS "{name}"')
            admin.close()
        except Exception as e:
            print(f"[usage_log] CREATE DATABASE {name}: {type(e).__name__}: {e}", flush=True)
        return _PgConnection(_pg_connect(name, token, host))
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(target))


def _connection():
    """Uma ligação por processo (o Streamlit corre todas as sessões no mesmo processo); usar sempre com _LOCK."""
    target = str(usage_db_path())
    con = _CON.get(target)
    if con is None:
        con = _open(target)
        con.execute(SCHEMA)
        _CON.clear()
        _CON[target] = con
    return con


def close():
    flush()
    with _LOCK:
        for con in _CON.values():
            con.close()
        _CON.clear()


def flush(timeout=30):
    """Espera pelas escritas em curso (testes, fim de uma avaliação)."""
    while _PENDING:
        f = _PENDING.pop(0)
        try:
            f.result(timeout=timeout)
        except Exception:
            pass


def _guarded(fn, *args, **kwargs):
    try:
        with _LOCK:
            return fn(*args, **kwargs)
    except Exception as e:
        print(f"[usage_log] {fn.__name__} falhou: {type(e).__name__}: {e}", flush=True)
        with _LOCK:                       # ligação perdida (rede, token): abrir de novo na próxima escrita
            for con in _CON.values():
                try:
                    con.close()
                except Exception:
                    pass
            _CON.clear()
        return None


def _safe(fn):
    """Leituras: correm já. Uma falha vai para o log e nunca chega ao utilizador."""
    def wrapper(*args, **kwargs):
        return _guarded(fn, *args, **kwargs)
    wrapper.__name__ = fn.__name__
    return wrapper


def _write(fn):
    """Escritas: no MotherDuck vão para a thread de escrita (por ordem); num ficheiro local correm já."""
    def wrapper(*args, **kwargs):
        if is_remote():
            _PENDING[:] = [f for f in _PENDING if not f.done()]
            _PENDING.append(_WRITER.submit(_guarded, fn, *args, **kwargs))
            return None
        return _guarded(fn, *args, **kwargs)
    wrapper.__name__ = fn.__name__
    return wrapper


def _now():
    return datetime.now(timezone.utc)


def _usage(response):
    """Tokens reportados pela API (usage_metadata); campos ausentes ficam NULL."""
    u = getattr(response, "usage_metadata", None)
    get = (lambda k: getattr(u, k, None)) if u is not None else (lambda k: None)
    return (get("prompt_token_count"), get("candidates_token_count"), get("thoughts_token_count"),
            get("cached_content_token_count"), get("tool_use_prompt_token_count"), get("total_token_count"))


def start_question(question, session_id=None, source="app", contract_focus=None, scope=None):
    qid = str(uuid.uuid4())
    _insert_question(qid, _now(), question, session_id, source, contract_focus, scope)
    return qid


@_write
def _insert_question(qid, asked_at, question, session_id, source, contract_focus, scope):
    _connection().execute(
        "INSERT INTO gemini_questions (question_id, asked_at, session_id, source, question, contract_focus, scope) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [qid, asked_at, session_id, source, (question or "")[:QUESTION_MAX_CHARS], contract_focus or None, scope])


def log_call(question_id, round, model, latency_s, response=None, error=None):
    names = None
    if response is not None:
        calls = getattr(response, "function_calls", None) or []
        names = ",".join(fc.name for fc in calls) or None
    row = [str(uuid.uuid4()), question_id, _now(), round, model, error is None,
           (str(error)[:ERROR_MAX_CHARS] if error is not None else None),
           *_usage(response), round_or_none(latency_s), names]
    _insert_call(row)


@_write
def _insert_call(row):
    _connection().execute("INSERT INTO gemini_calls VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", row)


def round_or_none(x, nd=3):
    return None if x is None else round(float(x), nd)


@_write
def finish_question(question_id, status, answer="", elapsed_s=None, error=None):
    if not question_id:
        return
    _connection().execute("""
        UPDATE gemini_questions SET
            status = ?, answer_chars = ?, elapsed_s = ?, error = ?,
            n_requests = (SELECT count(*) FROM gemini_calls WHERE question_id = ?),
            n_tool_calls = (SELECT coalesce(sum(len(string_split(function_calls, ','))), 0)
                            FROM gemini_calls WHERE question_id = ? AND function_calls IS NOT NULL)
        WHERE question_id = ?""",
        [status, len(answer or ""), round_or_none(elapsed_s),
         (str(error)[:ERROR_MAX_CHARS] if error is not None else None), question_id, question_id, question_id])


@_safe
def set_price(model, input_per_1m, output_per_1m, valid_from="2026-01-01", cached_per_1m=None, currency="USD"):
    """Regista ou substitui o preço de um modelo, a partir de uma data (preços oficiais da Google)."""
    con = _connection()
    con.execute("DELETE FROM gemini_prices WHERE model = ? AND valid_from = ?", [model, valid_from])
    con.execute("INSERT INTO gemini_prices VALUES (?, ?, ?, ?, ?, ?)",
                [model, valid_from, input_per_1m, output_per_1m, cached_per_1m, currency])


def query(sql, params=()):
    """Consulta de leitura sobre o registo (usada nos testes e na avaliação). Espera primeiro pelas escritas."""
    flush()
    return _guarded(_query, sql, params)


def _query(sql, params=()):
    return _connection().execute(sql, list(params)).df()
