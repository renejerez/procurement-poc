"""Registo do uso do Gemini: cada pedido (tokens, latência, modelo) e cada pergunta, numa base DuckDB própria.

Porque não em procurement.duckdb: a aplicação abre essa base só para leitura e substitui-a por uma cópia nova
(notebook 02 no Drive, release «dados» no Streamlit Cloud). Um registo escrito lá perdia-se na atualização seguinte.
Este ficheiro, usage.duckdb, é escrito só pela aplicação e pelo notebook de avaliação.

Onde fica, por ordem:
1. USAGE_DB (variável de ambiente ou secret): caminho do ficheiro, por exemplo na pasta do projeto no Drive
2. ao lado de PROCUREMENT_DB, se esta for um caminho local (Colab, computador)
3. pasta temporária (Streamlit Cloud: perde-se quando a aplicação reinicia)

O custo é estimado com os preços da tabela gemini_prices (por 1 milhão de tokens), que começa vazia: no plano
gratuito o custo real é 0. Os tokens de raciocínio contam como saída, como na faturação da Google.
Uma falha a gravar o registo nunca interrompe o assistente: aparece só no log.
"""
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
    question_id      VARCHAR PRIMARY KEY,
    asked_at         TIMESTAMPTZ NOT NULL,
    session_id       VARCHAR,
    source           VARCHAR,          -- 'app' ou 'avaliacao'
    question         VARCHAR,          -- primeiros 1.000 caracteres
    contract_focus   BIGINT,
    scope            VARCHAR,          -- âmbito escolhido: concelho; ano; procedimento
    status           VARCHAR,          -- 'ok', 'sem_resposta', 'erro'
    answer_chars     INTEGER,
    n_requests       INTEGER,
    n_tool_calls     INTEGER,
    elapsed_s        DOUBLE,
    error            VARCHAR
);
CREATE TABLE IF NOT EXISTS gemini_calls (
    call_id          VARCHAR PRIMARY KEY,
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
    currency         VARCHAR DEFAULT 'USD',
    PRIMARY KEY (model, valid_from)
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
    explicit = _setting("USAGE_DB")
    if explicit:
        return Path(explicit)
    main = os.environ.get("PROCUREMENT_DB")
    if main and not main.startswith("github-release://"):
        return Path(main).with_name("usage.duckdb")
    return Path(tempfile.gettempdir()) / "usage.duckdb"


def _connection():
    """Uma ligação por processo (o Streamlit corre todas as sessões no mesmo processo); usar sempre com _LOCK."""
    path = str(usage_db_path())
    con = _CON.get(path)
    if con is None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(path)
        con.execute(SCHEMA)
        _CON.clear()
        _CON[path] = con
    return con


def close():
    with _LOCK:
        for con in _CON.values():
            con.close()
        _CON.clear()


def _safe(fn):
    """Uma falha no registo vai para o log e nunca chega ao utilizador."""
    def wrapper(*args, **kwargs):
        try:
            with _LOCK:
                return fn(*args, **kwargs)
        except Exception as e:
            print(f"[usage_log] {fn.__name__} falhou: {type(e).__name__}: {e}", flush=True)
            return None
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


@_safe
def start_question(question, session_id=None, source="app", contract_focus=None, scope=None):
    qid = str(uuid.uuid4())
    _connection().execute(
        "INSERT INTO gemini_questions (question_id, asked_at, session_id, source, question, contract_focus, scope) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [qid, _now(), session_id, source, (question or "")[:QUESTION_MAX_CHARS], contract_focus or None, scope])
    return qid


@_safe
def log_call(question_id, round, model, latency_s, response=None, error=None):
    names = None
    if response is not None:
        calls = getattr(response, "function_calls", None) or []
        names = ",".join(fc.name for fc in calls) or None
    _connection().execute(
        "INSERT INTO gemini_calls VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [str(uuid.uuid4()), question_id, _now(), round, model, error is None,
         (str(error)[:ERROR_MAX_CHARS] if error is not None else None),
         *_usage(response), round_or_none(latency_s), names])


def round_or_none(x, nd=3):
    return None if x is None else round(float(x), nd)


@_safe
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
    _connection().execute("INSERT OR REPLACE INTO gemini_prices VALUES (?, ?, ?, ?, ?, ?)",
                          [model, valid_from, input_per_1m, output_per_1m, cached_per_1m, currency])


@_safe
def query(sql, params=()):
    """Consulta de leitura sobre o registo (usada nos testes e na avaliação)."""
    return _connection().execute(sql, list(params)).df()
