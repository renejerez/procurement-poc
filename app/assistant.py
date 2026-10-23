"""Assistente: o Gemini responde a perguntas usando ferramentas sobre os dados estruturados e os documentos."""
import concurrent.futures
import functools
import json
import os
import re
import time
from datetime import date

import streamlit as st

import lib
import rag

DEFAULT_MODEL = "gemini-3.6-flash"
# se o modelo escolhido estiver sobrecarregado ou lento (erro 503/504), tenta os seguintes, mantendo a conversa
FALLBACK_MODELS = ["gemini-3.6-flash", "gemini-3.8-flash", "gemini-3.5-flash-lite"]
SEARCH_TIMEOUT_S = 60


def api_key():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        try:
            key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            key = None
    return key


def model_name():
    name = os.environ.get("GEMINI_MODEL")
    if not name:
        try:
            name = st.secrets.get("GEMINI_MODEL")
        except Exception:
            name = None
    return name or DEFAULT_MODEL


# ─────────────────────────────── ferramentas ───────────────────────────────

def _cursor():
    path, _ = lib.db_snapshot()
    return lib._connection(path).cursor()


def _records(df, max_chars=400):
    """DataFrame -> lista de dicts serializável, com texto longo cortado."""
    out = []
    for rec in json.loads(df.to_json(orient="records", date_format="iso", force_ascii=False)):
        out.append({k: (v[:max_chars] + "…" if isinstance(v, str) and len(v) > max_chars else v) for k, v in rec.items()})
    return out


_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="pesquisa")
LIVE = None   # caixa de estado da página (st.status), onde cada passo aparece enquanto acontece


def _memory():
    """Memória usada por este processo e memória livre na máquina (só Linux, ex.: Colab), para o log."""
    try:
        status = open("/proc/self/status").read()
        rss = int(next(l for l in status.splitlines() if l.startswith("VmRSS")).split()[1]) / 1e6
        info = open("/proc/meminfo").read()
        free = int(next(l for l in info.splitlines() if l.startswith("MemAvailable")).split()[1]) / 1e6
        return f"[memória: app {rss:.1f} GB, livre {free:.1f} GB]"
    except Exception:
        return ""


def _say(text):
    print(time.strftime("%H:%M:%S"), text, _memory(), flush=True)          # vai para o log do Streamlit
    if LIVE is not None:
        try:
            LIVE.write(text)
        except Exception:
            pass


def _trace(tool, args, summary, evidence=None):
    """Regista cada chamada de ferramenta. `evidence` = fontes que a chamada encontrou, mostradas na página como
    Evidência: [{"kind": "sql"|"documento"|"contrato"|"alertas", "label": str, "detail": str, "link": str?}]."""
    st.session_state.setdefault("trace", []).append({"ferramenta": tool, "argumentos": args, "resultado": summary,
                                                     "evidencia": evidence or []})


LABELS = {"consultar_sql": "A consultar os dados estruturados (SQL)",
          "pesquisar_documentos": "A pesquisar nos documentos",
          "obter_contrato": "A obter a ficha do contrato",
          "listar_alertas": "A listar alertas"}


def timed(fn):
    """Mostra cada chamada de ferramenta em direto, com o tempo que demorou."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        label = LABELS.get(fn.__name__, fn.__name__)
        if fn.__name__ == "pesquisar_documentos" and not rag.model_loaded():
            label += " (a carregar o modelo de pesquisa, só na primeira vez: ~30 s)"
        _say(f"⏳ {label}...")
        t0 = time.time()
        result = fn(*args, **kwargs)
        ok = "⚠️ erro" if isinstance(result, dict) and "erro" in result else "✅"
        _say(f"{ok} {LABELS.get(fn.__name__, fn.__name__)} ({time.time() - t0:.1f} s)")
        return result
    return wrapper


FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|export|import|install|load|pragma|set|reset|"
    r"call|checkpoint|vacuum|use|getenv|read_\w+|glob|sniff_csv|query_table)\b", re.IGNORECASE)
MAX_ROWS = 100


def consultar_sql(sql: str) -> dict:
    """Executa uma consulta SQL (DuckDB) só de leitura sobre os dados estruturados dos contratos.

    Use para contagens, somas, médias, rankings, listas filtradas e comparações entre contratos, entidades ou
    adjudicatários. Só são permitidas instruções SELECT ou WITH, uma de cada vez. Devolve no máximo 100 linhas.

    Args:
        sql: a consulta SQL, por exemplo
            SELECT authority, SUM(contract_price) AS total FROM v_contracts GROUP BY 1 ORDER BY 2 DESC LIMIT 10
    """
    q = sql.strip().rstrip(";").strip()
    if ";" in q:
        result = {"erro": "Só é permitida uma instrução SQL de cada vez."}
    elif not re.match(r"(?is)^(select|with)\b", q):
        result = {"erro": "Só são permitidas consultas SELECT ou WITH."}
    elif FORBIDDEN_SQL.search(q):
        result = {"erro": f"A consulta usa uma instrução ou função não permitida: {FORBIDDEN_SQL.search(q).group(0)}"}
    else:
        try:
            df = _cursor().execute(f"SELECT * FROM ({q}) AS resultado LIMIT {MAX_ROWS + 1}").df()
            truncated = len(df) > MAX_ROWS
            result = {"colunas": list(df.columns), "linhas": _records(df.head(MAX_ROWS)), "n_linhas": min(len(df), MAX_ROWS),
                      "cortado_em_100": truncated}
        except Exception as e:
            result = {"erro": f"{type(e).__name__}: {str(e)[:500]}"}
    _trace("consultar_sql", {"sql": sql}, result.get("erro") or f"{result['n_linhas']} linhas",
           None if "erro" in result else [{"kind": "sql", "label": f"Consulta SQL · {result['n_linhas']} linhas",
                                           "detail": sql.strip()}])
    return result


def pesquisar_documentos(pergunta: str, contract_id: int = 0, n_resultados: int = 8) -> dict:
    """Pesquisa passagens relevantes no texto das peças dos procedimentos (cadernos de encargos, programas de
    concurso, convites, contratos).

    Use para perguntas sobre o conteúdo dos documentos: cláusulas, prazos, garantias, penalidades, requisitos,
    critérios de adjudicação, especificações técnicas. A pesquisa combina significado e palavras-chave, por isso
    reformule a pergunta com os termos que provavelmente aparecem no documento.

    Args:
        pergunta: o que procurar, em português, por exemplo "prazo de garantia dos equipamentos"
        contract_id: limitar a pesquisa aos documentos deste contrato (0 = todos os contratos)
        n_resultados: número de passagens a devolver (1 a 15)
    """
    cur = _cursor()
    if not rag.has_index(cur):
        result = {"erro": "O índice dos documentos ainda não foi criado (notebook 05_rag_index)."}
    else:
        try:
            future = _POOL.submit(rag.search, cur, pergunta, contract_id or None, max(1, min(int(n_resultados), 15)))
            try:
                hits = future.result(timeout=SEARCH_TIMEOUT_S)
            except concurrent.futures.TimeoutError:
                cur.interrupt()          # cancela a consulta em curso, para não continuar a ocupar CPU e memória
                raise TimeoutError(f"a pesquisa nos documentos excedeu {SEARCH_TIMEOUT_S} s e foi cancelada")
            result = {"passagens": [{
                "texto": h["text"][:1800],
                "fontes": [{"contract_id": c, "ficheiro": f, "link_base": f"{lib.BASE_DETAIL}{c}"}
                           for c, f in h["sources"][:5]],
                "outros_contratos_com_o_mesmo_texto": max(0, len(h["sources"]) - 5),
            } for h in hits]}
            if not hits:
                result["aviso"] = "Nenhuma passagem encontrada."
        except Exception as e:
            result = {"erro": f"{type(e).__name__}: {str(e)[:500]}"}
    _trace("pesquisar_documentos", {"pergunta": pergunta, "contract_id": contract_id or None},
           result.get("erro") or f"{len(result['passagens'])} passagens",
           [{"kind": "documento",
             "label": f"{p['fontes'][0]['ficheiro']} · contrato {p['fontes'][0]['contract_id']}"
                      + (f" (e mais {len(p['fontes']) - 1 + p['outros_contratos_com_o_mesmo_texto']})"
                         if len(p["fontes"]) > 1 or p["outros_contratos_com_o_mesmo_texto"] else ""),
             "detail": " ".join(p["texto"].split())[:280] + "…",
             "link": p["fontes"][0]["link_base"]}
            for p in result.get("passagens", [])[:5] if p["fontes"]])
    return result


def obter_contrato(contract_id: int) -> dict:
    """Devolve a ficha completa de um contrato: objeto, entidades, procedimento, preços, datas, concorrentes,
    alertas de risco calculados e lista dos documentos disponíveis.

    Args:
        contract_id: o ID do contrato no portal BASE (por exemplo 15645692)
    """
    flags = lib.compute_flags(lib.contracts(), **lib.DEFAULT_PARAMS)
    row = flags[flags["contract_id"] == int(contract_id)]
    if row.empty:
        result = {"erro": f"Contrato {contract_id} não encontrado na base de dados."}
    else:
        r = row.iloc[0]
        fields = ["contract_id", "object", "procedure_type", "contract_types", "authority", "authority_nif",
                  "contractor", "contractor_nif", "base_price", "contract_price", "effective_price", "limit",
                  "publication_date", "signing_date", "award_decision_date", "close_date", "execution_days",
                  "legal_basis", "bidders", "cpv_code", "cpv_description", "documents_link", "base_link"]
        result = _records(row[fields])[0]
        result["alertas"] = [{"indicador": lib.flag_label(k, lib.DEFAULT_PARAMS),
                              "explicacao": lib.flag_help(k, lib.DEFAULT_PARAMS)} for k in lib.FLAGS if r[k]]
        cur = _cursor()
        result["concorrentes"] = _records(cur.execute("""
            SELECT e.name AS nome, p.nif FROM contract_parties p JOIN entities e USING (nif)
            WHERE p.contract_id = ? AND p.role = 'competitor' ORDER BY e.name
        """, [int(contract_id)]).df())
        if lib.has_table("documents"):
            result["documentos"] = _records(cur.execute("""
                SELECT file_name AS ficheiro, file_type AS tipo, status AS estado, pages AS paginas
                FROM documents WHERE contract_id = ? ORDER BY file_name
            """, [int(contract_id)]).df())
    _trace("obter_contrato", {"contract_id": contract_id}, result.get("erro") or result.get("object", "")[:80],
           None if "erro" in result else [{
               "kind": "contrato", "label": f"Contrato {result['contract_id']}: {(result.get('object') or '')[:90]}",
               "detail": f"{result.get('authority')} → {result.get('contractor')} · {lib.eur(result.get('contract_price'), 2)}"
                         + (f" · {len(result['alertas'])} alerta(s)" if result.get("alertas") else ""),
               "link": result.get("base_link"), "contractor": result.get("contractor")}])
    return result


def listar_alertas(indicador: str = "", entidade: str = "", limite: int = 20) -> dict:
    """Lista contratos assinalados pelos indicadores de risco, ordenados por número de alertas e valor.

    Indicadores disponíveis (use o código): proposta_unica, perto_limiar, acima_limiar, fracionamento,
    acima_preco_base, derrapagem, publicacao_tardia, concentracao. Vazio = qualquer indicador.

    Args:
        indicador: código do indicador, ou vazio para todos
        entidade: parte do nome da entidade adjudicante ou do adjudicatário, para filtrar (opcional)
        limite: número máximo de contratos a devolver (até 50)
    """
    df = lib.compute_flags(lib.contracts(), **lib.DEFAULT_PARAMS)
    if indicador and indicador not in lib.FLAGS:
        result = {"erro": f"Indicador desconhecido. Use um de: {', '.join(lib.FLAGS)}"}
    else:
        view = df[df[indicador]] if indicador else df[df["n_flags"] > 0]
        if entidade:
            m = view["authority"].str.contains(entidade, case=False, regex=False) | \
                view["contractor"].str.contains(entidade, case=False, regex=False)
            view = view[m]
        total, value = len(view), float(view["contract_price"].sum())
        view = view.sort_values(["n_flags", "contract_price"], ascending=False).head(max(1, min(int(limite), 50)))
        view = view.assign(alertas=view.apply(lambda r: lib.flags_for(r, lib.DEFAULT_PARAMS), axis=1))
        result = {"total_contratos_assinalados": total, "valor_total": round(value, 2),
                  "indicadores": {k: lib.flag_help(k, lib.DEFAULT_PARAMS) for k in ([indicador] if indicador else lib.FLAGS)},
                  "contratos": _records(view[["contract_id", "object", "authority", "contractor", "procedure_type",
                                              "contract_price", "limit", "bidders", "publication_date", "alertas",
                                              "base_link"]])}
    _trace("listar_alertas", {"indicador": indicador, "entidade": entidade},
           result.get("erro") or f"{result['total_contratos_assinalados']} contratos",
           None if "erro" in result else [{
               "kind": "alertas",
               "label": (lib.flag_label(indicador, lib.DEFAULT_PARAMS) if indicador else "Contratos com alertas")
                        + f" · {result['total_contratos_assinalados']} contratos",
               "detail": ", ".join(str(c["contract_id"]) for c in result["contratos"][:10])}])
    return result


TOOLS = [timed(f) for f in (consultar_sql, pesquisar_documentos, obter_contrato, listar_alertas)]

# ─────────────────────────────── instruções ───────────────────────────────

SCHEMA_NOTES = """
Tabelas (DuckDB). Valores em euros, sem IVA.
- v_contracts (USE ESTA para quase tudo): uma linha por contrato.
  contract_id, publication_date, procedure_type, object, contract_price,
  authority_nif, authority (entidade adjudicante), contractor_nif, contractor (adjudicatário).
  Se houver vários adjudicantes/adjudicatários, os nomes vêm juntos com ' + '.
- contracts: todos os campos de cada contrato. Colunas úteis: contract_id, procedure_id, procedure_type, contract_types,
  object, description, publication_date, award_decision_date, signing_date, close_date, base_price, contract_price,
  effective_price, execution_days, legal_basis, direct_award_justification, framework_agreement, centralized,
  green_contract, documents_link, year. NÃO selecione raw_record.
- entities(nif, name): entidades e empresas, identificadas pelo NIF.
- contract_parties(contract_id, nif, role, is_sme): role = 'authority' | 'contractor' | 'competitor' (concorrente).
  Número de concorrentes de um contrato = COUNT(*) com role = 'competitor'.
- contract_cpvs(contract_id, cpv_code, cpv_description)
- contract_locations(contract_id, country, district, municipality)
- documents(contract_id, file_name, file_type, pages, status, text): NÃO selecione a coluna text; para o conteúdo dos
  documentos use a ferramenta pesquisar_documentos.
Valores de procedure_type: 'Ajuste Direto Regime Geral', 'Consulta Prévia', 'Concurso público',
'Concurso limitado por prévia qualificação', 'Ao abrigo de acordo-quadro (art.º 259.º)',
'Ao abrigo de acordo-quadro (art.º 258.º)', 'Consulta Prévia Simplificada', 'Contratação excluída II', ...
Para procurar nomes use ILIKE '%...%' (os nomes variam na grafia) ou, melhor, o NIF.
"""


def system_instruction():
    return f"""És o assistente de análise de contratação pública de um projeto de mestrado da Universidade da Beira Interior.
Os dados são contratos públicos com local de execução no concelho da Covilhã, publicados no portal BASE (IMPIC),
e o texto extraído das peças dos procedimentos. Hoje é {date.today():%d/%m/%Y}.

Como responder:
- Responde sempre em português de Portugal, de forma clara e concisa.
- Baseia-te SÓ nos resultados das ferramentas. Nunca inventes números, nomes, datas ou cláusulas.
- Números, totais, contagens, rankings e comparações: usa consultar_sql.
- Conteúdo dos documentos (cláusulas, prazos, garantias, penalidades, requisitos, critérios): usa pesquisar_documentos;
  se a pergunta for sobre um contrato concreto, passa o contract_id.
- Pergunta sobre um contrato concreto: começa por obter_contrato.
- Riscos ou irregularidades: usa listar_alertas e explica o que cada indicador significa. Os alertas são indícios
  que merecem análise, não provas; nunca acuses ninguém de ilegalidade.
- Se uma ferramenta der erro, corrige e tenta de novo UMA vez (por exemplo, ajusta o SQL). Se voltar a falhar,
  não insistas: responde com o que já tens e diz o que não foi possível consultar.
- Usa no máximo 5 chamadas de ferramentas por pergunta.
- Se os dados não permitirem responder, diz isso claramente e explica o que falta.
- Cita as fontes: para contratos, um link no formato [contrato 15645692](https://www.base.gov.pt/Base4/pt/detalhe/?type=contratos&id=15645692);
  para documentos, o nome do ficheiro e o contrato. Quando citares texto de um documento, usa poucas palavras entre aspas.
- Valores em euros com formato português (1.234,56 €). Os preços não incluem IVA.

{SCHEMA_NOTES}"""


# ─────────────────────────────── conversa ───────────────────────────────

@st.cache_resource
def _client(key):
    """Um só cliente para toda a aplicação. Tem de continuar vivo enquanto as conversas existirem:
    se fosse apagado, fecharia a ligação e as conversas deixariam de conseguir enviar pedidos."""
    from google import genai
    from google.genai import types
    return genai.Client(api_key=key, http_options=types.HttpOptions(
        timeout=45_000,                                                     # 45 s por pedido
        retry_options=types.HttpRetryOptions(attempts=2, initial_delay=2.0, max_delay=10.0,
                                             http_status_codes=[500, 502, 503, 504])))   # 429 = quota: mudar de modelo, não insistir


def models_to_try():
    first = model_name()
    return [first] + [m for m in FALLBACK_MODELS if m != first]


def new_chat(model=None, history=None):
    from google.genai import types

    client = _client(api_key())
    config = types.GenerateContentConfig(
        system_instruction=system_instruction(),
        tools=TOOLS,
        temperature=0.2,
        # o Gemini 3 "pensa" a nível alto por omissão, o que é lento; para escolher ferramentas e resumir basta "low"
        thinking_config=types.ThinkingConfig(thinking_level=os.environ.get("GEMINI_THINKING", "low")),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=10),
    )
    return client.chats.create(model=model or model_name(), config=config, history=history or [])


def _overloaded(e):
    """Erros temporários do lado da Google: modelo sobrecarregado (503) ou pedido que excedeu o prazo (504)."""
    text = str(e)
    return any(k in text for k in ["503", "UNAVAILABLE", "high demand", "504", "DEADLINE_EXCEEDED"]) \
        or "overloaded" in text.lower()


def _quota(e):
    """Limite do plano gratuito (429). Cada modelo tem a sua própria quota, por isso vale a pena mudar de modelo."""
    text = str(e)
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def _switch_model(e):
    return _overloaded(e) or _quota(e)


def send(message):
    """Envia a mensagem na conversa da sessão. Se o modelo estiver sobrecarregado, passa ao seguinte."""
    models = models_to_try()
    current = st.session_state.get("chat_model", models[0])
    start = models.index(current) if current in models else 0
    for i, model in enumerate(models[start:], start):
        chat = st.session_state.get("chat")
        if chat is None or st.session_state.get("chat_model") != model:
            history = chat.get_history() if chat is not None else None
            chat = new_chat(model, history)
            st.session_state["chat"], st.session_state["chat_model"] = chat, model
        try:
            response = chat.send_message(message)
            parts = (response.candidates[0].content.parts or []) if response.candidates else []
            text = "".join(p.text for p in parts if getattr(p, "text", None))
            if not text:
                _say("⚠️ o Gemini esgotou os passos sem escrever a resposta final")
                return ("Não consegui chegar a uma resposta final (o assistente usou demasiados passos). "
                        "Tente reformular a pergunta de forma mais concreta, ou escolha um contrato em foco.")
            return text
        except Exception as e:
            if _switch_model(e) and i + 1 < len(models):
                reason = "atingiu o limite do plano gratuito" if _quota(e) else "não respondeu a tempo ou está sobrecarregado"
                _say(f"⚠️ {model} {reason}; a tentar {models[i + 1]}")
                continue
            raise


def friendly_error(e):
    text = str(e)
    if _overloaded(e):
        return ("Os modelos do Gemini estão sobrecarregados ou lentos neste momento (erro 503/504 da Google). "
                "Tente daqui a alguns minutos.")
    if _quota(e):
        if "PerDay" in text or "per day" in text.lower():
            return ("Foi atingido o limite diário do plano gratuito do Gemini em todos os modelos disponíveis. "
                    "O limite renova-se à meia-noite da Califórnia (por volta das 8h em Portugal).")
        return ("Foi atingido o limite de pedidos por minuto do plano gratuito do Gemini em todos os modelos disponíveis. "
                "Aguarde um minuto e tente de novo.")
    if "API_KEY" in text.upper() or "401" in text or "403" in text:
        return "A chave da API do Gemini não é válida ou não tem acesso a este modelo."
    if "timeout" in text.lower() or "timed out" in text.lower():
        return "O Gemini demorou demasiado a responder (mais de 45 s). Tente de novo ou simplifique a pergunta."
    if "404" in text and "model" in text.lower():
        return f"O modelo '{model_name()}' não está disponível. Defina outro em GEMINI_MODEL."
    return f"Erro ao contactar o Gemini: {type(e).__name__}: {text[:300]}"
