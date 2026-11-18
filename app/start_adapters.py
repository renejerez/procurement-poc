"""
Ligação entre a página inicial (views/assistant.py) e o código existente: lib.py (dados, indicadores) e
assistant.py (Gemini com ferramentas).

Cada função devolve um valor seguro se algo correr mal, para a página inicial nunca deixar de abrir.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

import assistant
import lib
import usage_log

BRIEFING_DAYS = 7          # "novos" = publicados nos últimos N dias, até à data de publicação mais recente

# ordem de gravidade usada para escolher o alerta principal de cada contrato no Briefing
FLAG_PRIORITY = ["proposta_unica", "fracionamento", "acima_limiar", "perto_limiar", "acima_preco_base",
                 "derrapagem", "concentracao", "publicacao_tardia"]


COUNTRY = "Todo o país"


@dataclass
class Scope:
    municipio: str             # concelho, ou COUNTRY
    ano: int | None            # None = todos os anos
    procedimento: str | None   # None = todos

    @property
    def place(self) -> lib.Place:
        return lib.Place(None, None if self.municipio == COUNTRY else self.municipio,
                         (self.ano,) if self.ano else ())


@dataclass
class Flag:
    indicador: str
    contract_id: str
    entidade: str
    adjudicatario: str
    procedimento: str
    data: str  # "21/10"
    valor: float
    extra_flags: int = 0


@dataclass
class Evidence:
    kind: str  # "contrato" | "documento" | "sql" | "alertas"
    label: str
    detail: str = ""
    link: str | None = None


@dataclass
class Answer:
    text: str
    evidence: list[Evidence] = field(default_factory=list)
    follow_ups: list[str] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)      # chamadas de ferramentas ("Como cheguei a esta resposta")


def _in_scope(df: pd.DataFrame, scope: Scope) -> pd.DataFrame:
    if scope.ano:
        df = df[df["publication_date"].dt.year == scope.ano]
    if scope.procedimento:
        df = df[df["procedure_type"] == scope.procedimento]
    return df


# --------------------------------------------------------------------------
# Opções de âmbito (dentro do composer)
# --------------------------------------------------------------------------
def scope_options() -> dict:
    """Concelhos (o predefinido primeiro, depois todo o país e os outros por ordem alfabética), anos e procedimentos."""
    try:
        munis = sorted(set(lib.places()["municipality"]))
        first = [lib.DEFAULT_MUNICIPALITY] if lib.DEFAULT_MUNICIPALITY in munis else []
        municipios = first + [COUNTRY] + [m for m in munis if m not in first]
        procedimentos = lib.query("""
            SELECT procedure_type FROM contracts WHERE procedure_type IS NOT NULL
            GROUP BY 1 ORDER BY COUNT(*) DESC
        """)["procedure_type"].tolist()
        return {"municipios": municipios, "anos": lib.years_available(), "procedimentos": procedimentos}
    except Exception as e:
        print(f"scope_options: {type(e).__name__}: {e}", flush=True)
        return {"municipios": [lib.DEFAULT_MUNICIPALITY, COUNTRY], "anos": [], "procedimentos": []}


# --------------------------------------------------------------------------
# Frescura dos dados (barra lateral): última execução do notebook 02
# --------------------------------------------------------------------------
def last_pipeline_run() -> pd.Timestamp | None:
    try:
        value = lib.query("SELECT MAX(loaded_at) AS t FROM contracts")["t"].iloc[0]
        if value is None or pd.isna(value):
            return None
        ts = pd.Timestamp(value)
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts       # o Colab grava em UTC
    except Exception as e:
        print(f"last_pipeline_run: {type(e).__name__}: {e}", flush=True)
        return None


# --------------------------------------------------------------------------
# KPI numa linha (as mesmas contas da Visão geral)
# --------------------------------------------------------------------------
def kpis(scope: Scope) -> dict:
    """Calculado no DuckDB (agregados), por isso funciona também para todo o país."""
    try:
        ids_sql, params = lib.place_ids_sql(scope.place)
        if scope.procedimento:
            ids_sql += " AND procedure_type = ?"
            params = [*params, scope.procedimento]
        r = lib.query(f"""
            WITH ids AS ({ids_sql})
            SELECT (SELECT COUNT(*) FROM ids) AS contratos,
                   (SELECT SUM(CAST(contract_price AS DOUBLE)) FROM contracts WHERE contract_id IN (SELECT contract_id FROM ids)) AS valor,
                   (SELECT COUNT(DISTINCT nif) FROM contract_parties
                     WHERE role = 'authority' AND contract_id IN (SELECT contract_id FROM ids)) AS entidades,
                   (SELECT COUNT(DISTINCT nif) FROM contract_parties
                     WHERE role = 'contractor' AND contract_id IN (SELECT contract_id FROM ids)) AS adjudicatarios
        """, params).iloc[0]
        return {"contratos": int(r["contratos"]), "valor": float(r["valor"] or 0),
                "entidades": int(r["entidades"]), "adjudicatarios": int(r["adjudicatarios"])}
    except Exception as e:
        print(f"kpis: {type(e).__name__}: {e}", flush=True)
        return {"contratos": 0, "valor": 0.0, "entidades": 0, "adjudicatarios": 0}


# --------------------------------------------------------------------------
# Briefing: contratos publicados nos últimos dias + os que têm alertas.
# Vem dos indicadores (lib.compute_flags), NÃO do LLM: é instantâneo e nunca inventado.
# --------------------------------------------------------------------------
def briefing(scope: Scope, limit: int = 3) -> tuple[int, int, list[Flag], pd.Timestamp | None]:
    """Devolve (n.º de contratos recentes, n.º desses com alertas, os `limit` mais graves, data mais recente)."""
    if scope.municipio == COUNTRY:
        return 0, 0, [], None                  # o Briefing é por concelho (os indicadores precisam de um âmbito)
    try:
        # só o ano mais recente do âmbito: o Briefing mostra os últimos dias publicados
        year = scope.ano or (lib.years_available() or [None])[0]
        place = lib.Place(None, scope.municipio, (year,) if year else ())
        df = _in_scope(lib.compute_flags(lib.contracts_in_scope(place), **lib.DEFAULT_PARAMS), scope)
        if df.empty:
            return 0, 0, [], None
        latest = df["publication_date"].max()
        recent = df[df["publication_date"] > latest - pd.Timedelta(days=BRIEFING_DAYS)]
        flagged = recent[recent["n_flags"] > 0]
        rank = {k: i for i, k in enumerate(FLAG_PRIORITY)}
        first = flagged.apply(lambda r: min((k for k in lib.FLAGS if r[k]), key=lambda k: rank.get(k, 99)), axis=1) \
            if not flagged.empty else pd.Series(dtype=str)
        flagged = flagged.assign(_main=first, _rank=first.map(rank))
        top = flagged.sort_values(["_rank", "n_flags", "contract_price"], ascending=[True, False, False]).head(limit)
        flags = [Flag(indicador=lib.flag_label(r["_main"], lib.DEFAULT_PARAMS), contract_id=str(r["contract_id"]),
                      entidade=r["authority"], adjudicatario=r["contractor"], procedimento=r["procedure_type"],
                      data=f"{r['publication_date']:%d/%m}", valor=float(r["contract_price"] or 0),
                      extra_flags=int(r["n_flags"]) - 1)
                 for _, r in top.iterrows()]
        return len(recent), len(flagged), flags, latest
    except Exception as e:
        print(f"briefing: {type(e).__name__}: {e}", flush=True)
        return 0, 0, [], None


# --------------------------------------------------------------------------
# Assistente: usa a conversa Gemini de assistant.py (troca de modelo, limites, etc.)
# --------------------------------------------------------------------------
def scope_text(scope: Scope) -> str:
    parts = ["todo o país (todos os concelhos)" if scope.municipio == COUNTRY else f"concelho {scope.municipio}",
             f"contratos publicados em {scope.ano}" if scope.ano else "todos os anos",
             f"tipo de procedimento «{scope.procedimento}»" if scope.procedimento else "todos os tipos de procedimento"]
    return "; ".join(parts)


def ask(question: str, scope: Scope) -> Answer:
    """Envia a pergunta na conversa da sessão. O histórico fica na própria conversa do Gemini."""
    import streamlit as st

    st.session_state["trace"] = []
    message = (f"[Âmbito escolhido pelo utilizador: {scope_text(scope)}. Aplica este âmbito a totais, contagens e "
               f"listas, salvo se a pergunta indicar outro.]\n{question}")
    import time
    import uuid
    session = st.session_state.setdefault("usage_session", str(uuid.uuid4()))   # uma por investigação
    qid = usage_log.start_question(question, session_id=session, scope=scope_text(scope))
    t0 = time.time()
    try:
        text = assistant.send(message, question_id=qid)
        usage_log.finish_question(qid, "sem_resposta" if text == assistant.NO_FINAL_ANSWER else "ok",
                                  text, time.time() - t0)
    except Exception as e:
        text = assistant.friendly_error(e)
        usage_log.finish_question(qid, "erro", "", time.time() - t0, error=f"{type(e).__name__}: {e}")
    trace = st.session_state.get("trace", [])
    evidence = [Evidence(e["kind"], e["label"], e.get("detail", ""), e.get("link"))
                for t in trace for e in t.get("evidencia", [])]
    follow_ups = []
    contractors = [e.get("contractor") for t in trace for e in t.get("evidencia", []) if e.get("contractor")]
    if contractors:
        follow_ups = [f"Outros contratos de {contractors[0][:60]}", "Que alertas tem este contrato e porquê?"]
    return Answer(text=text, evidence=evidence, follow_ups=follow_ups, trace=trace)


def new_conversation() -> None:
    import streamlit as st
    for k in ["chat", "chat_model", "trace", "usage_session"]:
        st.session_state.pop(k, None)
