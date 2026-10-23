"""
Assistente — página inicial da aplicação.

Estado vazio: pergunta "O que quer investigar?", composer com âmbito, três formas de começar e o Briefing.
Com conversa: mensagens com Evidência, passos em direto e composer fixo em baixo.
"""
from __future__ import annotations

import html
import time

import pandas as pd
import streamlit as st

import assistant
import lib
import rag
import start_adapters as ad

# --------------------------------------------------------------------------
# Estilo (tokens do design system "Contratação Pública").
# O tema é sempre claro (config.toml), por isso não há variante escura destes tokens.
# --------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600&display=swap');
:root{
  --surface-000:#f7f6f2; --surface-100:#ffffff; --surface-200:#eeece6;
  --border:#e2dfd7; --border-strong:#8b877d; --ink:#18202b; --ink-muted:#5a6170;
  --accent:#1f4e8c; --accent-soft:#e6eef8; --flag:#9a5b00; --flag-soft:#fbf0dc; --fresh:#2f7a4b; --danger:#b3261e;
}
.block-container{max-width:840px; padding-top:3.5rem;}
[data-testid="stBottomBlockContainer"]{max-width:840px;}
.cp-hr{border:0; border-top:1px solid var(--border); margin:.25rem 0 .5rem;}
.cp-q{font-family:'Source Serif 4',Georgia,serif; font-weight:600; font-size:34px; line-height:42px;
      letter-spacing:-.3px; text-align:center; color:var(--ink); margin:2.5rem 0 1.25rem;}
.cp-h{font-family:'Source Serif 4',Georgia,serif; font-weight:600; font-size:17px; color:var(--ink);}
.cp-meta{font-size:13px; color:var(--ink-muted);}
.cp-ind{font-weight:600; color:var(--flag); font-size:15px; line-height:22px;}
.cp-who{font-size:14px; color:var(--ink-muted);}
.cp-amt{font-weight:600; font-variant-numeric:tabular-nums; text-align:right; padding-top:10px; color:var(--ink);}
.cp-kpi{text-align:center; font-size:13px; color:var(--ink-muted); margin-top:.75rem;}
.cp-kpi b{color:var(--ink);}
.cp-disc{text-align:center; font-size:13px; color:var(--ink-muted); margin-top:2rem;}
.cp-fresh{font-size:13px; color:var(--ink-muted); display:flex; gap:8px; align-items:flex-start;}
.cp-dot{width:8px; height:8px; border-radius:999px; margin-top:5px; flex:none;}
.cp-ev-k{font-size:12px; color:var(--ink-muted); text-transform:uppercase; letter-spacing:.4px;}
.cp-n{display:inline-block; background:var(--accent-soft); color:var(--accent); font-weight:600; font-size:12px;
      padding:0 6px; border-radius:6px; margin-right:6px;}
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Estado
# --------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("messages", [])      # [{"role", "content", "evidence"?, "follow_ups"?, "trace"?}]
ss.setdefault("pending", None)     # pergunta enviada por um botão
opts = ad.scope_options()


def default_scope() -> ad.Scope:
    return ad.Scope(opts["municipios"][0], opts["anos"][0] if opts["anos"] else None, None)


ss.setdefault("scope", default_scope())   # guardado fora dos widgets: estes desaparecem durante a conversa


def queue(prompt: str) -> None:
    ss.pending = prompt


def new_chat() -> None:
    ss.messages = []
    ss.pending = None
    ad.new_conversation()


def eur(v: float) -> str:
    return lib.eur(v, 2)


# --------------------------------------------------------------------------
# Barra lateral: nova investigação + frescura dos dados
# --------------------------------------------------------------------------
with st.sidebar:
    st.button("Nova investigação", icon=":material/edit_square:", on_click=new_chat, use_container_width=True)
    run = ad.last_pipeline_run()
    if run is not None:
        hours = (pd.Timestamp.now(tz="UTC") - run).total_seconds() / 3600
        color = "var(--fresh)" if hours <= 24 else ("var(--flag)" if hours <= 48 else "var(--danger)")
        st.markdown(
            f'<div class="cp-fresh"><span class="cp-dot" style="background:{color}"></span>'
            f'<span>Dados BASE (IMPIC)<br>atualizado {run.tz_convert("Europe/Lisbon"):%d/%m/%Y %H:%M}</span></div>',
            unsafe_allow_html=True,
        )

# --------------------------------------------------------------------------
# Pré-requisitos do assistente
# --------------------------------------------------------------------------
HAS_KEY = bool(assistant.api_key())
_path, _ = lib.db_snapshot()
HAS_INDEX = rag.has_index(lib._connection(_path).cursor())


def missing_key_help() -> None:
    st.warning("Falta a chave da API do Gemini: o assistente está desativado. O Briefing e as outras páginas "
               "funcionam na mesma.")
    st.markdown("Crie uma chave gratuita em [Google AI Studio](https://aistudio.google.com/apikey) e guarde-a como "
                "`GEMINI_API_KEY`: no Colab, nos *secrets* (🔑); no Streamlit Cloud, em *Settings → Secrets*.")


# --------------------------------------------------------------------------
# Âmbito
# --------------------------------------------------------------------------
ALL_PROC, ALL_YEARS = "Todos os procedimentos", "Todos os anos"


def scope_bar() -> ad.Scope:
    cur = ss.scope
    c1, c2, c3 = st.columns([1.1, 1, 2.2])
    with c1:
        mun = st.selectbox("Município", opts["municipios"], key="sc_mun", label_visibility="collapsed",
                           index=opts["municipios"].index(cur.municipio) if cur.municipio in opts["municipios"] else 0)
    with c2:
        years = opts["anos"] + [ALL_YEARS]
        ano = st.selectbox("Ano", years, key="sc_ano", label_visibility="collapsed",
                           index=years.index(cur.ano) if cur.ano in years else (0 if opts["anos"] else len(years) - 1))
    with c3:
        procs = [ALL_PROC] + opts["procedimentos"]
        proc = st.selectbox("Procedimento", procs, key="sc_proc", label_visibility="collapsed",
                            index=procs.index(cur.procedimento) if cur.procedimento in procs else 0)
    return ad.Scope(mun, None if ano == ALL_YEARS else int(ano), None if proc == ALL_PROC else proc)


def scope_label(scope: ad.Scope) -> str:
    return f'{scope.municipio} · {scope.ano or "todos os anos"} · {scope.procedimento or "todos os procedimentos"}'


PLACEHOLDER = "Pergunte sobre contratos, entidades ou alertas — ou cole um n.º de contrato ou NIF"

# --------------------------------------------------------------------------
# Estado vazio — página inicial
# --------------------------------------------------------------------------
if not ss.messages and not ss.pending:
    st.markdown('<div class="cp-q">O que quer investigar?</div>', unsafe_allow_html=True)

    with st.container(border=True):
        prompt = st.chat_input(PLACEHOLDER, key="chat_start", disabled=not HAS_KEY)
        scope = scope_bar()
    ss.scope = scope
    if not HAS_KEY:
        missing_key_help()
    elif not HAS_INDEX:
        st.caption("O índice dos documentos ainda não existe (notebook 05): o assistente responde só com os dados "
                   "estruturados.")
    if prompt:
        queue(prompt)
        st.rerun()

    # Três formas de começar
    year = scope.ano or (opts["anos"][0] if opts["anos"] else None)
    in_year = f" em {year}" if year else ""
    w1, w2, w3 = st.columns(3)
    w1.button("Seguir o dinheiro", icon=":material/arrow_forward:", use_container_width=True, disabled=not HAS_KEY,
              on_click=queue, args=(f"Quais são as entidades que mais gastaram{in_year} e com que fornecedores?",))
    w2.button("Explicar um alerta", icon=":material/flag:", use_container_width=True, disabled=not HAS_KEY,
              on_click=queue, args=(f"Quais são os contratos com mais alertas{in_year} e porquê?",))
    w3.button("Comparar", icon=":material/compare_arrows:", use_container_width=True, disabled=not HAS_KEY,
              on_click=queue, args=((f"Compara {year} com {year - 1} por tipo de procedimento." if year else
                                     "Compara os anos disponíveis por tipo de procedimento."),))

    # Briefing (indicadores, não o LLM)
    n_recent, n_flagged, flags, latest = ad.briefing(scope)
    with st.container(border=True):
        until = f" até {latest:%d/%m}" if latest is not None else ""
        st.markdown(f'<div style="display:flex; justify-content:space-between; align-items:baseline; gap:16px">'
                    f'<span class="cp-h">Briefing</span><span class="cp-meta" style="text-align:right">'
                    f'{n_recent} contratos publicados nos últimos {ad.BRIEFING_DAYS} dias{until} · '
                    f'{n_flagged} sinalizados</span></div>', unsafe_allow_html=True)
        if not flags:
            st.caption("Nenhum contrato recente foi sinalizado pelos indicadores.")
        for f in flags:
            st.markdown('<hr class="cp-hr">', unsafe_allow_html=True)
            c1, c2, c3 = st.columns([5.4, 1.5, 1.1], vertical_alignment="center")
            more = f" · +{f.extra_flags}" if f.extra_flags else ""
            c1.markdown(
                f'<div class="cp-ind">⚑ {html.escape(f.indicador)}{more}</div>'
                f'<div class="cp-who">{html.escape(f.entidade)} → {html.escape(f.adjudicatario)} · '
                f'{html.escape(f.procedimento)} · {f.data}</div>',
                unsafe_allow_html=True,
            )
            c2.markdown(f'<div class="cp-amt">{eur(f.valor)}</div>', unsafe_allow_html=True)
            c3.button("Explicar", key=f"exp_{f.contract_id}", use_container_width=True, on_click=queue,
                      disabled=not HAS_KEY,
                      args=(f"Explica o alerta «{f.indicador}» no contrato {f.contract_id} "
                            f"({f.entidade} → {f.adjudicatario}).",))
        st.page_link("views/alerts.py", label="Ver todos os alertas →")

    k = ad.kpis(scope)
    st.markdown(
        f'<div class="cp-kpi"><b>{lib.num(k["contratos"])}</b> contratos · <b>{lib.eur_short(k["valor"])}</b> · '
        f'<b>{k["entidades"]}</b> entidades adjudicantes · <b>{k["adjudicatarios"]}</b> adjudicatários</div>',
        unsafe_allow_html=True,
    )
    _, mid, _ = st.columns([1, 1, 1])
    mid.page_link("views/overview.py", label="Ver visão geral →")

# --------------------------------------------------------------------------
# Conversa
# --------------------------------------------------------------------------
else:
    scope = ss.scope

    def render_answer(msg: dict, idx: int) -> None:
        st.markdown(msg["content"])
        ev = msg.get("evidence") or []
        if ev:
            with st.expander(f"Evidência ({len(ev)})"):
                for i, e in enumerate(ev, 1):
                    st.markdown(f'<span class="cp-n">{i}</span><span class="cp-ev-k">{html.escape(e.kind)}</span>',
                                unsafe_allow_html=True)
                    st.markdown(f"**[{e.label}]({e.link})**" if e.link else f"**{e.label}**")
                    if e.kind == "sql":
                        st.code(e.detail, language="sql")
                    elif e.detail:
                        st.caption(e.detail)
        trace = msg.get("trace") or []
        if trace:
            with st.expander("Como cheguei a esta resposta"):
                for t in trace:
                    st.markdown(f"**{t['ferramenta']}** · {t['resultado']}")
                    st.json(t["argumentos"], expanded=False)
        fu = msg.get("follow_ups") or []
        if fu:
            cols = st.columns(len(fu))
            for j, (c, text) in enumerate(zip(cols, fu)):
                c.button(text, key=f"fu_{idx}_{j}", on_click=queue, args=(text,), use_container_width=True,
                         disabled=not HAS_KEY)

    AVATAR = {"user": ":material/person:", "assistant": ":material/manage_search:"}

    for i, m in enumerate(ss.messages):
        with st.chat_message(m["role"], avatar=AVATAR[m["role"]]):
            if m["role"] == "assistant":
                render_answer(m, i)
            else:
                st.markdown(m["content"])

    prompt = st.chat_input("Faça uma pergunta de seguimento…", key="chat_follow", disabled=not HAS_KEY)
    prompt = prompt or ss.pending
    ss.pending = None

    if prompt and not HAS_KEY:
        missing_key_help()
    elif prompt:
        ss.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=AVATAR["user"]):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar=AVATAR["assistant"]):
            t0 = time.time()
            with st.status("A analisar contratos e documentos…", expanded=True) as status:
                assistant.LIVE = status
                assistant._say("💬 Pergunta enviada ao Gemini")
                try:
                    ans = ad.ask(prompt, scope)
                    status.update(label=f"Concluído em {time.time() - t0:.0f} s", state="complete", expanded=False)
                except Exception as e:
                    ans = ad.Answer(text=assistant.friendly_error(e))
                    status.update(label="Erro", state="error", expanded=False)
                finally:
                    assistant.LIVE = None
            msg = {"role": "assistant", "content": ans.text, "evidence": ans.evidence,
                   "follow_ups": ans.follow_ups, "trace": ans.trace}
            ss.messages.append(msg)
            render_answer(msg, len(ss.messages) - 1)

    st.markdown(
        f'<div class="cp-disc">Âmbito: {html.escape(scope_label(scope))} — '
        'os alertas são indicadores de risco, não provas de irregularidade.</div>',
        unsafe_allow_html=True,
    )
