"""Contratação pública — Covilhã. Ponto de entrada da aplicação Streamlit.

Correr localmente:  streamlit run app/app.py
"""
import streamlit as st

import lib

st.set_page_config(page_title="Contratação pública na Covilhã", page_icon=":material/gavel:", layout="wide")

lib.require_db()

# o modelo de pesquisa nos documentos (~1 GB) começa a carregar logo no arranque, em segundo plano
import rag  # noqa: E402

_path, _ = lib.db_snapshot()
if rag.has_index(lib._connection(_path).cursor()):
    rag.warm_up()

# O Assistente é a primeira página e a página por defeito
pg = st.navigation([
    st.Page("views/assistant.py", title="Assistente", icon=":material/forum:", default=True),
    st.Page("views/alerts.py", title="Alertas", icon=":material/flag:"),
    st.Page("views/contracts.py", title="Contratos", icon=":material/description:"),
    st.Page("views/network.py", title="Rede de relações", icon=":material/hub:"),
    st.Page("views/overview.py", title="Visão geral", icon=":material/bar_chart:"),
    st.Page("views/pipeline.py", title="Estado do pipeline", icon=":material/database:"),
])

# Os filtros globais (Ano, Entidade, Tipo de procedimento) não aparecem no Assistente:
# lá o âmbito escolhe-se no próprio composer.
all_contracts = lib.contracts()
if pg.title == "Assistente":
    st.session_state["filtered"] = all_contracts
else:
    st.session_state["filtered"] = lib.sidebar_filters(all_contracts)

pg.run()

_, updated = lib.db_snapshot()
st.sidebar.divider()
st.sidebar.caption(
    "Projeto de mestrado · UBI\n\n"
    "Dados: portal BASE (IMPIC), via dados.gov.pt"
    + (f"\n\nBase de dados atualizada em {updated.tz_convert('Europe/Lisbon'):%d/%m/%Y %H:%M}"
       if updated is not None and pg.title != "Assistente" else "")
)
