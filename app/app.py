"""Contratação pública — Covilhã. Ponto de entrada da aplicação Streamlit.

Correr localmente:  streamlit run app/app.py
"""
import streamlit as st

import lib

st.set_page_config(page_title="Contratação Pública · Covilhã", page_icon="📑", layout="wide")

lib.require_db()

# o modelo de pesquisa nos documentos (~1 GB) começa a carregar logo no arranque, em segundo plano
import rag  # noqa: E402

_path, _ = lib.db_snapshot()
if rag.has_index(lib._connection(_path).cursor()):
    rag.warm_up()
all_contracts = lib.contracts()
st.session_state["filtered"] = lib.sidebar_filters(all_contracts)

_, updated = lib.db_snapshot()
st.sidebar.divider()
st.sidebar.caption(
    "Projeto de mestrado · UBI\n\n"
    "Dados: portal BASE (IMPIC), via dados.gov.pt"
    + (f"\n\nBase de dados atualizada em {updated.tz_convert('Europe/Lisbon'):%d/%m/%Y %H:%M}" if updated is not None else "")
)

pages = st.navigation([
    st.Page("views/overview.py", title="Visão geral", icon="📊", default=True),
    st.Page("views/contracts.py", title="Contratos", icon="🔎"),
    st.Page("views/alerts.py", title="Alertas", icon="🚩"),
    st.Page("views/assistant.py", title="Assistente", icon="💬"),
    st.Page("views/network.py", title="Rede de relações", icon="🕸️"),
    st.Page("views/pipeline.py", title="Estado do pipeline", icon="⚙️"),
])
pages.run()
