"""Contratação pública em Portugal. Ponto de entrada da aplicação Streamlit.

A base de dados tem os contratos de todo o país; cada página trabalha sobre um âmbito (distrito, concelho, anos)
escolhido na barra lateral, carregado do DuckDB já filtrado.

Correr localmente:  streamlit run app/app.py
"""
import streamlit as st

import lib

st.set_page_config(page_title="Contratação pública em Portugal", page_icon=":material/gavel:", layout="wide")

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

# O Assistente escolhe o âmbito no próprio composer; o Estado do pipeline mostra a base inteira (só agregados SQL).
# As outras páginas recebem os contratos do âmbito da barra lateral, já filtrados pela entidade e pelo procedimento.
if pg.title in ("Assistente", "Estado do pipeline"):
    st.session_state["filtered"] = None
    st.session_state["place"] = None
else:
    place = lib.place_selector()
    st.session_state["place"] = place
    try:
        scoped = lib.contracts_in_scope(place)
    except lib.ScopeTooLarge as e:
        st.warning(f"**{place.label}** tem {lib.num(e.n)} contratos, demasiados para carregar de uma vez "
                   f"(máximo {lib.num(lib.MAX_SCOPE_ROWS)}). Escolha um distrito, um concelho ou menos anos "
                   "na barra lateral. Para perguntas sobre todo o país, use o Assistente.")
        st.stop()
    st.session_state["filtered"] = lib.sidebar_filters(scoped)

pg.run()

_, updated = lib.db_snapshot()
st.sidebar.divider()
st.sidebar.caption(
    "Projeto de mestrado · UBI\n\n"
    "Dados: portal BASE (IMPIC), via dados.gov.pt"
    + (f"\n\nBase de dados atualizada em {updated.tz_convert('Europe/Lisbon'):%d/%m/%Y %H:%M}"
       if updated is not None and pg.title != "Assistente" else "")
)
