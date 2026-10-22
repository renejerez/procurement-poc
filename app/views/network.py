import networkx as nx
import plotly.graph_objects as go
import streamlit as st

import lib

st.title("Rede de relações")
st.caption("Ligações entre entidades adjudicantes e adjudicatários, agrupadas por NIF. "
           "A espessura da ligação é proporcional ao valor contratado.")

df = st.session_state["filtered"]
if df.empty:
    st.info("Nenhum contrato corresponde aos filtros.")
    st.stop()

c1, c2 = st.columns(2)
top_n = c1.slider("Ligações a mostrar (as de maior valor)", 10, 300, 80, 10)
min_contracts = c2.slider("Mínimo de contratos por ligação", 1, 10, 1)

edges = (df.groupby(["authority_nif", "authority", "contractor_nif", "contractor"], as_index=False)
           .agg(n=("contract_id", "size"), value=("contract_price", "sum")))
edges = edges[edges["n"] >= min_contracts].nlargest(top_n, "value")
if edges.empty:
    st.info("Nenhuma ligação com estes critérios.")
    st.stop()

G = nx.Graph()
for e in edges.itertuples():
    a, k = f"A:{e.authority_nif}", f"K:{e.contractor_nif}"
    G.add_node(a, name=e.authority, side="Entidade adjudicante")
    G.add_node(k, name=e.contractor, side="Adjudicatário")
    G.add_edge(a, k, n=e.n, value=e.value)
pos = nx.spring_layout(G, seed=7, k=1.2 / max(len(G) ** 0.5, 1), weight=None)
strength = {n: sum(d["value"] for *_, d in G.edges(n, data=True)) for n in G}

fig = go.Figure()
vmax = edges["value"].max()
for u, v, d in G.edges(data=True):
    fig.add_trace(go.Scatter(x=[pos[u][0], pos[v][0]], y=[pos[u][1], pos[v][1]], mode="lines",
                             line=dict(width=0.5 + 8 * d["value"] / vmax, color="rgba(120,120,120,0.45)"),
                             hoverinfo="skip", showlegend=False))
for side, color, symbol in [("Entidade adjudicante", "#1f6feb", "square"), ("Adjudicatário", "#e8743b", "circle")]:
    nodes = [n for n, d in G.nodes(data=True) if d["side"] == side]
    smax = max(strength.values())
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers", name=side,
        marker=dict(size=[8 + 30 * (strength[n] / smax) ** 0.5 for n in nodes], color=color, symbol=symbol,
                    line=dict(width=1, color="white")),
        text=[f"<b>{G.nodes[n]['name']}</b><br>NIF {n[2:]}<br>{lib.eur(strength[n])} · {G.degree(n)} ligações"
              for n in nodes],
        hovertemplate="%{text}<extra></extra>"))
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)
st.plotly_chart(lib.style_fig(fig, 680))

st.subheader("Ligações")
st.dataframe(edges.drop(columns=["authority_nif", "contractor_nif"]).sort_values("value", ascending=False),
             hide_index=True,
             column_config={"authority": "Entidade adjudicante", "contractor": "Adjudicatário",
                            "n": "Contratos", "value": st.column_config.NumberColumn("Valor (€)", format="localized")})
