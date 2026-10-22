import pandas as pd
import plotly.express as px
import streamlit as st

import lib

st.title("Alertas de risco")
st.markdown(
    "Indicadores simples, baseados em regras, calculados sobre os dados do BASE. "
    "**Um alerta não é prova de irregularidade**: assinala contratos que merecem uma leitura mais atenta. "
    "Esta é a primeira camada do sistema de deteção; a análise dos documentos com RAG/LLM virá depois."
)

with st.expander("Parâmetros dos indicadores"):
    p1, p2, p3, p4 = st.columns(4)
    params = {
        "pct": p1.slider("“Logo abaixo do limiar” a partir de", 0.70, 0.99, 0.90, 0.01, format="%.2f"),
        "slip": p2.slider("Derrapagem acima de", 0.05, 1.0, 0.20, 0.05, format="%.2f"),
        "days": p3.slider("Publicação tardia: dias", 5, 120, 30, 5),
        "share": p4.slider("Concentração: quota mínima", 0.10, 0.90, 0.25, 0.05, format="%.2f"),
    }
    st.caption(
        "Limiares do CCP usados (sem IVA) — até 30/09/2026: ajuste direto 20.000 € (bens/serviços) e 30.000 € (empreitadas); "
        "consulta prévia 75.000 € e 150.000 €. Desde 1/10/2026 (DL n.º 177/2026): ajuste direto 75.000 € e 150.000 €; "
        "consulta prévia 130.000 € e 1.000.000 €. O regime é escolhido pela data da decisão de adjudicação "
        "(ou, na falta dela, da celebração), uma aproximação à data de início do procedimento."
    )

df = lib.compute_flags(st.session_state["filtered"], **params)
if df.empty:
    st.info("Nenhum contrato corresponde aos filtros.")
    st.stop()

summary = []
for key in lib.FLAGS:
    hit = df[df[key]]
    summary.append({"key": key, "Indicador": lib.flag_label(key, params), "Contratos": len(hit),
                    "Valor": hit["contract_price"].sum(), "Descrição": lib.flag_help(key, params)})
summary = pd.DataFrame(summary)

m1, m2, m3 = st.columns(3)
flagged = df[df["n_flags"] > 0]
m1.metric("Contratos com pelo menos um alerta", f"{lib.num(len(flagged))} de {lib.num(len(df))}")
m2.metric("Valor desses contratos", lib.eur_short(flagged["contract_price"].sum()))
m3.metric("Com 2 ou mais alertas", lib.num((df["n_flags"] >= 2).sum()))

fig = px.bar(summary.sort_values("Contratos"), x="Contratos", y="Indicador", orientation="h", text="Contratos")
st.plotly_chart(lib.style_fig(fig, 340))
st.dataframe(summary.drop(columns="key"), hide_index=True,
             column_config={"Valor": lib.EURO, "Descrição": st.column_config.TextColumn(width="large")})

st.subheader("Contratos assinalados")
choice = st.selectbox("Indicador", ["Todos (ordenados por n.º de alertas)"] + [lib.flag_label(k, params) for k in lib.FLAGS])
if choice.startswith("Todos"):
    view = flagged
else:
    key = next(k for k in lib.FLAGS if lib.flag_label(k, params) == choice)
    st.caption(lib.flag_help(key, params))
    view = df[df[key]]

view = view.assign(alertas=view.apply(lambda r: " · ".join(lib.flags_for(r, params)), axis=1))
view = view.sort_values(["n_flags", "contract_price"], ascending=False)
st.dataframe(
    view[["contract_id", "publication_date", "object", "authority", "contractor", "procedure_type",
          "contract_price", "limit", "bidders", "n_flags", "alertas", "base_link"]],
    hide_index=True, height=480,
    column_config={
        "contract_id": st.column_config.NumberColumn("ID", format="%d"),
        "publication_date": st.column_config.DateColumn("Publicação", format="DD/MM/YYYY"),
        "object": st.column_config.TextColumn("Objeto", width="large"),
        "authority": "Entidade adjudicante", "contractor": "Adjudicatário", "procedure_type": "Procedimento",
        "contract_price": st.column_config.NumberColumn("Preço contratual (€)", format="localized"),
        "limit": st.column_config.NumberColumn("Limiar aplicável (€)", format="localized"),
        "bidders": st.column_config.NumberColumn("Concorrentes", format="%d"),
        "n_flags": st.column_config.NumberColumn("N.º alertas", format="%d"),
        "alertas": st.column_config.TextColumn("Alertas", width="large"),
        "base_link": st.column_config.LinkColumn("BASE", display_text="abrir"),
    },
)
st.download_button("Descarregar CSV", view.drop(columns=["base_link"]).to_csv(index=False).encode("utf-8-sig"),
                   file_name="alertas.csv", mime="text/csv")
