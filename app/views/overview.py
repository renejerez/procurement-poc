import plotly.express as px
import streamlit as st

import lib

df = st.session_state["filtered"]

st.title("Contratação pública na Covilhã")
st.caption("Contratos públicos com local de execução no concelho da Covilhã, publicados no portal BASE.")

if df.empty:
    st.info("Nenhum contrato corresponde aos filtros.")
    st.stop()

st.caption(f"Publicados entre {df['publication_date'].min():%d/%m/%Y} e {df['publication_date'].max():%d/%m/%Y}.")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Contratos", lib.num(len(df)))
c2.metric("Valor contratual", lib.eur_short(df["contract_price"].sum()), help=lib.eur(df["contract_price"].sum(), 2))
c3.metric("Entidades adjudicantes", lib.num(df["authority_nif"].nunique()))
c4.metric("Adjudicatários", lib.num(df["contractor_nif"].nunique()))

st.subheader("Valor publicado por mês")
monthly = df.groupby(["month", "procedure_type"], as_index=False)["contract_price"].sum()
fig = px.bar(monthly, x="month", y="contract_price", color="procedure_type",
             labels={"month": "", "contract_price": "Valor (€)", "procedure_type": "Procedimento"})
fig.update_xaxes(dtick="M1", tickformat="%m/%Y")
st.plotly_chart(lib.style_fig(fig, 420))

left, right = st.columns(2)
with left:
    st.subheader("Por tipo de procedimento")
    by_proc = (df.groupby("procedure_type")
                 .agg(contratos=("contract_id", "size"), valor=("contract_price", "sum"))
                 .sort_values("valor", ascending=False).reset_index())
    by_proc["% do valor"] = by_proc["valor"] / by_proc["valor"].sum() * 100
    st.dataframe(by_proc.rename(columns={"procedure_type": "Procedimento", "contratos": "Contratos", "valor": "Valor"}),
                 hide_index=True,
                 column_config={"Valor": lib.EURO, "% do valor": st.column_config.ProgressColumn(
                     format="%.1f%%", min_value=0, max_value=100)})
with right:
    st.subheader("Tipo de contrato")
    types = df.assign(t=df["contract_types"].fillna("—").str.split("; ")).explode("t")
    by_type = types.groupby("t", as_index=False)["contract_price"].sum().sort_values("contract_price")
    fig = px.bar(by_type, x="contract_price", y="t", orientation="h", labels={"t": "", "contract_price": "Valor (€)"})
    st.plotly_chart(lib.style_fig(fig, 300))

left, right = st.columns(2)
for col, who, title in [(left, "authority", "Maiores entidades adjudicantes"), (right, "contractor", "Maiores adjudicatários")]:
    with col:
        st.subheader(title)
        top = (df.groupby(who).agg(contratos=("contract_id", "size"), valor=("contract_price", "sum"))
                 .sort_values("valor", ascending=False).head(10).reset_index())
        st.dataframe(top.rename(columns={who: "Entidade", "contratos": "Contratos", "valor": "Valor"}),
                     hide_index=True, column_config={"Valor": lib.EURO})
