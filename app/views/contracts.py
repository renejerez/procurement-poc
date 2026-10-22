import streamlit as st

import lib

df = lib.compute_flags(st.session_state["filtered"], **lib.DEFAULT_PARAMS)
docs_available = lib.has_table("documents")

st.title("Contratos")

c1, c2, c3 = st.columns([3, 1, 1])
text = c1.text_input("Pesquisar no objeto do contrato", placeholder="ex.: iluminação, pavimentação, software...")
in_docs = c2.toggle("Incluir documentos", disabled=not docs_available,
                    help="Procura também no texto extraído das peças do procedimento (notebook 03).")
only_flagged = c3.toggle("Só com alertas")

price_max = float(df["contract_price"].max() or 0)
low, high = st.slider("Valor contratual (€)", 0.0, max(price_max, 1.0), (0.0, max(price_max, 1.0)), step=1000.0, format="%.0f €")

view = df[(df["contract_price"] >= low) & (df["contract_price"] <= high)]
if text:
    hit = view["object"].fillna("").str.contains(text, case=False, regex=False)
    if in_docs:
        ids = lib.query("SELECT DISTINCT contract_id FROM documents WHERE text ILIKE ?", [f"%{text}%"])["contract_id"]
        hit |= view["contract_id"].isin(ids)
    view = view[hit]
if only_flagged:
    view = view[view["n_flags"] > 0]

st.caption(f"{lib.num(len(view))} contratos · {lib.eur(view['contract_price'].sum())}. Clique numa linha para ver o detalhe.")

cols = ["contract_id", "publication_date", "object", "authority", "contractor", "procedure_type",
        "contract_price", "bidders", "n_flags", "base_link"]
table = view.sort_values("publication_date", ascending=False)[cols]
event = st.dataframe(
    table, hide_index=True, height=420, on_select="rerun", selection_mode="single-row",
    column_config={
        "contract_id": st.column_config.NumberColumn("ID", format="%d"),
        "publication_date": st.column_config.DateColumn("Publicação", format="DD/MM/YYYY"),
        "object": st.column_config.TextColumn("Objeto", width="large"),
        "authority": "Entidade adjudicante",
        "contractor": "Adjudicatário",
        "procedure_type": "Procedimento",
        "contract_price": st.column_config.NumberColumn("Preço contratual (€)", format="localized"),
        "bidders": st.column_config.NumberColumn("Concorrentes", format="%d"),
        "n_flags": st.column_config.NumberColumn("Alertas", format="%d"),
        "base_link": st.column_config.LinkColumn("BASE", display_text="abrir"),
    },
)

if not event.selection.rows:
    st.stop()

row = table.iloc[event.selection.rows[0]]
c = view.set_index("contract_id").loc[row["contract_id"]]
cid = int(row["contract_id"])

st.divider()
st.subheader(c["object"] or f"Contrato {cid}")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Preço contratual", lib.eur(c["contract_price"]))
m2.metric("Preço base", lib.eur(c["base_price"]))
m3.metric("Concorrentes", "—" if c["bidders"] != c["bidders"] else int(c["bidders"]))
m4.metric("Prazo de execução", f"{int(c['execution_days'])} dias" if c["execution_days"] == c["execution_days"] else "—")

flags = lib.flags_for(c, lib.DEFAULT_PARAMS)
if flags:
    st.warning("**Alertas:** " + " · ".join(flags))

left, right = st.columns(2)
with left:
    st.markdown("  \n".join([
        f"**Entidade adjudicante:** {c['authority']} (NIF {c['authority_nif']})",
        f"**Adjudicatário:** {c['contractor']} (NIF {c['contractor_nif']})",
        f"**Procedimento:** {c['procedure_type']}",
        f"**Tipo de contrato:** {c['contract_types'] or '—'}",
        f"**CPV:** {c['cpv_code'] or '—'} {c['cpv_description'] or ''}",
        f"**Fundamentação:** {c['legal_basis'] or '—'}",
    ]))
with right:
    def d(x):
        return "—" if x != x or x is None else f"{x:%d/%m/%Y}"
    links = f"[Ver no portal BASE]({lib.BASE_DETAIL}{cid})"
    if c["documents_link"]:
        links += f" · [Peças do procedimento]({c['documents_link']})"
    st.markdown("  \n".join([
        f"**Celebração:** {d(c['signing_date'])}",
        f"**Publicação no BASE:** {d(c['publication_date'])}",
        f"**Fecho:** {d(c['close_date'])}",
        f"**Preço efetivo:** {lib.eur(c['effective_price'], 2)}",
        "", links,
    ]))

competitors = lib.query("""
    SELECT e.name AS concorrente, p.nif AS nif,
           p.nif IN (SELECT nif FROM contract_parties WHERE contract_id = ? AND role = 'contractor') AS vencedor
    FROM contract_parties p JOIN entities e USING (nif)
    WHERE p.contract_id = ? AND p.role = 'competitor' ORDER BY vencedor DESC, concorrente
""", [cid, cid])
if not competitors.empty:
    st.markdown("**Concorrentes**")
    st.dataframe(competitors, hide_index=True,
                 column_config={"concorrente": "Concorrente", "nif": "NIF", "vencedor": "Adjudicado"})

if docs_available:
    docs = lib.query("""
        SELECT file_name, file_type, status, pages, pages_ocr, text
        FROM documents WHERE contract_id = ? ORDER BY file_name
    """, [cid])
    st.markdown(f"**Documentos do procedimento** ({len(docs)})")
    if docs.empty:
        st.caption("Sem documentos descarregados para este contrato.")
    for _, doc in docs.iterrows():
        info = f"{doc['file_type']} · {doc['status']}"
        if doc["pages"] == doc["pages"] and doc["pages"] is not None:
            info += f" · {int(doc['pages'])} páginas"
            if doc["pages_ocr"]:
                info += f" ({int(doc['pages_ocr'])} por OCR)"
        with st.expander(f"{doc['file_name']} — {info}"):
            if doc["text"]:
                st.text(doc["text"][:20000] + ("\n\n[... texto truncado]" if len(doc["text"]) > 20000 else ""))
            else:
                st.caption("Texto não extraído.")
