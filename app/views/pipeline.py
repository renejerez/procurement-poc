import plotly.express as px
import streamlit as st

import lib

st.title("Estado do pipeline")
st.markdown(
    "Progresso das três etapas do pipeline diário: **(1) recolha** dos contratos (dados abertos do BASE), "
    "**(2) armazenamento** em DuckDB e **(3) extração de texto** das peças do procedimento."
)

_, updated = lib.db_snapshot()
general = lib.query("""
    SELECT COUNT(*) AS contracts, MIN(publication_date) AS first_pub, MAX(publication_date) AS last_pub,
           MAX(loaded_at) AS loaded_at,
           COUNT(*) FILTER (WHERE documents_link IS NOT NULL) AS with_link
    FROM contracts
""").iloc[0]
entities = lib.query("SELECT COUNT(*) AS n FROM entities")["n"].iloc[0]

st.subheader("1–2 · Contratos e armazenamento")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Contratos na base", lib.num(general["contracts"]))
m2.metric("Entidades (por NIF)", lib.num(entities))
m3.metric("Último contrato publicado", f"{general['last_pub']:%d/%m/%Y}", help=f"Primeiro: {general['first_pub']:%d/%m/%Y}")
m4.metric("Último carregamento", f"{general['loaded_at']:%d/%m/%Y %H:%M}" if general["loaded_at"] == general["loaded_at"] else "—")

st.subheader("3 · Documentos")
if not (lib.has_table("document_fetch_log") and lib.has_table("documents")):
    st.info("O notebook 03 ainda não foi executado sobre esta base de dados.")
    st.stop()

log = lib.query("""
    SELECT platform,
           CASE WHEN status LIKE 'error%' THEN 'erro' ELSE status END AS status,
           COUNT(*) AS contracts, SUM(files) AS files
    FROM document_fetch_log GROUP BY ALL
""")
fetched = int(log["contracts"].sum())
ok = int(log.loc[log["status"] == "ok", "contracts"].sum())

m1, m2, m3 = st.columns(3)
m1.metric("Contratos com link para as peças", lib.num(general["with_link"]))
m2.metric("Já processados", f"{lib.num(fetched)} ({fetched / max(general['with_link'], 1):.0%})")
m3.metric("Com documentos obtidos", lib.num(ok))
st.progress(min(fetched / max(general["with_link"], 1), 1.0))

labels = {"ok": "com documentos", "no_documents": "sem documentos", "unsupported_platform": "plataforma não suportada",
          "erro": "erro (será repetido)"}
log["Estado"] = log["status"].map(labels).fillna(log["status"])
fig = px.bar(log, x="contracts", y="platform", color="Estado", orientation="h",
             labels={"contracts": "Contratos", "platform": ""})
st.plotly_chart(lib.style_fig(fig, 260))

docs = lib.query("""
    SELECT file_type AS tipo,
           CASE WHEN status LIKE 'not_extracted%' THEN 'não extraído' WHEN status = 'needs_ocr' THEN 'sem texto'
                ELSE 'extraído' END AS estado,
           COUNT(*) AS ficheiros, SUM(pages) AS paginas, SUM(pages_ocr) AS paginas_ocr,
           SUM(pages_without_text) AS paginas_sem_texto,
           ROUND(SUM(size_bytes) / 1e6, 1) AS mb, SUM(length(text)) AS caracteres
    FROM documents GROUP BY ALL ORDER BY ficheiros DESC
""")
st.markdown("**Ficheiros por tipo**")
st.dataframe(docs, hide_index=True, column_config={
    "tipo": "Tipo", "estado": "Estado", "ficheiros": "Ficheiros", "paginas": "Páginas", "paginas_ocr": "Páginas por OCR",
    "paginas_sem_texto": "Páginas sem texto", "mb": "MB", "caracteres": st.column_config.NumberColumn("Caracteres", format="localized")})

recent = lib.query("""
    SELECT CAST(fetched_at AS DATE) AS dia, COUNT(*) AS contratos FROM document_fetch_log GROUP BY 1 ORDER BY 1
""")
if len(recent) > 1:
    st.markdown("**Contratos processados por dia**")
    st.plotly_chart(lib.style_fig(px.bar(recent, x="dia", y="contratos", labels={"dia": "", "contratos": "Contratos"}), 260))
