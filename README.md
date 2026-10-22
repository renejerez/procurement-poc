# procurement-poc

Proof of concept for the dissertation *"A Daily Pipeline and RAG-Based System for
Detecting Anomalies in Public Procurement Contracts"*. It is a simplified,
notebook-first version of [`../procurement-rag-pipeline`](../procurement-rag-pipeline).

Scope for now: **(1) collect** the contract data, **(2) store** it, and **(3) extract the text** of the procedure documents.
RAG and LLM flagging come later.

| notebook | does | output |
|---|---|---|
| [01_collection.ipynb](01_collection.ipynb) | downloads the official BASE contract dataset, filters it to one municipality | `data/raw/contracts_covilha.json` |
| [02_storage.ipynb](02_storage.ipynb) | parses it into a small relational schema and upserts it into DuckDB | `data/procurement.duckdb` |
| [03_documents.ipynb](03_documents.ipynb) | downloads each contract's procedure documents and extracts text from PDF/DOCX (OCR for pages without text) | `data/docs/<contract_id>/`, `documents` table |

Each notebook runs on its own, **in Google Colab or locally**. Its first code cell detects which one it is.

## Running in Google Colab

1. In Google Drive, create the folder `My Drive/procurement-poc/` and upload the three notebooks into it.
   Don't upload `.venv/` (Colab has its own Python), and you don't need to upload `data/` either: the notebooks rebuild it.
2. Open `01_collection.ipynb` in Colab (double-click it in Drive → *Open with Google Colaboratory*) and choose
   *Runtime → Run all*. Approve the Google Drive access prompt.
3. Do the same for `02_storage.ipynb`, then `03_documents.ipynb`.

Where things live in Colab:

| what | where | why |
|---|---|---|
| raw zips, `contracts_<slug>.json`, downloaded documents | `My Drive/procurement-poc/data/` | persistent, written straight to Drive |
| `procurement.duckdb` (master copy) | `My Drive/procurement-poc/data/` | persistent |
| `procurement.duckdb` (working copy) | `/content/work/` | DuckDB is slow and can corrupt on a network drive |

`02` and `03` copy the database from Drive at the start (`pull_db()`) and back at the end (`push_db()`).
`03` also saves it every `PUSH_EVERY` contracts, so a Colab disconnect loses at most that many contracts' work.
Don't run two notebooks that write to the database at the same time.

Colab limits to keep in mind: free sessions stop after ~90 minutes idle or 12 hours, and the machine has ~12 GB RAM and
~100 GB disk (wiped at the end). The free Drive quota is 15 GB; all of Covilhã 2026 needs roughly 2 GB.

**Network check:** Colab runs on Google servers, usually outside Portugal. `03` starts with a cell that checks the
Vortal API, acingov and dados.gov.pt answer from that machine. If one shows *NOT REACHABLE*, run `03` locally instead
(the database on Drive works in both places; see below).

## Running locally (VS Code / Jupyter)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name procurement-poc --display-name "Python (procurement-poc)"
```

Open the notebooks, select the **Python (procurement-poc)** kernel, and run `01`, `02`, then `03`.
Locally everything is read from and written to `./data/`. To share data with Colab, use
[Google Drive for desktop](https://www.google.com/drive/download/) and run the notebooks from the synced
`procurement-poc` folder.

For OCR on Windows, install Tesseract from the [UB Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki)
and tick *Portuguese* under additional languages. Without it, `03` still runs with OCR off.

## Data source

The collection notebook uses IMPIC's open-data export of the BASE portal on
[dados.gov.pt](https://dados.gov.pt/pt/datasets/contratos-publicos-portal-base-impic-contratos-de-2012-a-2026/):
one zipped JSON file per year (2012–2026), refreshed by IMPIC (the files were
republished on 2026-09-27 with contracts up to the previous day).

It does not use the Selenium scraper from the original prototype. As of
2026-09-27, the BASE search page's backend (`POST /Base4/pt/resultados/`) returns
HTTP 404 even inside a real Chrome session, and plain HTTP requests are blocked
by the site's F5 firewall. The open-data files hold the same records, with more
fields (NIFs, all bidders, base price, procedure-documents link), and need no browser.

## Running it daily

Re-run the notebooks in order:
- `01` downloads a year's file again only if IMPIC has republished it (the cached file name holds the `last_modified` stamp)
- `02` upserts on `contract_id`, so re-loading never creates duplicates (the notebook asserts this)
- `03` only fetches contracts not yet in `document_fetch_log` (failed ones are retried), `LIMIT` per run

To backfill more years, change `YEARS` in `01` (one year at a time fits comfortably in Colab's RAM).
To target another municipality, change `MUNICIPALITY` and `SLUG`.

## Current results (Covilhã, 2026)

- 180,386 contracts nationally → **1,260** executed in Covilhã (published 2026-01-07 to 2026-09-25, total €76.7M)
- 926 distinct entities, keyed by NIF: the same body can appear under up to 5 different spellings in the raw data
- 938 contracts link to their procedure documents: Vortal 827, acingov 100, other platforms 11 (not supported yet)

## Documents (notebook 03)

- **Vortal**: the link opens a JavaScript app; the notebook calls the app's public JSON API instead (link → contract-notice id → document list → downloads)
- **acingov**: the link downloads a ZIP of all the documents (1 of the 100 links is a process page that needs a login)
- **OCR**: PDFs are read from their native text layer first. In a 513-page sample only 9 pages had no text, all technical
  drawings. Only those pages go to Tesseract (`por` model, 300 dpi, at most `OCR_MAX_PAGES` per file).
  `pages_ocr` counts OCR'd pages; `pages_without_text` counts pages still empty after OCR; `needs_ocr` means no page has text.
  Section 6 of `03` re-runs extraction on files already on disk, without downloading them again.
- `local_path` is stored relative to the data folder (`docs/<contract_id>/<file>`), so the same database works on Windows and in Colab.
- XLSX/XLS/XML/ZIP files are saved to disk but not extracted yet
- First run: 25 contracts took 85 s and about 38 MB, so all ~938 would take roughly an hour and about 1.5 GB

### Why Tesseract for OCR

| option | cost | verdict |
|---|---|---|
| **Tesseract** (via PyMuPDF) | free, runs in Colab and locally | **used**: good on typed Portuguese text, and OCR is only a fallback here |
| Google Document AI / Cloud Vision | free tier ~1,000 pages/month, then paid; needs a GCP billing account | better on poor scans; worth it only if OCR'd pages turn out to matter |
| Azure Document Intelligence | free tier 500 pages/month | same trade-off |
| Vision LLM (Claude, GPT) | paid per page | best on tables and handwriting; overkill for a fallback |

## How this relates to `msc_final_project.ipynb` (the earlier prototype)

| prototype section | status in this PoC |
|---|---|
| Selenium scraping of the BASE search and detail pages, 2024, one request per day (`covilha.csv`, 1,135 rows) | **replaced** by `01` (open-data files). The scraper no longer works (see *Data source*). To compare, run `01` with `YEARS = [2024]`; note that the prototype filtered by **signing date**, while each open-data file holds a publication year |
| Price parsing (`"1.234,56 €"` → float), duplicate column removal | **replaced** by `02` (prices arrive as numbers; typed schema) |
| Charts: top spenders, spend by procedure type, authority–contractor network | **to redo** as queries on `v_contracts`. Group by NIF, not name, or one body splits into several nodes |
| Tesseract OCR on every page (`covilha_text.csv`) | **replaced** by `03`: native text first, OCR only for empty pages. The prototype set `lang='por'` but never passed it to `image_to_string`, so it OCR'd Portuguese with the English model (hence errors like *FORMACAQ*, *Vestudrio*) |
| OpenAI name extraction (`gpt-3.5-turbo`, `openai.ChatCompletion`) | **later stage**. That API was removed in `openai>=1.0` and the model is retired, so it needs rewriting |
| RAG demo: Wikipedia COVID pages + Redis vector search | **later stage**. It's a generic template; the RAG step will index the `documents` text instead |
| Streamlit app, presentation video | **later stage** |
