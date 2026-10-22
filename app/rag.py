"""Indexação e pesquisa no texto dos documentos (RAG).

Usado pelo notebook 05 (para criar o índice) e pela página "Assistente" da aplicação (para pesquisar).
Ter uma só implementação garante que a pergunta e os documentos são transformados em vetores da mesma forma.

Tabelas criadas na procurement.duckdb:
- rag_chunks   uma linha por passagem de texto: (text_hash, chunk_no) + texto + vetor
- rag_doc_map  que contratos/ficheiros têm esse texto (o mesmo caderno de encargos aparece em vários contratos)
- rag_meta     modelo e parâmetros usados, para não misturar vetores de modelos diferentes
"""
import re
import threading
import time

import numpy as np

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"   # multilingue, 768 dimensões, ~1 GB
EMBED_BACKEND = "sentence-transformers"
EMBED_DIM = 768
CHUNK_CHARS = 1200      # ~300 tokens, abaixo do limite de 512 do modelo
OVERLAP_CHARS = 200

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS rag_chunks (
    chunk_id  BIGINT PRIMARY KEY,
    text_hash VARCHAR,
    chunk_no  INTEGER,
    text      VARCHAR,
    embedding FLOAT[{EMBED_DIM}]
);
CREATE TABLE IF NOT EXISTS rag_doc_map (
    text_hash   VARCHAR,
    contract_id BIGINT,
    file_name   VARCHAR
);
CREATE TABLE IF NOT EXISTS rag_meta (key VARCHAR PRIMARY KEY, value VARCHAR);
"""


# ─────────────────────────────── divisão em passagens ───────────────────────────────

def clean(text):
    text = text.replace("\r", "")
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[_.\-=]{6,}", " ", text)          # linhas de pontos dos índices, sublinhados de formulários
    return text.strip()


def chunk_text(text, size=CHUNK_CHARS, overlap=OVERLAP_CHARS):
    """Janelas de ~size caracteres que terminam num fim de parágrafo/frase/palavra, com sobreposição."""
    text = clean(text)
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text[start:end]
            for sep in ["\n\n", "\n", ". ", " "]:
                cut = window.rfind(sep)
                if cut > size * 0.5:
                    end = start + cut + len(sep)
                    break
        piece = text[start:end].strip()
        if len(piece) >= 40:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
        nxt = text.find(" ", start)                     # não começar a meio de uma palavra
        if 0 <= nxt < end:
            start = nxt + 1
    return chunks


# ─────────────────────────────── vetores ───────────────────────────────

_MODEL_LOCK = threading.Lock()
_MODEL = []


def _load_model():
    """O modelo corre na GPU quando existe (Colab: Runtime > Change runtime type > T4 GPU), senão no CPU."""
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(2)          # o Colab gratuito tem 2 CPUs; mais threads só competem com o DuckDB
    print(f"a carregar o modelo de pesquisa {EMBED_MODEL} ({device})", flush=True)
    model = SentenceTransformer(EMBED_MODEL, device=device)
    print("modelo de pesquisa carregado", flush=True)
    return model


def _model():
    """Carrega o modelo uma única vez por processo, mesmo que várias threads o peçam ao mesmo tempo."""
    if not _MODEL:
        with _MODEL_LOCK:
            if not _MODEL:
                _MODEL.append(_load_model())
    return _MODEL[0]


def model_loaded():
    return bool(_MODEL)


_WARMUP = []


def _warm_up_work():
    t0 = time.time()
    embed(["aquecimento"])
    print(f"modelo de pesquisa pronto ({time.time() - t0:.1f} s)", flush=True)
    try:                                   # instala já a extensão de pesquisa por palavras-chave (descarga única)
        import duckdb
        duckdb.connect().execute("INSTALL fts")
        print("extensão fts instalada", flush=True)
    except Exception as e:
        print(f"extensão fts indisponível ({type(e).__name__}); a pesquisa por palavras-chave usa o modo simples", flush=True)


def warm_up():
    """Prepara em segundo plano (uma vez por processo) o modelo e a extensão fts, para não atrasar a primeira pergunta."""
    if not _WARMUP:
        _WARMUP.append(threading.Thread(target=_warm_up_work, daemon=True, name="rag-warmup"))
        _WARMUP[0].start()


def device():
    return str(_model().device)


def embed(texts, batch_size=None):
    """Lista de textos -> matriz float32 (n, EMBED_DIM) normalizada."""
    model = _model()
    batch_size = batch_size or (128 if model.device.type == "cuda" else 16)
    vecs = model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False)
    return vecs.astype(np.float32)


# ─────────────────────────────── pesquisa ───────────────────────────────

def has_index(con):
    return con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'rag_chunks'").fetchall()[0][0] > 0


_FTS_LOADABLE = []


def _fts_available(con):
    """True se a extensão fts carrega e o índice BM25 (criado pelo notebook 05) existe."""
    if _FTS_LOADABLE == [False]:          # a extensão não se instala nesta máquina: não voltar a tentar
        return False
    try:
        con.execute("LOAD fts")           # sem INSTALL aqui: a instalação (descarga) é feita no warm_up / notebook 05
        _FTS_LOADABLE[:] = [True]
    except Exception:
        _FTS_LOADABLE[:] = [False]
        return False
    return con.execute("SELECT COUNT(*) FROM information_schema.schemata "
                       "WHERE schema_name = 'fts_main_rag_chunks'").fetchall()[0][0] > 0


STOPWORDS = set("""a o as os um uma uns umas de do da dos das em no na nos nas por para com sem sob sobre que qual quais
quem como onde quando e ou se é são foi ser ao aos à às pelo pela pelos pelas este esta estes estas esse essa isso isto
the of and to in for contrato contratos""".split())


def _keyword_terms(question):
    words = re.findall(r"[\wÀ-ÿ\.\-/º]+", question.lower())
    return [w.strip(".-") for w in words if len(w.strip(".-")) > 2 and w not in STOPWORDS][:8]


# BM25 calculado de uma só vez a partir das tabelas do índice fts (mesma fórmula e parâmetros k=1.2, b=0.75
# da função match_bm25 do DuckDB, que dá resultados idênticos). A função original é chamada passagem a passagem
# e, no DuckDB 2.0, demorava mais de 60 s; esta versão lê só as passagens que contêm os termos da pergunta.
BM25_SQL = """
WITH tokens AS (
    SELECT DISTINCT stem(unnest(fts_main_rag_chunks.tokenize(?)), 'portuguese') AS t
),
qterms AS (
    SELECT t.docid, t.termid
    FROM fts_main_rag_chunks.terms t
    JOIN fts_main_rag_chunks.dict d ON d.termid = t.termid
    JOIN tokens ON d.term = tokens.t
),
term_tf AS (SELECT docid, termid, COUNT(*) AS tf FROM qterms GROUP BY ALL),
stats AS (SELECT num_docs, avgdl FROM fts_main_rag_chunks.stats)
SELECT docs.name AS chunk_id,
       SUM(log(((stats.num_docs - dict.df + 0.5) / (dict.df + 0.5)) + 1)
           * ((tf * (1.2 + 1)) / (tf + 1.2 * ((1 - 0.75) + 0.75 * (docs.len / stats.avgdl))))) AS score
FROM term_tf
JOIN fts_main_rag_chunks.docs docs ON docs.docid = term_tf.docid
JOIN fts_main_rag_chunks.dict dict ON dict.termid = term_tf.termid
CROSS JOIN stats
WHERE TRUE {scope}
GROUP BY docs.name
ORDER BY score DESC, chunk_id
LIMIT ?
"""


class _StepLog:
    """Escreve no log o tempo de cada passo da pesquisa, para se perceber onde demora."""

    def __init__(self, question):
        self.t = time.time()
        print(f"  pesquisa: {question[:60]!r}", flush=True)

    def __call__(self, step):
        now = time.time()
        print(f"  pesquisa: {step} ({now - self.t:.2f} s)", flush=True)
        self.t = now


def search(con, question, contract_id=None, k=8, candidates=40):
    """Pesquisa híbrida: vetores (semântica) + palavras-chave (BM25), combinadas por Reciprocal Rank Fusion.

    Devolve uma lista de dicts: chunk_id, text, score, sources=[(contract_id, file_name), ...]
    """
    scope_sql, scope_params = "", []
    if contract_id is not None:
        scope_sql = "AND c.text_hash IN (SELECT text_hash FROM rag_doc_map WHERE contract_id = ?)"
        scope_params = [int(contract_id)]

    log = _StepLog(question)
    qvec = embed([question])[0].tolist()
    log("vetor da pergunta")
    dense = con.execute(f"""
        SELECT c.chunk_id FROM rag_chunks c WHERE TRUE {scope_sql}
        ORDER BY array_cosine_similarity(c.embedding, ?::FLOAT[{EMBED_DIM}]) DESC LIMIT ?
    """, scope_params + [qvec, candidates]).fetchall()
    log("pesquisa semântica")

    use_fts = _fts_available(con)
    log("extensão fts" + ("" if use_fts else " (indisponível: modo simples)"))
    keyword = None
    if use_fts:
        try:
            keyword = con.execute(BM25_SQL.format(
                scope="AND docs.name IN (SELECT c.chunk_id FROM rag_chunks c WHERE TRUE " + scope_sql + ")" if scope_sql else ""),
                [question] + scope_params + [candidates]).fetchall()
            keyword = [(cid,) for cid, _ in keyword]
        except Exception as e:                              # ex.: memória insuficiente: usar o modo simples
            print(f"  pesquisa: BM25 falhou ({type(e).__name__}: {str(e)[:120]}); a usar o modo simples", flush=True)
            keyword = None
    if keyword is None:                                     # sem a extensão fts: conta termos presentes
        terms = _keyword_terms(question)
        if terms:
            score = " + ".join(["(CASE WHEN c.text ILIKE ? THEN 1 ELSE 0 END)"] * len(terms))
            keyword = con.execute(f"""
                SELECT chunk_id FROM (SELECT c.chunk_id, {score} AS s FROM rag_chunks c WHERE TRUE {scope_sql})
                WHERE s > 0 ORDER BY s DESC, chunk_id LIMIT ?
            """, [f"%{t}%" for t in terms] + scope_params + [candidates]).fetchall()
        else:
            keyword = []

    log("pesquisa por palavras-chave")

    fused = {}
    for ranking in (dense, keyword):
        for rank, (cid,) in enumerate(ranking):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + rank)
    top = sorted(fused, key=fused.get, reverse=True)[:k]
    if not top:
        return []

    rows = con.execute(f"""
        SELECT c.chunk_id, c.text, c.chunk_no,
               LIST(DISTINCT {{'contract_id': m.contract_id, 'file_name': m.file_name}}) AS sources
        FROM rag_chunks c JOIN rag_doc_map m USING (text_hash)
        WHERE c.chunk_id IN ({",".join("?" * len(top))})
          {"AND m.contract_id = ?" if contract_id is not None else ""}
        GROUP BY ALL
    """, top + scope_params).fetchall()
    log("textos e fontes")
    by_id = {r[0]: r for r in rows}
    out = []
    for cid in top:
        if cid in by_id:
            _, text, chunk_no, sources = by_id[cid]
            out.append({"chunk_id": cid, "chunk_no": chunk_no, "text": text, "score": round(fused[cid], 4),
                        "sources": [(s["contract_id"], s["file_name"]) for s in sources]})
    return out
