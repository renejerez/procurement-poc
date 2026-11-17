"""Acesso aos dados e indicadores de risco, partilhados por todas as páginas da aplicação."""
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).parent
BASE_DETAIL = "https://www.base.gov.pt/Base4/pt/detalhe/?type=contratos&id="

# ─────────────────────────────── base de dados ───────────────────────────────
# De onde vem a base de dados, por ordem:
# 1. PROCUREMENT_DB (variável de ambiente): caminho local (Colab, computador)
# 2. Streamlit Cloud: secrets GITHUB_DATA_REPO + GITHUB_TOKEN -> ficheiro procurement.duckdb da release "dados"
#    do repositório (publicado pela secção 3 do notebook 04; uma release aceita ficheiros até 2 GB)
# 3. data/procurement.duckdb ao lado da pasta app/
# A aplicação lê sempre uma cópia, para nunca bloquear a base enquanto os notebooks escrevem nela.
RELEASE_TAG = "dados"
DB_ASSET = "procurement.duckdb"


def _secret(name):
    value = os.environ.get(name)
    if not value:
        try:
            value = st.secrets.get(name)
        except Exception:
            value = None
    return value


def db_source():
    env = os.environ.get("PROCUREMENT_DB")
    if env:
        return env
    repo = _secret("GITHUB_DATA_REPO") or _secret("GITHUB_REPO")      # os dois nomes funcionam
    if repo:
        repo = repo.strip().removeprefix("https://github.com/").strip("/")
        return f"github-release://{repo}"
    for candidate in [APP_DIR / "data" / "procurement.duckdb", APP_DIR.parent / "data" / "procurement.duckdb"]:
        if candidate.exists():
            return str(candidate)
    return None


DB_PROBLEM = []      # porque é que a base de dados não foi encontrada (mostrado na página de erro)


def _github_release_db(repo, tmp):
    import requests
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    token = _secret("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    DB_PROBLEM.clear()
    r = requests.get(f"https://api.github.com/repos/{repo}/releases/tags/{RELEASE_TAG}", headers=headers, timeout=30)
    if r.status_code in (401, 403):
        DB_PROBLEM.append(f"O GitHub recusou o token (HTTP {r.status_code}). Verifique o secret GITHUB_TOKEN: tem de dar "
                          f"acesso ao repositório {repo} com a permissão Contents (Read-only chega) e não pode ter expirado.")
        return None, None
    if r.status_code == 404:
        repo_ok = requests.get(f"https://api.github.com/repos/{repo}", headers=headers, timeout=30).status_code == 200
        DB_PROBLEM.append(
            f"O repositório {repo} existe, mas ainda não tem a release '{RELEASE_TAG}' com a base de dados. "
            "Corra a secção 3 do notebook 04 até ao fim ('base de dados publicada')." if repo_ok else
            f"Não encontro o repositório '{repo}'. Confirme o nome (formato utilizador/repositório) no secret "
            "GITHUB_DATA_REPO e que o GITHUB_TOKEN tem acesso a esse repositório (num repositório privado, "
            "sem token o GitHub responde como se não existisse).")
        return None, None
    r.raise_for_status()
    asset = next((a for a in r.json().get("assets", []) if a["name"] == DB_ASSET), None)
    if asset is None:
        DB_PROBLEM.append(f"A release '{RELEASE_TAG}' existe mas não tem o ficheiro {DB_ASSET}. "
                          "Corra outra vez a secção 3 do notebook 04.")
        return None, None
    updated = pd.Timestamp(asset["updated_at"])
    local = tmp / f"procurement_gh_{asset['id']}.duckdb"
    if not local.exists():
        part = local.with_suffix(".part")
        # o GitHub redireciona para o armazenamento de ficheiros; o requests não reenvia o token para outro domínio
        with requests.get(asset["url"], headers={**headers, "Accept": "application/octet-stream"},
                          stream=True, timeout=600) as d:
            d.raise_for_status()
            with open(part, "wb") as f:
                for chunk in d.iter_content(1 << 20):
                    f.write(chunk)
        part.replace(local)
        for old in tmp.glob("procurement_gh_*.duckdb"):        # apagar versões antigas (libertar disco)
            if old != local:
                try:
                    old.unlink()
                except OSError:
                    pass
    return str(local), updated


class _NotFound(Exception):
    pass


def db_snapshot():
    """Devolve (caminho da cópia local, data da última atualização), ou (None, None) se não houver base de dados.
    Só os sucessos ficam em cache (10 minutos): depois de corrigir um problema, basta recarregar a página."""
    try:
        return _db_snapshot_cached(db_source())
    except _NotFound:
        return None, None


@st.cache_resource(ttl=600, show_spinner="A obter a base de dados...")
def _db_snapshot_cached(src):
    if not src:
        raise _NotFound()
    tmp = Path(tempfile.gettempdir())
    if src.startswith("github-release://"):
        result = _github_release_db(src[len("github-release://"):], tmp)
        if result[0] is None:
            raise _NotFound()
        return result
    path = Path(src)
    if not path.exists():
        raise _NotFound()
    stat = path.stat()
    local = tmp / f"procurement_{int(stat.st_mtime)}_{stat.st_size}.duckdb"
    if not local.exists():
        shutil.copy2(path, local)
    return str(local), pd.Timestamp(stat.st_mtime, unit="s", tz="UTC")


@st.cache_resource
def _connection(path):
    con = duckdb.connect(path, read_only=True)
    # Sem limite, o DuckDB usa até 80% da memória da máquina; somado ao modelo de pesquisa, isso fez o Colab
    # matar a aplicação. Com limite, consultas grandes usam o disco em vez da memória.
    spill = Path(tempfile.gettempdir()) / "duckdb_spill"
    spill.mkdir(exist_ok=True)
    con.execute(f"SET memory_limit = '1GB'; SET threads = 2; SET temp_directory = '{spill.as_posix()}'")
    return con


@st.cache_data(ttl=600, show_spinner=False)
def _query(path, sql, params):
    return _connection(path).cursor().execute(sql, _thaw(params)).df()


def query(sql, params=()):
    path, _ = db_snapshot()
    return _query(path, sql, tuple(_freeze(p) for p in params))


def has_table(name):
    return not query("SELECT 1 FROM information_schema.tables WHERE table_name = ?", [name]).empty


def require_db():
    path, _ = db_snapshot()
    if path is None:
        if DB_PROBLEM:
            st.error("Base de dados não encontrada. " + DB_PROBLEM[0])
        elif db_source() is None:
            st.error("Base de dados não encontrada: falta o secret GITHUB_DATA_REPO (Streamlit Cloud: Settings → Secrets), "
                     "ou corra os notebooks 01 e 02 / defina PROCUREMENT_DB.")
        else:
            st.error(f"Base de dados não encontrada em {db_source()}. Corra os notebooks 01 e 02.")
        st.caption("Depois de corrigir, recarregue a página.")
        st.stop()


# ─────────────────────────────── formatação ───────────────────────────────

def eur(x, decimals=0):
    if x is None or pd.isna(x):
        return "—"
    s = f"{x:,.{decimals}f}".replace(",", " ").replace(".", ",").replace(" ", ".")
    return f"{s} €"


def eur_short(x):
    """Valor compacto para métricas: 62,8 M€, 950 mil €."""
    if x is None or pd.isna(x):
        return "—"
    if abs(x) >= 1e6:
        return f"{x / 1e6:,.1f} M€".replace(".", ",")
    if abs(x) >= 1e4:
        return f"{x / 1e3:,.0f} mil €".replace(",", ".")
    return eur(x)


def num(x):
    return f"{int(x):,}".replace(",", ".")


def style_fig(fig, height=None):
    fig.update_layout(separators=",.", margin=dict(l=10, r=10, t=40, b=10), legend_title_text="")
    if height:
        fig.update_layout(height=height)
    return fig


EURO = st.column_config.NumberColumn("Valor (€)", format="localized")

# ─────────────────────────────── âmbito (base de dados nacional) ───────────────────────────────
# A base de dados tem todos os contratos de Portugal (2012 em diante, ~2,3 milhões). Nunca se carregam todos para a
# memória: cada página trabalha sobre um âmbito (distrito, concelho, anos) filtrado no DuckDB antes de chegar ao pandas.
DOCS_SCOPE = ["Covilhã"]          # concelhos com documentos indexados (notebook 03: documents.scope no config.yaml)
DEFAULT_DISTRICT = "Castelo Branco"
DEFAULT_MUNICIPALITY = "Covilhã"
ALL = "Todos"
MAX_SCOPE_ROWS = 50_000           # contratos carregados por âmbito; o Streamlit Cloud tem pouca memória e o modelo RAG ocupa ~1 GB


class ScopeTooLarge(Exception):
    def __init__(self, n):
        super().__init__(f"{n} contratos")
        self.n = n


@dataclass(frozen=True)
class Place:
    district: str | None = None           # None = todos os distritos
    municipality: str | None = None       # None = todos os concelhos do distrito
    years: tuple = ()                     # anos de publicação; () = todos

    @property
    def label(self):
        where = self.municipality or (f"distrito de {self.district}" if self.district else "Portugal")
        if not self.years:
            return f"{where} · todos os anos"
        ys = sorted(self.years)
        when = str(ys[0]) if len(ys) == 1 else (f"{ys[0]}–{ys[-1]}" if ys == list(range(ys[0], ys[-1] + 1))
                                                 else ", ".join(map(str, ys)))
        return f"{where} · {when}"


@st.cache_data(ttl=600, show_spinner=False)
def _places(path):
    return _connection(path).cursor().execute("""
        SELECT district, municipality, COUNT(DISTINCT contract_id) AS n
        FROM contract_locations
        WHERE country = 'Portugal' AND district IS NOT NULL AND municipality IS NOT NULL
        GROUP BY ALL ORDER BY district, municipality
    """).df()


def places():
    """Distritos e concelhos com contratos (tabela district, municipality, n)."""
    return _places(db_snapshot()[0])


@st.cache_data(ttl=600, show_spinner=False)
def _years(path):
    df = _connection(path).cursor().execute("""
        SELECT DISTINCT CAST(year(publication_date) AS INTEGER) AS y FROM contracts
        WHERE publication_date IS NOT NULL ORDER BY y DESC
    """).df()
    return df["y"].tolist()


def years_available():
    """Anos de publicação, do mais recente para o mais antigo."""
    return _years(db_snapshot()[0])


def place_ids_sql(place):
    """SQL (e parâmetros) que devolve os contract_id de um âmbito. Um contrato com vários locais conta uma vez."""
    conds, params = [], []
    if place.district or place.municipality:
        loc, lp = [], []
        if place.district:
            loc.append("district = ?")
            lp.append(place.district)
        if place.municipality:
            loc.append("municipality = ?")
            lp.append(place.municipality)
        conds.append(f"contract_id IN (SELECT contract_id FROM contract_locations WHERE {' AND '.join(loc)})")
        params += lp
    if place.years:
        conds.append("list_contains(?::INTEGER[], CAST(year(publication_date) AS INTEGER))")
        params.append(list(place.years))
    where = " AND ".join(conds) or "TRUE"
    return f"SELECT contract_id FROM contracts WHERE {where}", params


@st.cache_data(ttl=600, show_spinner=False)
def _count(path, ids_sql, params):
    return _connection(path).cursor().execute(f"SELECT COUNT(*) FROM ({ids_sql})", list(params)).fetchone()[0]


def count_ids(ids_sql, params=()):
    return _count(db_snapshot()[0], ids_sql, tuple(_freeze(p) for p in params))


def _freeze(p):
    return tuple(p) if isinstance(p, list) else p


def _thaw(params):
    return [list(p) if isinstance(p, tuple) else p for p in params]


@st.cache_data(ttl=600, show_spinner="A carregar contratos...", max_entries=6)
def _load(path, ids_sql, params):
    df = _connection(path).cursor().execute(f"""
        WITH ids AS ({ids_sql}),
        names AS (
            SELECT p.contract_id, p.role,
                   string_agg(p.nif, ' + ' ORDER BY p.nif)  AS nifs,
                   string_agg(e.name, ' + ' ORDER BY p.nif) AS names
            FROM contract_parties p JOIN entities e USING (nif)
            WHERE p.role IN ('authority', 'contractor') AND p.contract_id IN (SELECT contract_id FROM ids)
            GROUP BY ALL
        ),
        bids AS (
            SELECT contract_id, COUNT(*) AS bidders FROM contract_parties
            WHERE role = 'competitor' AND contract_id IN (SELECT contract_id FROM ids) GROUP BY contract_id
        ),
        cpv AS (
            SELECT contract_id, MIN(cpv_code) AS cpv_code, ARG_MIN(cpv_description, cpv_code) AS cpv_description
            FROM contract_cpvs WHERE contract_id IN (SELECT contract_id FROM ids) GROUP BY contract_id
        )
        SELECT c.contract_id, c.publication_date, c.signing_date, c.award_decision_date, c.close_date,
               c.procedure_type, c.contract_types, c.object, c.legal_basis,
               CAST(c.base_price AS DOUBLE)      AS base_price,
               CAST(c.contract_price AS DOUBLE)  AS contract_price,
               CAST(c.effective_price AS DOUBLE) AS effective_price,
               c.execution_days, c.documents_link, c.year,
               a.nifs AS authority_nif, a.names AS authority, k.nifs AS contractor_nif, k.names AS contractor,
               b.bidders, cpv.cpv_code, cpv.cpv_description
        FROM contracts c
        LEFT JOIN names a ON a.contract_id = c.contract_id AND a.role = 'authority'
        LEFT JOIN names k ON k.contract_id = c.contract_id AND k.role = 'contractor'
        LEFT JOIN bids b ON b.contract_id = c.contract_id
        LEFT JOIN cpv ON cpv.contract_id = c.contract_id
        WHERE c.contract_id IN (SELECT contract_id FROM ids)
    """, _thaw(params)).df()
    for col in ["publication_date", "signing_date", "award_decision_date", "close_date"]:
        df[col] = pd.to_datetime(df[col])
    df["authority"] = df["authority"].fillna("(sem entidade)")
    df["contractor"] = df["contractor"].fillna("(sem adjudicatário)")
    df["month"] = df["publication_date"].dt.to_period("M").dt.to_timestamp()
    df["base_link"] = BASE_DETAIL + df["contract_id"].astype(str)
    return df


def load_contracts(ids_sql, params=(), max_rows=MAX_SCOPE_ROWS):
    """Carrega os contratos devolvidos por ids_sql. Recusa (ScopeTooLarge) se forem mais de max_rows."""
    frozen = tuple(_freeze(p) for p in params)
    n = count_ids(ids_sql, frozen)
    if n > max_rows:
        raise ScopeTooLarge(n)
    return _load(db_snapshot()[0], ids_sql, frozen)


def contracts_in_scope(place):
    return load_contracts(*place_ids_sql(place))


def contracts_matching(place, entity_text, max_rows=MAX_SCOPE_ROWS):
    """Contratos do âmbito em que a entidade adjudicante ou o adjudicatário contém entity_text (ou é esse NIF)."""
    ids_sql, params = place_ids_sql(place)
    ids_sql += """ AND contract_id IN (
        SELECT p.contract_id FROM contract_parties p JOIN entities e USING (nif)
        WHERE p.role IN ('authority', 'contractor') AND (e.name ILIKE ? OR p.nif = ?))"""
    return load_contracts(ids_sql, [*params, f"%{entity_text}%", entity_text.strip()], max_rows)


def contracts_for_contract(contract_id):
    """O contrato e os outros contratos da(s) mesma(s) entidade(s) adjudicante(s) no mesmo ano: o contexto de que os
    indicadores precisam (fracionamento e concentração são calculados por entidade)."""
    ids_sql = """
        SELECT DISTINCT p.contract_id FROM contract_parties p JOIN contracts c USING (contract_id)
        WHERE p.role = 'authority'
          AND p.nif IN (SELECT nif FROM contract_parties WHERE contract_id = ? AND role = 'authority')
          AND year(c.publication_date) = (SELECT year(publication_date) FROM contracts WHERE contract_id = ?)
        UNION SELECT ?"""
    return load_contracts(ids_sql, [int(contract_id)] * 3, max_rows=10 * MAX_SCOPE_ROWS)


def _index_of(options, value, default=0):
    return options.index(value) if value in options else default


def place_selector():
    """Âmbito geográfico e temporal (barra lateral), partilhado por todas as páginas exceto o Assistente."""
    st.sidebar.header("Âmbito")
    pl = places()
    districts = sorted(pl["district"].unique().tolist())
    d_opts = [ALL] + districts
    # o valor escolhido fica guardado pela key do widget; o index só define o valor da primeira vez
    district = st.sidebar.selectbox("Distrito", d_opts, key="p_district", index=_index_of(d_opts, DEFAULT_DISTRICT))
    munis = sorted(pl.loc[pl["district"] == district, "municipality"].tolist()) if district != ALL else []
    m_opts = [ALL] + munis
    muni = st.sidebar.selectbox("Concelho", m_opts, key="p_muni", index=_index_of(m_opts, DEFAULT_MUNICIPALITY),
                                disabled=district == ALL)
    years = years_available()
    sel_years = st.sidebar.multiselect("Ano de publicação", years, default=years[:1], key="p_years",
                                       placeholder="Todos os anos")
    return Place(None if district == ALL else district,
                 None if district == ALL or muni == ALL else muni,
                 tuple(sorted(int(y) for y in sel_years)))


def sidebar_filters(df):
    """Filtros dentro do âmbito (barra lateral), aplicados em todas as páginas."""
    st.sidebar.header("Filtros")
    authorities = df.groupby("authority")["contract_price"].sum().sort_values(ascending=False).index.tolist()
    sel_auth = st.sidebar.multiselect("Entidade adjudicante", authorities, key="f_auth",
                                      placeholder="Todas")
    procs = df["procedure_type"].value_counts().index.tolist()
    sel_proc = st.sidebar.multiselect("Tipo de procedimento", procs, key="f_proc", placeholder="Todos")
    out = df
    if sel_auth:
        out = out[out["authority"].isin(sel_auth)]
    if sel_proc:
        out = out[out["procedure_type"].isin(sel_proc)]
    st.sidebar.caption(f"{num(len(out))} de {num(len(df))} contratos no âmbito")
    return out


# ─────────────────────────────── indicadores de risco ───────────────────────────────
# Limiares do CCP (valores sem IVA): (ajuste direto, consulta prévia).
# O DL n.º 177/2026 alterou-os para procedimentos iniciados a partir de 1/10/2026.
REVISION_DATE = pd.Timestamp(2026, 10, 1)
LIMITS = {
    False: {"works": (30_000, 150_000), "goods": (20_000, 75_000)},       # até 30/09/2026
    True: {"works": (150_000, 1_000_000), "goods": (75_000, 130_000)},     # desde 1/10/2026 (DL 177/2026)
}
COMPETITIVE = {"Concurso público", "Concurso limitado por prévia qualificação", "Consulta Prévia"}
DIRECT = "Ajuste Direto Regime Geral"
PRIOR = "Consulta Prévia"

FLAGS = {
    "proposta_unica": ("Proposta única em procedimento concorrencial",
                       "Concurso público, concurso limitado ou consulta prévia em que só houve um concorrente. "
                       "Sinal clássico de concorrência fraca ou de caderno de encargos restritivo."),
    "perto_limiar": ("Valor logo abaixo do limiar",
                     "Ajuste direto ou consulta prévia com valor entre {pct:.0%} e 100% do limiar legal. "
                     "Pode indicar um valor ajustado para evitar um procedimento mais exigente."),
    "acima_limiar": ("Valor acima do limiar do procedimento",
                     "Ajuste direto ou consulta prévia acima do limiar por valor, sem fundamentação em critérios "
                     "materiais (art.º 24.º e seguintes do CCP). Verificar a fundamentação."),
    "fracionamento": ("Possível fracionamento",
                      "Vários ajustes diretos/consultas prévias da mesma entidade ao mesmo adjudicatário, no mesmo ano "
                      "e no mesmo grupo CPV, cada um abaixo do limiar mas cuja soma o ultrapassa."),
    "acima_preco_base": ("Preço contratual acima do preço base",
                         "O preço do contrato excede o preço base anunciado no procedimento."),
    "derrapagem": ("Derrapagem do preço efetivo",
                   "O preço total efetivo (contrato encerrado) supera o preço contratual em mais de {slip:.0%}."),
    "publicacao_tardia": ("Publicação tardia no BASE",
                          "Mais de {days} dias entre a celebração do contrato e a publicação no portal BASE."),
    "concentracao": ("Concentração no adjudicatário",
                     "Um adjudicatário recebe pelo menos {share:.0%} do valor em ajustes diretos e consultas prévias "
                     "de uma entidade (com 3 ou mais contratos)."),
    # ── relações entre empresas (notebook 07: atos societários, tabela alertas_relacoes) ──
    "concorrentes_relacionados": ("Concorrentes relacionados entre si",
                                  "Duas empresas ligadas (sócio ou gerente/administrador em comum, ou uma é sócia ou "
                                  "gestora da outra) concorreram no mesmo procedimento sem o ganharem juntas. "
                                  "A concorrência pode não ser real. Consórcios não são assinalados."),
    "rotacao_relacionados": ("Empresas relacionadas na mesma entidade",
                             "Ajuste direto ou consulta prévia ganho por uma empresa quando uma empresa ligada a ela "
                             "ganhou outro contrato da mesma entidade adjudicante no espaço de um ano. "
                             "Pode indicar trabalho repartido entre empresas do mesmo grupo."),
}
RELATION_FLAGS = ("concorrentes_relacionados", "rotacao_relacionados")
LINK_LABELS = {"shared_person": "pessoa em comum", "owns": "sócia", "manages": "gestora"}


@st.cache_data(ttl=600, show_spinner=False)
def _relation_alerts(path):
    """Uma linha por contrato com os alertas de relações e o texto que os explica. Só NIFs e nomes de empresas:
    os nomes das pessoas (tabela responsaveis) nunca são lidos pela aplicação."""
    empty = pd.DataFrame(columns=["contract_id", *RELATION_FLAGS, "relacoes"])
    if path is None:
        return empty
    con = _connection(path).cursor()
    if con.execute("SELECT 1 FROM information_schema.tables WHERE table_name = 'alertas_relacoes'").fetchall() == []:
        return empty
    rows = con.execute("SELECT contract_id, rule, name_a, nif_a, name_b, nif_b, link_types FROM alertas_relacoes").df()
    if rows.empty:
        return empty

    def describe(r):
        how = ", ".join(LINK_LABELS.get(t.strip(), t.strip()) for t in str(r.link_types).split(","))
        return f"{r.name_a or r.nif_a} ↔ {r.name_b or r.nif_b} ({how})"
    rows["relacao"] = rows.apply(describe, axis=1)
    out = rows.pivot_table(index="contract_id", columns="rule", values="relacao", aggfunc="size", fill_value=0) > 0
    out = out.reindex(columns=list(RELATION_FLAGS), fill_value=False)
    out["relacoes"] = rows.groupby("contract_id")["relacao"].agg(lambda s: " · ".join(sorted(set(s))))
    return out.reset_index()


def flag_label(key, params):
    return FLAGS[key][0]


def flag_help(key, params):
    return FLAGS[key][1].format(**params)


DEFAULT_PARAMS = {"pct": 0.9, "slip": 0.2, "days": 30, "share": 0.25}


@st.cache_data(ttl=600, show_spinner="A calcular indicadores...")
def compute_flags(df, pct=0.9, slip=0.2, days=30, share=0.25):
    d = df.copy()
    ref = d["award_decision_date"].fillna(d["signing_date"]).fillna(d["publication_date"])
    new_regime = ref >= REVISION_DATE
    works = d["contract_types"].fillna("").str.contains("Empreitada", case=False)
    kind = np.where(works, "works", "goods")
    ad_lim = [LIMITS[n][k][0] for n, k in zip(new_regime, kind)]
    cp_lim = [LIMITS[n][k][1] for n, k in zip(new_regime, kind)]
    d["limit"] = np.where(d["procedure_type"] == DIRECT, ad_lim,
                          np.where(d["procedure_type"] == PRIOR, cp_lim, np.nan))
    d["regime"] = np.where(new_regime, "DL 177/2026", "anterior")
    price = d["contract_price"]
    material = d["legal_basis"].fillna("").str.contains(r"24\.º|25\.º|26\.º|27\.º", regex=True)

    d["proposta_unica"] = d["procedure_type"].isin(COMPETITIVE) & (d["bidders"] == 1)
    d["perto_limiar"] = d["limit"].notna() & (price >= pct * d["limit"]) & (price < d["limit"])
    d["acima_limiar"] = d["limit"].notna() & (price >= d["limit"]) & ~material
    d["acima_preco_base"] = (d["base_price"] > 0) & (price > d["base_price"] * 1.001)
    d["derrapagem"] = d["effective_price"].notna() & (price > 0) & (d["effective_price"] > price * (1 + slip))
    lag = (d["publication_date"] - d["signing_date"]).dt.days
    d["publication_lag_days"] = lag
    d["publicacao_tardia"] = lag > days

    # fracionamento: grupos (entidade, adjudicatário, ano, grupo CPV, tipo) abaixo do limiar que somados o ultrapassam
    small = d[d["limit"].notna() & (price < d["limit"])].copy()
    small["cpv_group"] = small["cpv_code"].fillna("").str[:3]
    keys = ["authority_nif", "contractor_nif", "year", "cpv_group", "procedure_type", "regime"]
    g = small.groupby(keys, dropna=False).agg(n=("contract_id", "size"), total=("contract_price", "sum"),
                                              limit=("limit", "min"))
    hit = g[(g["n"] >= 2) & (g["total"] >= g["limit"])].reset_index()[keys]
    split_ids = small.merge(hit, on=keys)["contract_id"]
    d["fracionamento"] = d["contract_id"].isin(split_ids)

    # concentração: quota do adjudicatário no valor de AD + CP de cada entidade
    ad = d[d["procedure_type"].isin([DIRECT, PRIOR])]
    pair = ad.groupby(["authority_nif", "contractor_nif"]).agg(n=("contract_id", "size"), v=("contract_price", "sum"))
    pair["share"] = pair["v"] / pair.groupby(level=0)["v"].transform("sum")
    conc = pair[(pair["share"] >= share) & (pair["n"] >= 3)].reset_index()[["authority_nif", "contractor_nif"]]
    conc_ids = ad.merge(conc, on=["authority_nif", "contractor_nif"])["contract_id"]
    d["concentracao"] = d["contract_id"].isin(conc_ids)

    rel = _relation_alerts(db_snapshot()[0])
    d = d.drop(columns=[c for c in [*RELATION_FLAGS, "relacoes"] if c in d.columns]).merge(rel, on="contract_id", how="left")
    d["relacoes"] = d["relacoes"].fillna("")

    for key in FLAGS:
        d[key] = d[key].fillna(False).astype(bool)
    d["n_flags"] = d[list(FLAGS)].sum(axis=1)
    return d


def flags_for(row, params):
    return [flag_label(k, params) for k in FLAGS if row.get(k)]
