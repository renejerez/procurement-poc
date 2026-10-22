"""Acesso aos dados e indicadores de risco, partilhados por todas as páginas da aplicação."""
import os
import shutil
import tempfile
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
    repo = _secret("GITHUB_DATA_REPO")
    if repo:
        return f"github-release://{repo}"
    for candidate in [APP_DIR / "data" / "procurement.duckdb", APP_DIR.parent / "data" / "procurement.duckdb"]:
        if candidate.exists():
            return str(candidate)
    return None


def _github_release_db(repo, tmp):
    import requests
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    token = _secret("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.get(f"https://api.github.com/repos/{repo}/releases/tags/{RELEASE_TAG}", headers=headers, timeout=30)
    if r.status_code == 404:
        return None, None
    r.raise_for_status()
    asset = next((a for a in r.json().get("assets", []) if a["name"] == DB_ASSET), None)
    if asset is None:
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


@st.cache_resource(ttl=600, show_spinner="A obter a base de dados...")
def db_snapshot():
    """Devolve (caminho da cópia local, data da última atualização). Verifica se há versão nova a cada 10 minutos."""
    src = db_source()
    if not src:
        return None, None
    tmp = Path(tempfile.gettempdir())
    if src.startswith("github-release://"):
        return _github_release_db(src[len("github-release://"):], tmp)
    path = Path(src)
    if not path.exists():
        return None, None
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
    return _connection(path).cursor().execute(sql, list(params)).df()


def query(sql, params=()):
    path, _ = db_snapshot()
    return _query(path, sql, tuple(params))


def has_table(name):
    return not query("SELECT 1 FROM information_schema.tables WHERE table_name = ?", [name]).empty


def require_db():
    path, _ = db_snapshot()
    if path is None:
        st.error("Base de dados não encontrada. Corra os notebooks 01 e 02 (e, para o Streamlit Cloud, a secção 3 do "
                 "notebook 04), ou defina PROCUREMENT_DB / GITHUB_DATA_REPO.")
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

# ─────────────────────────────── contratos ───────────────────────────────


@st.cache_data(ttl=600, show_spinner="A carregar contratos...")
def _contracts(path):
    df = _connection(path).cursor().execute("""
        WITH bids AS (
            SELECT contract_id, COUNT(*) AS bidders
            FROM contract_parties WHERE role = 'competitor' GROUP BY contract_id
        ),
        cpv AS (
            SELECT contract_id, MIN(cpv_code) AS cpv_code, ARG_MIN(cpv_description, cpv_code) AS cpv_description
            FROM contract_cpvs GROUP BY contract_id
        )
        SELECT c.contract_id, c.publication_date, c.signing_date, c.award_decision_date, c.close_date,
               c.procedure_type, c.contract_types, c.object, c.legal_basis,
               CAST(c.base_price AS DOUBLE)      AS base_price,
               CAST(c.contract_price AS DOUBLE)  AS contract_price,
               CAST(c.effective_price AS DOUBLE) AS effective_price,
               c.execution_days, c.documents_link, c.year,
               v.authority_nif, v.authority, v.contractor_nif, v.contractor,
               b.bidders, cpv.cpv_code, cpv.cpv_description
        FROM contracts c
        JOIN v_contracts v USING (contract_id)
        LEFT JOIN bids b USING (contract_id)
        LEFT JOIN cpv USING (contract_id)
    """).df()
    for col in ["publication_date", "signing_date", "award_decision_date", "close_date"]:
        df[col] = pd.to_datetime(df[col])
    df["authority"] = df["authority"].fillna("(sem entidade)")
    df["contractor"] = df["contractor"].fillna("(sem adjudicatário)")
    df["month"] = df["publication_date"].dt.to_period("M").dt.to_timestamp()
    df["base_link"] = BASE_DETAIL + df["contract_id"].astype(str)
    return df


def contracts():
    path, _ = db_snapshot()
    return _contracts(path)


def sidebar_filters(df):
    """Filtros globais (barra lateral), aplicados em todas as páginas."""
    st.sidebar.header("Filtros")
    years = sorted(df["publication_date"].dt.year.dropna().astype(int).unique())
    sel_years = st.sidebar.multiselect("Ano de publicação", years, default=years, key="f_years")
    authorities = df.groupby("authority")["contract_price"].sum().sort_values(ascending=False).index.tolist()
    sel_auth = st.sidebar.multiselect("Entidade adjudicante", authorities, key="f_auth",
                                      placeholder="Todas")
    procs = df["procedure_type"].value_counts().index.tolist()
    sel_proc = st.sidebar.multiselect("Tipo de procedimento", procs, key="f_proc", placeholder="Todos")
    out = df[df["publication_date"].dt.year.isin(sel_years)]
    if sel_auth:
        out = out[out["authority"].isin(sel_auth)]
    if sel_proc:
        out = out[out["procedure_type"].isin(sel_proc)]
    st.sidebar.caption(f"{num(len(out))} de {num(len(df))} contratos")
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
}


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

    for key in FLAGS:
        d[key] = d[key].fillna(False).astype(bool)
    d["n_flags"] = d[list(FLAGS)].sum(axis=1)
    return d


def flags_for(row, params):
    return [flag_label(k, params) for k in FLAGS if row.get(k)]
