"""Paper-protocol fidelity: the rerun against the historical code, stage by stage.

``tests/legacy_ref/`` holds verbatim copies of the artifact's ``legacy/*.py`` and of the
helpers recovered from the notebooks. The notebook cells used below are copied from the
recovered notebooks (03-qpricing, 06-qpricing) with only their fixed sizes and seeds
turned into arguments; each is marked with its source cell.

The historical support sampler drew every deleted row with DuckDB's unseeded
``USING SAMPLE 1 ROWS``, which no rerun can repeat. In these tests the historical
``delete_one`` draws the same row the rerun draws (same seeded stream, same order) and
keeps its own value-match test. With that one substitution, the historical cells and the
rerun must agree exactly: same attrition, same lattice, same support sets, same prices,
same results table.

Everything runs on the synthetic release built by ``selftest.build_fixture``.
"""
from __future__ import annotations

import collections
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
LEGACY = Path(__file__).resolve().parent / "legacy_ref"
sys.path.insert(0, str(ROOT))
if str(LEGACY) not in sys.path:
    sys.path.append(str(LEGACY))

import market_run as legacy_market             # noqa: E402
import run_full as legacy_full                 # noqa: E402
import sqlshare_lattice as legacy_lattice      # noqa: E402
import sqlshare_load as legacy_load            # noqa: E402
import sqlshare_retention as legacy_retention  # noqa: E402
import sqlshare_translate as legacy_translate  # noqa: E402
import tpc_experiment as legacy_tpc            # noqa: E402

from sqlshare_rerun import corpus as C         # noqa: E402
from sqlshare_rerun import database as D       # noqa: E402
from sqlshare_rerun import pricing as P        # noqa: E402
from sqlshare_rerun import stages as ST        # noqa: E402
from sqlshare_rerun.config import PILOT_OWNERS, make_settings  # noqa: E402
from sqlshare_rerun.lattice import cover_structure, market_lattice, row_keys  # noqa: E402
from sqlshare_rerun.selftest import SELFTEST_SETTINGS, build_fixture  # noqa: E402
from sqlshare_rerun.translate import classify_error, transpile_tsql  # noqa: E402
from sqlshare_rerun.util import load_pickle, read_csv, read_json  # noqa: E402

REL = "sqlshare_data_release1"
WORKERS = int(os.environ.get("SQLSHARE_TEST_WORKERS", "2"))     # run_all.ps1 sets 1 if workers cannot start
THREADS = int(os.environ.get("SQLSHARE_TEST_THREADS", "4"))
PATIDX = re.compile(r"\bPATINDEX\s*\(", re.I)                     # 03-qpricing cell 4
HDR2 = re.compile(r"(?is)^create\s+view\s+\[([^\]]+)\]\s*\.\s*\[([^\]]+)\]\s*(\(.*?\))?\s*as\s+(.*)$")


# ========================================================================================
# historical code that lived in notebook cells
# ========================================================================================
def recovered(**env) -> dict:
    """The recovered notebook helpers, with the globals they expect supplied."""
    path = LEGACY / "recovered_original_helpers.py"
    ns = {"__name__": "recovered_original_helpers"}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)
    ns.update(env)
    return ns


def legacy_ent(zip_path):
    """03-qpricing cell 3: the release index."""
    z = zipfile.ZipFile(zip_path)
    infos = z.infolist()
    ent = {}
    for i in infos:
        p = i.filename.split("/")
        if len(p) >= 4 and p[1] == "data" and not i.is_dir():
            o, f = p[2].lower(), p[3]
            for key in {f.lower(), f.rsplit(".", 1)[0].lower(),
                        *(f[len(pr):].lower() for pr in ("table_", "materialized_") if f.lower().startswith(pr))}:
                ent.setdefault((o, key), (i.filename, i.file_size))
    z.close()
    return ent, infos


def legacy_base_closure(ent, views):
    """03-qpricing cell 3."""
    def base_closure(obj, seen=None):
        seen = seen or set()
        if obj in seen: return set()                                   # noqa: E701
        seen = seen | {obj}
        if obj in ent: return {obj}                                    # noqa: E701
        if obj in views:
            out = set()
            for d in views[obj]: out |= base_closure(d, seen)           # noqa: E701
            return out
        return set()
    return base_closure


def legacy_select(rows, ent, views, min_q=50, max_gb=2):
    """03-qpricing cell 3: components of the query-table graph and the selected markets."""
    base_closure = legacy_base_closure(ent, views)
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]   # noqa: E701,E702
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb                                    # noqa: E701
    qidx = [k for k, r in enumerate(rows) if r["status"] == "resolvable"]
    qtabs = {}
    for k in qidx:
        bs = set()
        for ref in rows[k]["refs"]: bs |= base_closure(ref)             # noqa: E701
        qtabs[k] = bs
        for b in bs: union(("q", k), ("t", b))                          # noqa: E701
    comp = collections.defaultdict(lambda: {"q": [], "t": set()})
    for k in qidx:
        if qtabs[k]: comp[find(("q", k))]["q"].append(k)                # noqa: E701
    for k in qidx:
        for b in qtabs[k]: comp[find(("t", b))]["t"].add(b)             # noqa: E701
    selected = [(c, v) for c, v in sorted(comp.items(), key=lambda kv: -len(kv[1]["q"]))
                if len(v["q"]) >= min_q and sum(ent[b][1] for b in v["t"]) / 1e9 < max_gb]
    sel_q = [k for _, v in selected for k in v["q"]]
    return comp, selected, sel_q, find


def legacy_vbody(path):
    """02-qpricing cells 36, 37, 42: view bodies from the extracted file, text mode."""
    with open(path, encoding="utf-8", errors="replace") as fh:    # historical: open(...).read()
        vs = fh.read()
    blocks = re.split(r"(?im)^create\s+view\s+", vs)
    SEPLINE = re.compile(r"\n_{10,}")
    vbody = {}
    for b in blocks[1:]:
        m = re.match(r"\[([^\]]+)\]\s*\.\s*\[([^\]]+)\]", b)
        if not m: continue                                             # noqa: E701
        cut = SEPLINE.split(b)[0].rstrip()
        vbody[(m.group(1).lower(), m.group(2).lower())] = "create view " + cut
    return vbody


def legacy_need_v(rows, sel_q, vbody, views):
    """02-qpricing: transitive closure of views needed by selected queries."""
    need_v, stack = set(), [ref for k in sel_q for ref in rows[k]["refs"]]
    while stack:
        o = stack.pop()
        if o in vbody and o not in need_v:
            need_v.add(o)
            stack.extend(views.get(o, ()))
    return need_v


def seeded_delete_one(row_seed: int):
    """The historical ``delete_one`` with ``USING SAMPLE 1 ROWS`` replaced by the rerun's
    seeded row draw. A row is drawn only when the object can be counted and is not empty,
    as in the rerun; a DELETE on a view raised historically, so views fail here too."""
    rng_r = np.random.default_rng(row_seed)

    def delete_one(con, tab, rng):
        try:
            cnt = con.execute(f'SELECT count(*) FROM "{tab[0]}"."{tab[1]}"').fetchone()[0]
        except Exception:
            return False
        if cnt == 0:
            return False
        r = int(rng_r.integers(cnt))
        kind = con.execute("SELECT table_type FROM information_schema.tables WHERE lower(table_schema) = ? "
                           "AND lower(table_name) = ?", [tab[0].lower(), tab[1].lower()]).fetchone()
        if kind is None or kind[0] != "BASE TABLE":
            raise RuntimeError("cannot delete from a view")
        # ---- the historical body (recovered helpers), with the sampled row fixed ----------
        cols = [d[0] for d in con.execute(f'SELECT * FROM "{tab[0]}"."{tab[1]}" LIMIT 0').description]
        row = con.execute(f'SELECT * FROM "{tab[0]}"."{tab[1]}" ORDER BY rowid LIMIT 1 OFFSET {r}').fetchone()
        if row is None: return False                                   # noqa: E701
        preds = []
        for c, v in zip(cols, row):
            if v is None: preds.append(f'"{c}" IS NULL')               # noqa: E701
            else:
                lit = str(v).replace("'", "''")
                preds.append(f'"{c}" IS NOT DISTINCT FROM \'{lit}\'')
        n = con.execute(f'SELECT count(*) FROM "{tab[0]}"."{tab[1]}" WHERE {" AND ".join(preds)}').fetchone()[0]
        if n != 1: return False                                        # noqa: E701
        con.execute(f'DELETE FROM "{tab[0]}"."{tab[1]}" WHERE {" AND ".join(preds)}')
        return True
    return delete_one


# ========================================================================================
# fixture: the synthetic release, prepared once by the rerun (paper protocol)
# ========================================================================================
@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("fidelity")
    (root / "data").mkdir()
    build_fixture(root / "data" / f"{REL}.zip")
    s = make_settings("paper", root, **SELFTEST_SETTINGS, workers=WORKERS, threads=THREADS, pool_stall_s=120)
    ST.stage_prepare(s)
    return s


@pytest.fixture(scope="module")
def legacy_corpus(prepared):
    """03-qpricing cell 3: analyse(), ent, and the market selection, all historical."""
    rows, tables, views, owner_files = legacy_retention.analyse(str(prepared.zip_file))
    ent, infos = legacy_ent(prepared.zip_file)
    comp, selected, sel_q, find = legacy_select(rows, ent, views, prepared.min_component_queries,
                                                prepared.max_component_gb)
    return {"rows": rows, "tables": tables, "views": views, "owner_files": owner_files, "ent": ent,
            "infos": infos, "comp": comp, "selected": selected, "sel_q": sel_q, "find": find}


def _clean(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    if isinstance(x, (float, np.floating)) and float(x).is_integer():
        return str(int(x))
    return str(x)


def _records(df, cols):
    return [tuple(_clean(v) for v in row) for row in df[cols].itertuples(index=False, name=None)]


# ========================================================================================
# 1. corpus, translation, and market selection
# ========================================================================================
EXTRA_TSQL = [
    "SELECT TOP 5 a + ' x' + b AS c FROM [1].[t.csv]",
    "SELECT ISNULL(a, 0) AS a, LEN(b) AS n FROM [1].[t]",
    "SELECT CONVERT(varchar(10), a) AS s FROM [1].[t]",
    "SELECT DATEPART(year, d) AS y FROM [1].[t]",
    "SELECT a FROM [1].[t] WHERE b LIKE '%x%' ORDER BY a DESC",
    "SELECT CAST(a AS float) / 2 AS h FROM [x].[y] GROUP BY a HAVING count(*) > 1",
    "SELECT [col with space] FROM [o].[t w s.csv]",
    "SELECT * FROM [446].[bob's data.csv]",
    "SELECT 'a' + 'b' + c + 'd' AS s FROM [o].[t]",
    "SELEC broken FROM",
]


def test_translation_matches_legacy(prepared, legacy_corpus):
    for q in [r["text"] for r in legacy_corpus["rows"]] + EXTRA_TSQL:
        def run(fn):
            try:
                return fn(q)
            except Exception as e:                    # both must fail the same way
                return ("error", type(e).__name__)
        assert run(transpile_tsql) == run(legacy_translate.transpile_tsql), q


def test_error_classes_match_legacy():
    msgs = ["Catalog Error: Table with name x does not exist!", "Binder Error: Referenced column y not found",
            "Parser Error: syntax error at or near", "Binder Error: something", "Conversion Error: bad",
            "Invalid Input Error: odd", "Binder Error: does not have a column named z"]
    for m in msgs:
        assert classify_error(m) == legacy_translate.classify_error(m), m


def test_parse_matches_legacy(prepared, legacy_corpus, tmp_path):
    corpus = C.parse_release(prepared.zip_file)
    L = legacy_corpus
    assert corpus.rows == L["rows"]
    assert corpus.tables == L["tables"]
    assert corpus.views == L["views"]
    assert corpus.owner_files == L["owner_files"]
    assert corpus.ent == L["ent"]
    # view bodies: the historical run read the extracted file in text mode
    with zipfile.ZipFile(prepared.zip_file) as z:
        z.extract(f"{REL}/view_script.txt", tmp_path)
    assert corpus.vbody == legacy_vbody(tmp_path / REL / "view_script.txt")


def test_view_bodies_with_crlf_match_legacy(prepared, tmp_path):
    """If the release used CRLF line ends, text-mode reading changed them; so does the rerun."""
    src = zipfile.ZipFile(prepared.zip_file)
    crlf = tmp_path / f"{REL}.zip"
    with zipfile.ZipFile(crlf, "w") as z:
        for i in src.infolist():
            data = src.read(i.filename)
            if i.filename.endswith("view_script.txt"):
                data = data.replace(b"\n", b"\r\n")
            z.writestr(i.filename, data)
    src.close()
    corpus = C.parse_release(crlf)
    with zipfile.ZipFile(crlf) as z:
        z.extract(f"{REL}/view_script.txt", tmp_path / "x")
    assert corpus.vbody == legacy_vbody(tmp_path / "x" / REL / "view_script.txt")
    assert corpus.vbody == C.parse_release(prepared.zip_file).vbody     # same bodies as with LF


@pytest.mark.parametrize("min_q", [1, 5, 50])
def test_market_selection_matches_legacy(prepared, legacy_corpus, min_q):
    L = legacy_corpus
    corpus = C.parse_release(prepared.zip_file)
    comp, selected, sel_q, find = legacy_select(L["rows"], L["ent"], L["views"], min_q, 2)
    sel = C.select_markets(corpus, min_q, 2.0)
    assert sel.sel_q == sel_q
    assert [sel.comp_queries[c] for c in sel.selected] == [v["q"] for _, v in selected]
    assert [sel.comp_tables[c] for c in sel.selected] == [v["t"] for _, v in selected]
    assert sel.n_components == len(comp)
    for _, v in comp.items():                       # every historical component is one market
        assert len({sel.comp_of_query[k] for k in v["q"]}) == 1


def test_needed_views_match_legacy(prepared, legacy_corpus, tmp_path):
    L = legacy_corpus
    corpus = C.parse_release(prepared.zip_file)
    sel = C.select_markets(corpus, prepared.min_component_queries, prepared.max_component_gb)
    with zipfile.ZipFile(prepared.zip_file) as z:
        z.extract(f"{REL}/view_script.txt", tmp_path)
    vbody = legacy_vbody(tmp_path / REL / "view_script.txt")
    need = C.needed_views(corpus, sel.sel_q)
    assert need == legacy_need_v(L["rows"], L["sel_q"], vbody, L["views"])
    assert need


def test_view_statements_match_legacy(prepared, legacy_corpus):
    corpus = C.parse_release(prepared.zip_file)
    ns = recovered(HDR2=HDR2, _re=re, vbody=corpus.vbody, transpile_tsql=legacy_translate.transpile_tsql)
    assert corpus.vbody
    for o, body in corpus.vbody.items():
        assert D.view_statement(body) == ns["build_view2"](o), o


# ========================================================================================
# 2. the database substrate: load, aliases, views, bind, execute (legacy/pipeline.py)
# ========================================================================================
def _historical_disk(prepared, legacy_corpus, where: Path):
    """The historical Windows disk: data/sqlshare/<release>/data/<owner>/<sanitized name>."""
    corpus = C.parse_release(prepared.zip_file)
    sel = C.select_markets(corpus, prepared.min_component_queries, prepared.max_component_gb)
    need_v = C.needed_views(corpus, sel.sel_q)
    members = C.files_to_extract(corpus, sel, need_v, PILOT_OWNERS)
    manifest = C.extract(prepared.zip_file, members, where / "extracted")
    disk = D.windows_disk_view(manifest, {n: i for i, (n, _) in enumerate(corpus.infos)})
    root = where / "data" / "sqlshare" / REL
    for m in disk:
        dst = root / "data" / m["owner"] / m["disk_name"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(m["local"], dst)
    with zipfile.ZipFile(prepared.zip_file) as z:
        z.extract(f"{REL}/view_script.txt", where / "data" / "sqlshare")
    return disk


def test_load_matches_legacy(prepared, legacy_corpus, tmp_path):
    disk = _historical_disk(prepared, legacy_corpus, tmp_path)
    root = tmp_path / "data" / "sqlshare" / REL
    con_l, con_r = duckdb.connect(), duckdb.connect()
    owners = sorted({m["owner"] for m in disk})
    ref = pd.DataFrame([r for o in owners for r in legacy_load.load_market(con_l, str(root), o)])
    got = D.load_all(con_r, disk)
    cols = ["owner", "file", "table", "mode", "rows", "cols", "delim", "error"]
    assert _records(got, cols) == _records(ref, cols)
    assert (got["file"] == "table_bob's data.csv").any()             # the quote case is present
    for r in got[got["mode"].notna()].itertuples():
        q = f'SELECT * FROM "{r.owner}"."{r.table}"'
        pd.testing.assert_frame_equal(con_r.execute(q).df(), con_l.execute(q).df())


def test_prepare_matches_legacy_pipeline(prepared, legacy_corpus, tmp_path, monkeypatch):
    """legacy/pipeline.py, fed by the historical parse and selection, against stage_prepare."""
    L = legacy_corpus
    _historical_disk(prepared, legacy_corpus, tmp_path)
    monkeypatch.chdir(tmp_path)
    root = f"data/sqlshare/{REL}"
    vbody = legacy_vbody(f"{root}/view_script.txt")
    need_v = legacy_need_v(L["rows"], L["sel_q"], vbody, L["views"])
    helpers = recovered(HDR2=HDR2, _re=re, vbody=vbody, transpile_tsql=legacy_translate.transpile_tsql)
    path = LEGACY / "pipeline.py"
    ns = {"__name__": "legacy_pipeline"}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)
    ns.update(root=root, infos=L["infos"], ent=L["ent"], views=L["views"], vbody=vbody, rows=L["rows"],
              sel_q=L["sel_q"], need_v=need_v, load_market=legacy_load.load_market,
              object_name=legacy_load.object_name, transpile_tsql=legacy_translate.transpile_tsql,
              PATIDX=PATIDX, classify_error=legacy_translate.classify_error,
              build_view2=helpers["build_view2"], DB_PATH=str(tmp_path / "legacy.duckdb"))
    ref = ns["run_pipeline"]()
    ref["con"].close()

    work = prepared.work_dir
    summary = read_json(work / "prepare_summary.json")
    cols = ["owner", "file", "table", "mode", "rows", "cols", "delim", "error"]
    assert _records(read_csv(work / "load_log.csv", keep_default_na=False), cols) == _records(ref["load"], cols)
    assert summary["aliases"] == ref["aliases"]
    assert load_pickle(work / "catalog.pkl")["schemas"] == ref["schemas"]
    assert summary["views_needed"] == len(need_v)
    assert summary["views_created"] == len(ref["views_made"])
    vf = read_csv(work / "views_failed.csv", dtype=str, keep_default_na=False)
    assert {(r.owner, r.view): r.reason for r in vf.itertuples()} == ref["views_failed"]
    st = read_csv(work / "bind_stages.csv")
    assert list(zip(st.k, st.stage)) == list(zip(ref["stages"].k, ref["stages"].stage))
    ex = read_csv(work / "executions.csv", dtype={"own": str}, keep_default_na=False)
    rx = ref["execs"]
    assert list(zip(ex.k, ex.ok, ex.rows, ex.own)) == list(zip(rx.k, rx.ok, rx.rows, rx.own))
    bad = ~rx.ok.astype(bool)
    assert bad.any()                                     # the fixture has an execution failure
    assert list(ex.err[bad.to_numpy()]) == list(rx.err[bad])
    assets = load_pickle(work / "assets.pkl")
    assert sorted(assets.k) == sorted(rx[(rx.ok) & (rx.rows > 0)].k)


# ========================================================================================
# 3. pure functions: pricing, buyers, scores, ports
# ========================================================================================
def test_pricing_helpers_match_legacy():
    rng = np.random.default_rng(0)
    for trial in range(20):
        k = int(rng.integers(3, 40))
        info = rng.uniform(0.01, 1, k)
        a = legacy_tpc.buyer_sample(300, info, np.random.default_rng(trial))
        b = P.buyer_sample(300, info, np.random.default_rng(trial))
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
        t, v, _ = a
        s = rng.uniform(0, 1, k)
        s[rng.random(k) < 0.2] = 0
        assert P.fit_scale_grid(t, v, s) == legacy_tpc.fit_scale(t, v, s)
        np.testing.assert_array_equal(P.monopoly_prices(t, v, k), legacy_tpc.monopoly_prices(t, v, k))
        pairs = [(int(i), int(j)) for i, j in rng.integers(0, k, (15, 2)) if i != j]
        p = rng.uniform(0, 10, k)
        np.testing.assert_array_equal(P.enforce_monotone(p, pairs), legacy_tpc.enforce_monotone(p, pairs))
        assert P.audit_monotone(p, pairs) == legacy_tpc.audit_monotone(p, pairs)
        assert P.score(t, v, p) == legacy_full.score(t, v, p)
        rows_, ss, nc = rng.integers(1, 99, k), rng.integers(0, 999, k), rng.integers(1, 9, k)
        fa, fb = P.info_families(rows_, ss, nc), legacy_tpc.info_families(rows_, ss, nc)
        assert list(fa) == list(fb)
        for f in fa:
            np.testing.assert_array_equal(fa[f], fb[f])
        edges = [frozenset(rng.choice(50, int(rng.integers(0, 50)), replace=False).tolist()) for _ in range(k)]
        qa, qb = P.qirana_from_edges(edges, 50), recovered()["qirana_from_edges"](edges, 50)
        for f in qb:
            np.testing.assert_array_equal(qa[f], qb[f])


def test_querymarket_single_predecessor_matches_legacy():
    rng = np.random.default_rng(1)
    for _ in range(20):
        n = int(rng.integers(2, 30))
        pairs = [(int(i), int(j)) for i, j in rng.integers(0, n, (int(rng.integers(0, 40)), 2)) if i != j]
        base = rng.uniform(1, 500, n)
        # 03-qpricing cell 51 (and legacy/market_run.py), verbatim
        derivable_from = collections.defaultdict(list)
        for s, t in pairs: derivable_from[t].append(s)                  # noqa: E701
        cat = [j for j in range(n) if j not in derivable_from]
        catp = {j: float(base[j]) for j in cat}
        bigM = max(catp.values()) * 1.5 if catp else 1.0
        qm_s = np.array([catp.get(j, min([catp.get(s, bigM) for s in derivable_from.get(j, [])] or [bigM]))
                         for j in range(n)])
        qm, n_cat = P.querymarket_single_predecessor(n, pairs, base)
        np.testing.assert_array_equal(qm, qm_s)
        assert n_cat == len(cat)


@pytest.mark.parametrize("seed", range(4))
def test_chawla_ports_match_legacy(seed):
    rng = np.random.default_rng(seed)
    n_items, n_assets = 60, 35
    edges = [frozenset(np.flatnonzero(rng.random(n_items) < rng.uniform(0, 0.15)).tolist()) for _ in range(n_assets)]
    tr_t = rng.integers(0, n_assets, 150)
    tr_v = 100 * rng.lognormal(0, 0.6, 150) * rng.uniform(0.1, 1, n_assets)[tr_t] ** 0.7
    got = P.chawla_all(edges, tr_t, tr_v, n_items, max_lps=12)
    for alg in ("chawla_ubp", "chawla_uip", "chawla_lpip", "chawla_cip", "chawla_layering", "chawla_xos"):
        kw = {"max_lps": 12} if alg in ("chawla_lpip", "chawla_xos") else {}
        np.testing.assert_array_equal(got[alg], legacy_full.chawla_prices(alg, edges, tr_t, tr_v, n_items, **kw), alg)


def test_lattice_process_pool_matches_legacy(tmp_path):
    """A synthetic market large enough (45 assets) to use the parallel path."""
    rng = np.random.default_rng(7)
    base = pd.DataFrame({"st": rng.choice(list("ABCDEFG"), 300), "yr": rng.integers(2000, 2006, 300),
                         "x": rng.integers(0, 50, 300), "y": rng.normal(size=300).round(2)})
    answers = {}
    for k in range(45):
        cols = sorted(rng.choice(base.columns, int(rng.integers(1, 5)), replace=False).tolist())
        df = base[base.st.isin(rng.choice(list("ABCDEFG"), int(rng.integers(2, 8)), replace=False))]
        if rng.random() < 0.3 and "x" in cols and len(cols) > 1:
            key = [c for c in cols if c != "x" and c != "y"]
            df = df.groupby(key, as_index=False)["x"].sum() if key else df
            cols = [c for c in cols if c in df.columns]
        df = df.loc[:, cols]
        if rng.random() < 0.5:
            df = df.drop_duplicates()
        answers[1000 + k] = df.reset_index(drop=True)
    ref = legacy_lattice.component_lattice(answers)
    for workers in sorted({1, 3 if WORKERS > 1 else 1}):
        got = market_lattice(answers, "value_set", None, workers, tmp_path, stall_s=120)
        assert got["verified"] == ref["verified"]
        assert got["by_family"] == ref["by_family"]
        assert got["attempted"] == ref["attempted"]
    assert len(ref["verified"]) > 20


# ========================================================================================
# 4. lattice on the prepared markets
# ========================================================================================
def _answers_by_market(s):
    corpus, sel, assets, cat = ST._load_prepared(s)
    con = ST._connect(s)
    out = {cid: ST._answers(s, con, assets, cid) for _, cid, _ in ST._market_ranks(assets, sel.selected)}
    con.close()
    return out


def test_lattice_stage_matches_legacy(prepared):
    ST.stage_lattice(prepared)
    comps = read_csv(prepared.results_dir / "lattice" / "components.csv")
    assert len(comps) >= 2
    for cid, data in _answers_by_market(prepared).items():
        path = prepared.results_dir / "lattice" / "pairs" / f"{cid}.pkl"
        if not path.exists():
            continue
        got, ref = load_pickle(path), legacy_lattice.component_lattice(data["answers"])
        assert got["verified"] == ref["verified"]
        assert got["by_family"] == ref["by_family"] and got["attempted"] == ref["attempted"]


def test_cover_structure_matches_legacy(prepared):
    for cid, data in _answers_by_market(prepared).items():
        ans1 = data["answers"]
        keys1 = sorted(ans1)
        # 03-qpricing cell 10, first lines
        cache = {}; contrib_map = {}; nrows_map = {}                   # noqa: E702
        for k in keys1:
            c_, n_ = legacy_lattice.partial_sources(ans1, k, cache, target_cap=2000)
            if c_ is not None and len(ans1[k]) == len(ans1[k].drop_duplicates()):
                contrib_map[k], nrows_map[k] = c_, n_
        contrib, nrows = cover_structure(ans1, keys1, 2000)
        assert contrib.keys() == contrib_map.keys() and nrows == nrows_map
        for k in contrib:
            lk, sk = list(legacy_lattice.row_keys(ans1[k])), sorted(row_keys(ans1[k]))
            assert contrib[k].keys() == contrib_map[k].keys()
            for s_ in contrib[k]:
                assert {sk[i] for i in contrib[k][s_]} == {lk[i] for i in contrib_map[k][s_]}


# ========================================================================================
# 5. the main market: Table 6, falsification, and the full protocol (Tables 3 to 5)
# ========================================================================================
def test_main_market_matches_legacy_cells(prepared, legacy_corpus):
    s = prepared
    summary = ST.stage_main_market(s)
    out = s.results_dir / "main_market"
    L = legacy_corpus
    base_closure = legacy_base_closure(L["ent"], L["views"])
    rows = L["rows"]
    corpus, sel, assets, cat = ST._load_prepared(s)
    rank, c1, _ = ST._market_ranks(assets, sel.selected)[0]
    db2 = ST._connect(s)
    schemas2 = cat["schemas"]
    fp_sql = recovered()["fp_sql"]
    ROW_CAP = s.row_cap

    # ---- 03-qpricing cell 5 (answers; timings taken from the rerun's cache) -------------
    E2ok = assets
    sub1 = E2ok[(E2ok.comp == c1) & (E2ok.rows <= ROW_CAP)]
    ans1 = {}
    for r in sub1.itertuples():
        db2.execute(f"SET search_path='{r.own}'")
        ans1[r.k] = legacy_lattice.norm_frame(db2.execute(r.duck).df())
    db2.execute("SET search_path='main'")
    keys1 = sorted(ans1); ki = {k: i for i, k in enumerate(keys1)}    # noqa: E702
    lat = legacy_lattice.component_lattice(ans1)
    pairs1i = [(ki[s_], ki[t]) for s_, t, _ in lat["verified"] if s_ in ki and t in ki]
    prov = {}
    for k in keys1:
        bs = set()
        for ref in rows[k]["refs"]: bs |= base_closure(ref)              # noqa: E701
        tot = 0
        for b in bs:
            try: tot += db2.execute(f'SELECT count(*) FROM "{b[0]}"."{b[1]}"').fetchone()[0]   # noqa: E701
            except Exception: pass                                      # noqa: E701
        prov[k] = tot
    etime1 = load_pickle(s.work_dir / "answers" / f"{c1}_cap{s.row_cap}.pkl")["seconds"]
    rowsz1 = np.array([len(ans1[k]) for k in keys1], float)
    prov1 = np.array([prov[k] for k in keys1], float)
    et1 = np.array([etime1[k] for k in keys1], float); et1[et1 <= 0] = et1[et1 > 0].min()   # noqa: E702

    # ---- cell 6 (tables sorted: the historical order came from set iteration) ------------
    own_of = sub1.set_index("k").own.to_dict()
    paths = {k: ",".join([own_of[k]] + [x for x in schemas2 if x != own_of[k]]) for k in keys1}
    duckmap = sub1.set_index("k").duck.to_dict()
    base_fp = {}
    for k in keys1:
        db2.execute(f"SET search_path='{paths[k]}'")
        base_fp[k] = db2.execute(fp_sql(duckmap[k])).fetchone()
    c1_tables = {}
    for k in keys1:
        for ref in rows[k]["refs"]:
            for b in base_closure(ref):
                if b not in c1_tables:
                    try: c1_tables[b] = db2.execute(f'SELECT count(*) FROM "{b[0]}"."{b[1]}"').fetchone()[0]  # noqa: E701
                    except Exception: pass                              # noqa: E701
    tabs = sorted(b for b, n in c1_tables.items() if n > 0)
    w = np.array([c1_tables[b] for b in tabs], float); w /= w.sum()     # noqa: E702
    # ---- cell 7 ----------------------------------------------------------------------
    readers = collections.defaultdict(list)
    for k in keys1:
        bs = set()
        for ref in rows[k]["refs"]: bs |= base_closure(ref)              # noqa: E701
        for b in bs: readers[b].append(k)                               # noqa: E701

    def fp_changed(k):
        db2.execute(f"SET search_path='{paths[k]}'")
        return db2.execute(fp_sql(duckmap[k])).fetchone() != base_fp[k]

    # ---- cell 9: resolution curve --------------------------------------------------------
    CKPTS = list(s.curve_checkpoints)
    delete_one = seeded_delete_one(ST._row_seed(s.curve_seed))
    rng = np.random.default_rng(s.curve_seed)
    cum = {k: 0 for k in keys1}; used = 0; curve = []                 # noqa: E702
    while used < CKPTS[-1]:
        b = tabs[int(rng.choice(len(tabs), p=w))]
        ks = readers.get(b, [])
        if not ks: continue                                            # noqa: E701
        db2.execute("BEGIN TRANSACTION")
        try: ok = delete_one(db2, b, rng)                              # noqa: E701
        except Exception: ok = False                                   # noqa: E701
        if not ok:
            db2.execute("ROLLBACK"); continue                          # noqa: E702
        used += 1
        for k in ks:
            try:
                if fp_changed(k): cum[k] += 1                          # noqa: E701
            except Exception: pass                                     # noqa: E701
        db2.execute("ROLLBACK")
        if used in CKPTS:
            v = np.array([cum[k] for k in keys1], float)
            curve.append({"support_rows": used, "zero_price": int((v == 0).sum()),
                          "distinct_prices": len(set(v.tolist())), "median": float(np.median(v)),
                          "max": float(v.max())})
    db2.execute("SET search_path='main'")
    dc = np.array([cum[k] for k in keys1], float)
    got_curve = read_csv(out / "resolution_curve.csv")
    assert got_curve[["support_rows", "zero_price", "distinct_prices", "median", "max"]].to_dict("records") == curve

    # ---- cell 38: witnesses --------------------------------------------------------------
    viol = [(s_, t) for s_, t in pairs1i if dc[t] > dc[s_] + 1e-9]
    involved = sorted({i for p in viol for i in p})
    track = [keys1[i] for i in involved]
    inc = {k: set() for k in track}
    delete_one = seeded_delete_one(ST._row_seed(s.witness_seed + 7919))
    rng = np.random.default_rng(s.witness_seed); used = 0; N = s.witness_n   # noqa: E702
    while used < N:
        b = tabs[int(rng.choice(len(tabs), p=w))]
        ks = [k for k in readers.get(b, []) if k in inc]
        db2.execute("BEGIN TRANSACTION")
        try: ok = delete_one(db2, b, rng)                              # noqa: E701
        except Exception: ok = False                                   # noqa: E701
        if not ok:
            db2.execute("ROLLBACK"); continue                          # noqa: E702
        used += 1
        for k in ks:
            try:
                if fp_changed(k): inc[k].add(used)                     # noqa: E701
            except Exception: pass                                     # noqa: E701
        db2.execute("ROLLBACK")
    db2.execute("SET search_path='main'")
    wit = [(s_, t, len(inc[keys1[t]] - inc[keys1[s_]])) for s_, t in viol]
    # ---- cell 39 ---------------------------------------------------------------------
    falsified = set(viol)
    clean_pairs = [p for p in pairs1i if p not in falsified]
    got_f = read_csv(out / "falsified_edges.csv") if viol else pd.DataFrame(columns=["src", "tgt"])
    ref_f = [(keys1[a], keys1[b_], int(dc[a]), int(dc[b_]), n_) for a, b_, n_ in wit]
    assert list(got_f.itertuples(index=False, name=None)) == ref_f
    assert summary["falsified_curve"] == len(viol)
    assert summary["witnessed"] == sum(1 for x in wit if x[2] > 0)

    # ---- cells 47 and 48: Table 6 ----------------------------------------------------------
    def build_edges(weights, n, seed, row_seed):
        """cell 48 (cell 47 is the same loop with the size weights)."""
        delete_one = seeded_delete_one(row_seed)
        incl = {k: set() for k in keys1}; used = 0                    # noqa: E702
        rg = np.random.default_rng(seed)
        while used < n:
            b = tabs[int(rg.choice(len(tabs), p=weights))]
            db2.execute("BEGIN TRANSACTION")
            try: ok = delete_one(db2, b, rg)                           # noqa: E701
            except Exception: ok = False                               # noqa: E701
            if not ok:
                db2.execute("ROLLBACK"); continue                      # noqa: E702
            used += 1
            for k in readers.get(b, []):
                if k not in incl: continue                             # noqa: E701
                try:
                    if fp_changed(k): incl[k].add(used - 1)            # noqa: E701
                except Exception: pass                                 # noqa: E701
            db2.execute("ROLLBACK")
        db2.execute("SET search_path='main'")
        return [frozenset(incl[k]) for k in keys1]
    S_N = s.table6_n
    edges_sql = build_edges(w, S_N, s.table6_seed, ST._row_seed(s.table6_seed))
    nread = np.array([len(readers.get(b, [])) for b in tabs], float)
    w_read = nread / nread.sum()
    e_read = build_edges(w_read, S_N, s.table6_seed, ST._row_seed(s.table6_seed + 1))
    t6 = read_csv(out / "table6_support_sampling.csv")
    for row, E in zip(t6.itertuples(), (edges_sql, e_read)):
        assert row.assets_priceable == sum(1 for e in E if e)
        assert row.distinct_price_levels == len({len(e) for e in E})
        assert row.median_conflict_set == float(np.median([len(e) for e in E]))

    # ---- cell 10 (cover targets) and cell 11 (the adversary) -----------------------------
    cache = {}; contrib_map = {}; nrows_map = {}                       # noqa: E702
    for k in keys1:
        c_, n_ = legacy_lattice.partial_sources(ans1, k, cache, target_cap=2000)
        if c_ is not None and len(ans1[k]) == len(ans1[k].drop_duplicates()):
            contrib_map[k], nrows_map[k] = c_, n_
    ns = recovered(keys1=keys1, ki=ki, contrib_map=contrib_map, nrows_map=nrows_map, ans1=ans1,
                   min_cost_row_cover=legacy_lattice.min_cost_row_cover,
                   verify_row_cover=legacy_lattice.verify_row_cover)
    # ---- cell 51: scores and the fit, freeze, and evaluate runs -------------------------
    qs_s = ns["qirana_from_edges"](e_read, S_N)
    derivable_from = collections.defaultdict(list)
    for s_, t in clean_pairs: derivable_from[t].append(s_)               # noqa: E701
    cat_ = [j for j in range(len(keys1)) if j not in derivable_from]
    catp = {j: float(prov1[j]) for j in cat_}
    bigM = max(catp.values()) * 1.5 if catp else 1.0
    qm_s = np.array([catp.get(j, min([catp.get(s_, bigM) for s_ in derivable_from.get(j, [])] or [bigM]))
                     for j in range(len(keys1))])
    scale_s = {"uniform": np.ones(len(keys1)), "size_proportional": rowsz1.copy(), "compute_metered": et1.copy(),
               **{k: v.copy() for k, v in qs_s.items()}, "querymarket_viewcover": qm_s}
    info_s = legacy_tpc.info_families(rowsz1, prov1, np.ones(len(keys1)))

    def plan_fn_s(p):
        c, psz, vok, vbad = ns["sqlshare_plan_all"](p)
        return np.minimum(p, c), psz, vok, vbad
    RS2 = legacy_full.run(keys1, ans1, clean_pairs, scale_s, e_read, S_N, info_s, plan_fn_s,
                          legacy_tpc.audit_monotone, legacy_tpc.fit_scale, legacy_tpc.buyer_sample,
                          legacy_tpc.monopoly_prices, legacy_tpc.enforce_monotone,
                          seeds=s.run_seeds, n_buyers=s.n_buyers, max_lps=s.max_lps)
    db2.close()
    # ---- cell 53 ---------------------------------------------------------------------
    dc2 = np.array([len(e) for e in e_read], float)
    new_viol = [(s_, t) for s_, t in clean_pairs if dc2[t] > dc2[s_] + 1e-9]

    assert summary["claimed_edges"] == len(pairs1i)
    assert summary["clean_pairs_used"] == len(clean_pairs)
    assert summary["falsified_by_pricing_support"] == len(new_viol)
    assert summary["cover_targets"] == len(contrib_map)
    assert summary["querymarket_catalogue"] == len(cat_)
    assert summary["zero_priced_readership"] == sum(1 for e in e_read if not e)
    assert summary["plans_verified"] == int(RS2.plans_verified.sum())
    assert summary["plans_failed"] == int(RS2.plans_failed.sum())
    runs = read_csv(out / "runs.csv")
    assert set(runs.arm) == {"published"}
    got = runs[list(RS2.columns)].reset_index(drop=True)
    pd.testing.assert_frame_equal(got, RS2.reset_index(drop=True), check_dtype=False, check_exact=True)
    assert len(RS2) == 15 * 4 * len(s.run_seeds)


# ========================================================================================
# 6. all markets: legacy/market_run.py (Table 7, Figure 2b)
# ========================================================================================
def test_markets_match_legacy_market_run(prepared, legacy_corpus):
    s = prepared
    ST.stage_markets(s)
    got_all = read_csv(s.results_dir / "markets" / "runs.csv")
    L = legacy_corpus
    base_closure = legacy_base_closure(L["ent"], L["views"])
    corpus, sel, assets, cat = ST._load_prepared(s)
    E2ok = assets
    # 06-qpricing cell 6
    q2t = {}
    for r in E2ok.itertuples():
        s_ = set()
        for ref in L["rows"][r.k].get("refs", ()): s_ |= base_closure(ref)   # noqa: E701
        q2t[r.k] = {b for b in s_ if b in L["ent"]}
    db2 = ST._connect(s)
    ns = recovered()
    n_ok = 0
    for rank, cid, _ in ST._market_ranks(assets, sel.selected):
        ctx = (E2ok, s.row_cap, db2, legacy_lattice.norm_frame, legacy_lattice.component_lattice, q2t,
               L["ent"], ns["fp_sql"], seeded_delete_one(ST._row_seed(s.market_seed)),
               ns["qirana_from_edges"], legacy_tpc.info_families, legacy_tpc.buyer_sample,
               legacy_tpc.fit_scale, legacy_tpc.monopoly_prices, legacy_tpc.audit_monotone,
               legacy_full.chawla_prices, legacy_full.score)
        ref = pd.DataFrame(legacy_market.run_market(
            cid, ctx, support_n=s.market_support_n, n_buyers=s.market_buyers, seeds=s.market_seeds,
            max_lps=s.market_max_lps, families=s.market_families, min_assets=s.market_min_assets))
        got = got_all[got_all.cid == cid].reset_index(drop=True)
        assert list(got.status) == list(ref.status) and list(got.assets) == list(ref.assets)
        if ref.status.iloc[0] != "ok":
            continue
        n_ok += 1
        cols = ["pairs", "falsified", "zero_priced", "family", "seed", "mechanism", "revenue_norm",
                "welfare_norm", "served", "mono_violation_rate"]
        pd.testing.assert_frame_equal(got[cols], ref[cols], check_dtype=False, check_exact=True)
    db2.close()
    assert n_ok >= 2
