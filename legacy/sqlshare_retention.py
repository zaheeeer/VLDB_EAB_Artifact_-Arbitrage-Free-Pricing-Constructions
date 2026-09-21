"""Retention analysis for the SQLShare corpus.

A query is priceable only if every relation it references resolves to a table or view
that the consent-scoped release actually ships. This script measures that fraction, the
per-owner market sizes, and how much of the surviving corpus carries the grouping and
aggregation structure a derivability lattice needs.

Nothing here evaluates a query. It reads the archive index and the two text files only.
"""
from __future__ import annotations

import collections
import re
import zipfile

SEP = re.compile(r"\n_{10,}\n")
REF = re.compile(r"\[([^\[\]]+)\]\s*\.\s*\[([^\[\]]+)\]")
CREATE_VIEW = re.compile(r"CREATE\s+VIEW\s+\[([^\[\]]+)\]\s*\.\s*\[([^\[\]]+)\]", re.I)
FROM_ISH = re.compile(r"\b(FROM|JOIN)\b", re.I)
GROUPBY = re.compile(r"\bGROUP\s+BY\b", re.I)
AGG = re.compile(r"\b(sum|count|avg|min|max)\s*\(", re.I)
WHERE = re.compile(r"\bWHERE\b", re.I)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().rstrip(";").lower()


def load(zip_path: str):
    z = zipfile.ZipFile(zip_path)
    root = "sqlshare_data_release1"

    tables = set()
    owner_files = collections.Counter()
    for i in z.infolist():
        p = i.filename.split("/")
        if len(p) >= 4 and p[1] == "data" and not i.is_dir():
            owner, fname = p[2], p[3]
            # The release stores base tables as ``table_<dataset name>`` and materialized
            # views as ``materialized_<view name>``. Dataset names keep their original
            # extension and may carry a collision suffix, so nothing may be stripped from
            # the right: match the remainder after the prefix verbatim.
            for pref in ("table_", "materialized_"):
                if fname.lower().startswith(pref):
                    tables.add((owner.lower(), fname[len(pref):].lower()))
            tables.add((owner.lower(), fname.lower()))
            tables.add((owner.lower(), fname.rsplit(".", 1)[0].lower()))
            owner_files[owner] += 1

    vs = z.read(f"{root}/view_script.txt").decode("utf-8", "replace")
    views = {}
    for block in re.split(r"(?i)\bCREATE\s+VIEW\b", vs):
        m = CREATE_VIEW.search("CREATE VIEW " + block)
        if not m:
            continue
        owner, name = m.group(1).lower(), m.group(2).lower()
        body = block
        deps = {(o.lower(), t.lower()) for o, t in REF.findall(body)}
        deps.discard((owner, name))
        views[(owner, name)] = deps

    qtext = z.read(f"{root}/queries.txt").decode("utf-8", "replace")
    queries = [q.strip() for q in SEP.split(qtext) if q.strip()]
    return queries, tables, views, owner_files


def resolvable(obj, tables, views, seen=None) -> bool:
    if obj in tables:
        return True
    if obj not in views:
        return False
    seen = seen or set()
    if obj in seen:
        return False
    seen = seen | {obj}
    return all(resolvable(d, tables, views, seen) for d in views[obj])


def analyse(zip_path: str):
    queries, tables, views, owner_files = load(zip_path)

    rows = []
    for q in queries:
        refs = {(o.lower(), t.lower()) for o, t in REF.findall(q)}
        if not refs or not FROM_ISH.search(q):
            status = "no_relation_ref"
        elif all(resolvable(r, tables, views) for r in refs):
            status = "resolvable"
        else:
            status = "missing_relation"
        owners = {o for o, _ in refs}
        rows.append(
            {
                "text": q,
                "norm": norm(q),
                "status": status,
                "n_refs": len(refs),
                "owner": next(iter(owners)) if len(owners) == 1 else ("multi" if owners else "none"),
                "has_group": bool(GROUPBY.search(q)),
                "has_agg": bool(AGG.search(q)),
                "has_where": bool(WHERE.search(q)),
                "refs": refs,
            }
        )
    return rows, tables, views, owner_files
