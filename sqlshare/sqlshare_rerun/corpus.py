"""Corpus stage: download the release, parse queries and views, select markets, extract.

Parsing follows the historical code line for line (legacy/sqlshare_retention.py and the
02/03-qpricing notebook cells), because the selected population (2,660 queries in 21
components) depends on these exact regular expressions.
"""
from __future__ import annotations

import collections
import hashlib
import http.client
import os
import re
import zlib
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .util import LOG, replace_file, sha256_file, windows_name

RELEASE = "sqlshare_data_release1"

SEP = re.compile(r"\n_{10,}\n")
SEPLINE = re.compile(r"\n_{10,}")
REF = re.compile(r"\[([^\[\]]+)\]\s*\.\s*\[([^\[\]]+)\]")
CREATE_VIEW = re.compile(r"CREATE\s+VIEW\s+\[([^\[\]]+)\]\s*\.\s*\[([^\[\]]+)\]", re.I)
VIEW_SPLIT_DEPS = re.compile(r"(?i)\bCREATE\s+VIEW\b")
VIEW_SPLIT_BODY = re.compile(r"(?im)^create\s+view\s+")
VIEW_HEAD = re.compile(r"\[([^\]]+)\]\s*\.\s*\[([^\]]+)\]")
FROM_ISH = re.compile(r"\b(FROM|JOIN)\b", re.I)
GROUPBY = re.compile(r"\bGROUP\s+BY\b", re.I)
AGG = re.compile(r"\b(sum|count|avg|min|max)\s*\(", re.I)
WHERE = re.compile(r"\bWHERE\b", re.I)
PREFIXES = ("table_", "materialized_")


# ----------------------------------------------------------------------------------------
# download
# ----------------------------------------------------------------------------------------
BLOCKED_HINT = ("The download site refused the connection ({why}): your network or a proxy blocks it. "
                "Download the file in a browser from {url} , save it (for example as "
                "C:\\sqlshare\\sqlshare_data_release1.zip), and run again with -Zip <that path>.")


def _blocked(text: str) -> bool:
    return any(s in text for s in ("403", "Forbidden", "Tunnel connection failed", "407"))


TAIL = 4 << 20                       # bytes re-fetched when a partial download is resumed


def download(url: str, dest: Path, retries: int = 30) -> dict:
    """Resumable download to ``dest``. Safe to interrupt and rerun: the partial file
    ``<dest>.part`` is continued, and ``dest`` appears only when the download is complete.
    ``retries`` counts failures in a row without progress; a site that refuses the
    connection (HTTP 403, a proxy block) stops after three attempts with instructions."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    if dest.exists():
        LOG.info("release already present: %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
        return {"path": str(dest), "bytes": dest.stat().st_size}
    total = None
    fails = blocked = 0
    LOG.info("downloading %s", url)
    LOG.info("to %s (about 3.6 GB; progress is shown every 30 s)", dest)
    if part.exists() and part.stat().st_size > TAIL:
        # After a crash or power cut the end of the partial file may not have reached the
        # disk; fetch the last few MB again rather than trust them.
        with open(part, "r+b") as fh:
            fh.truncate(part.stat().st_size - TAIL)
    best = part.stat().st_size if part.exists() else 0
    while True:
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "sqlshare-rerun/2.0"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        why = None
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                status = getattr(resp, "status", 200)
                if have and status != 206:
                    LOG.warning("the server ignored the resume request; starting the download again")
                    have = 0
                length = resp.headers.get("Content-Length")
                if length is not None:
                    total = have + int(length)
                t0 = last = time.perf_counter()
                start = have
                with open(part, "ab" if have else "wb") as fh:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                        have += len(chunk)
                        if have % (256 << 20) < len(chunk):
                            fh.flush()
                            os.fsync(fh.fileno())
                        now = time.perf_counter()
                        if now - last >= 30:
                            last = now
                            rate = (have - start) / 1e6 / max(now - t0, 1e-6)
                            LOG.info("downloaded %.0f MB%s (%.1f MB/s)", have / 1e6,
                                     f" of {total / 1e6:.0f} MB" if total else "", rate)
            if total is not None and have >= total:
                break
            if total is None:
                raise RuntimeError("the server did not report the file size, so the download cannot be "
                                   "checked. Download the zip in a browser and run again with -Zip <path>.")
            why = f"connection closed at {have / 1e6:.0f} of {total / 1e6:.0f} MB"
        except urllib.error.HTTPError as e:
            if e.code == 416 and have > 0:          # nothing left to send: the part file is complete
                LOG.info("download already complete (%.0f MB)", have / 1e6)
                break
            why = f"HTTP {e.code}"
            if e.code in (401, 403, 404, 407, 410, 451):
                blocked += 1
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError, ConnectionError, OSError) as e:
            why = str(e) or type(e).__name__
            if _blocked(why):
                blocked += 1
        now_have = part.stat().st_size if part.exists() else 0
        if now_have > best:                  # real progress: beyond anything reached before
            best, fails, blocked = now_have, 0, 0
        else:
            fails += 1
        if blocked >= 3:
            raise RuntimeError(BLOCKED_HINT.format(why=why, url=url))
        if fails >= retries:
            raise RuntimeError(f"the download failed {retries} times in a row ({why}). Run the same command "
                               "again later to resume it, or download the zip in a browser and use -Zip.")
        wait = min(60, 5 * max(fails, 1))
        LOG.warning("download interrupted (%s); resuming in %d s", why, wait)
        time.sleep(wait)
    with open(part, "rb+") as fh:
        os.fsync(fh.fileno())
    replace_file(part, dest)
    LOG.info("download complete: %.1f MB", dest.stat().st_size / 1e6)
    return {"path": str(dest), "bytes": dest.stat().st_size}


#: The release the paper used: 3,435 zip entries, 25.40 GB uncompressed (02-qpricing cell 1).
EXPECTED_ENTRIES = 3435


def verify_zip(path: Path) -> dict:
    """Open the zip, check it is the SQLShare release, and fingerprint it."""
    try:
        with zipfile.ZipFile(path) as z:
            infos = z.infolist()
    except (zipfile.BadZipFile, OSError) as e:
        raise RuntimeError(f"{path} is not a complete zip file ({e}).") from e
    names = {i.filename for i in infos}
    need = {f"{RELEASE}/queries.txt", f"{RELEASE}/view_script.txt"}
    if not need <= names:
        raise RuntimeError(f"{path} does not look like the SQLShare release (missing {sorted(need - names)})")
    unpacked = sum(i.file_size for i in infos)
    LOG.info("zip index: %d entries, %.2f GB unpacked; computing SHA-256 once for the record",
             len(infos), unpacked / 1e9)
    if len(infos) != EXPECTED_ENTRIES:
        LOG.warning("this release has %d entries; the one the paper used had %d. "
                    "Results may differ from the paper for that reason.", len(infos), EXPECTED_ENTRIES)
    return {"entries": len(infos), "unpacked_bytes": unpacked, "sha256": sha256_file(path)}


# ----------------------------------------------------------------------------------------
# parsing
# ----------------------------------------------------------------------------------------
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().rstrip(";").lower()


@dataclass
class Corpus:
    rows: list
    tables: set
    views: dict
    vbody: dict
    ent: dict
    owner_files: collections.Counter
    infos: list                       # (zip name, size) for files under data/
    data_owner_dirs: set = field(default_factory=set)

    # ---- resolution ------------------------------------------------------------------
    def resolvable(self, obj, seen=None) -> bool:
        if obj in self.tables:
            return True
        if obj not in self.views:
            return False
        seen = seen or set()
        if obj in seen:
            return False
        seen = seen | {obj}
        return all(self.resolvable(d, seen) for d in self.views[obj])

    def base_closure(self, obj, seen=None) -> set:
        seen = seen or set()
        if obj in seen:
            return set()
        seen = seen | {obj}
        if obj in self.ent:
            return {obj}
        if obj in self.views:
            out = set()
            for d in self.views[obj]:
                out |= self.base_closure(d, seen)
            return out
        return set()

    def query_tables(self, k: int) -> set:
        out = set()
        for ref in self.rows[k]["refs"]:
            out |= self.base_closure(ref)
        return out


def parse_release(zip_path: Path) -> Corpus:
    with zipfile.ZipFile(zip_path) as z:
        infos = z.infolist()
        vs = z.read(f"{RELEASE}/view_script.txt").decode("utf-8", "replace")
        qtext = z.read(f"{RELEASE}/queries.txt").decode("utf-8", "replace")
    tables, owner_files = set(), collections.Counter()
    ent, data_infos, owner_dirs = {}, [], set()
    for i in infos:
        p = i.filename.split("/")
        if len(p) >= 4 and p[1] == "data" and not i.is_dir():
            owner, fname = p[2], p[3]
            owner_dirs.add(owner)
            data_infos.append((i.filename, i.file_size))
            for pref in PREFIXES:
                if fname.lower().startswith(pref):
                    tables.add((owner.lower(), fname[len(pref):].lower()))
            tables.add((owner.lower(), fname.lower()))
            tables.add((owner.lower(), fname.rsplit(".", 1)[0].lower()))
            owner_files[owner] += 1
            o, f = owner.lower(), fname
            keys = {f.lower(), f.rsplit(".", 1)[0].lower(),
                    *(f[len(pr):].lower() for pr in PREFIXES if f.lower().startswith(pr))}
            for key in keys:
                ent.setdefault((o, key), (i.filename, i.file_size))

    views = {}
    for block in VIEW_SPLIT_DEPS.split(vs):
        m = CREATE_VIEW.search("CREATE VIEW " + block)
        if not m:
            continue
        owner, name = m.group(1).lower(), m.group(2).lower()
        deps = {(o.lower(), t.lower()) for o, t in REF.findall(block)}
        deps.discard((owner, name))
        views[(owner, name)] = deps

    # The historical view bodies were read from the extracted file in text mode, which
    # turns CRLF and CR line ends into LF (the dependency parse above read the raw bytes).
    vs_text = vs.replace("\r\n", "\n").replace("\r", "\n")
    vbody = {}
    for b in VIEW_SPLIT_BODY.split(vs_text)[1:]:
        m = VIEW_HEAD.match(b)
        if not m:
            continue
        cut = SEPLINE.split(b)[0].rstrip()
        vbody[(m.group(1).lower(), m.group(2).lower())] = "create view " + cut

    queries = [q.strip() for q in SEP.split(qtext) if q.strip()]
    corpus = Corpus(rows=[], tables=tables, views=views, vbody=vbody, ent=ent,
                    owner_files=owner_files, infos=data_infos, data_owner_dirs=owner_dirs)
    for q in queries:
        refs = {(o.lower(), t.lower()) for o, t in REF.findall(q)}
        if not refs or not FROM_ISH.search(q):
            status = "no_relation_ref"
        elif all(corpus.resolvable(r) for r in refs):
            status = "resolvable"
        else:
            status = "missing_relation"
        owners = {o for o, _ in refs}
        corpus.rows.append({
            "text": q, "norm": norm(q), "status": status, "n_refs": len(refs),
            "owner": next(iter(owners)) if len(owners) == 1 else ("multi" if owners else "none"),
            "has_group": bool(GROUPBY.search(q)), "has_agg": bool(AGG.search(q)),
            "has_where": bool(WHERE.search(q)), "refs": refs,
        })
    return corpus


# ----------------------------------------------------------------------------------------
# components and market selection
# ----------------------------------------------------------------------------------------
@dataclass
class Selection:
    comp_of_query: dict              # query index -> component id (smallest query index)
    comp_queries: dict               # component id -> sorted query indices
    comp_tables: dict                # component id -> set of base tables
    selected: list                   # component ids, historical order
    sel_q: list                      # selected query indices in historical order
    sel_t: list
    n_components: int


def select_markets(corpus: Corpus, min_queries: int, max_gb: float) -> Selection:
    rows = corpus.rows
    qidx = [k for k, r in enumerate(rows) if r["status"] == "resolvable"]
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    qtabs = {}
    for k in qidx:
        bs = corpus.query_tables(k)
        qtabs[k] = bs
        for b in sorted(bs):
            union(("q", k), ("t", b))

    root_q = collections.defaultdict(list)
    root_t = collections.defaultdict(set)
    for k in qidx:
        if qtabs[k]:
            root_q[find(("q", k))].append(k)
    for k in qidx:
        for b in qtabs[k]:
            root_t[find(("t", b))].add(b)
    # Component id = its smallest query index. The historical dict was filled in query
    # order, so sorting by (-size, id) reproduces its stable-sort tie-breaking.
    comp_queries, comp_tables, comp_of_query = {}, {}, {}
    for r, qs in root_q.items():
        cid = min(qs)
        comp_queries[cid] = sorted(qs)
        comp_tables[cid] = root_t[r]
        for k in qs:
            comp_of_query[k] = cid
    order = sorted(comp_queries, key=lambda c: (-len(comp_queries[c]), c))

    def gb(c):
        return sum(corpus.ent[b][1] for b in comp_tables[c]) / 1e9

    selected = [c for c in order if len(comp_queries[c]) >= min_queries and gb(c) < max_gb]
    sel_q = [k for c in selected for k in comp_queries[c]]
    sel_t = sorted({b for c in selected for b in comp_tables[c]})
    return Selection(comp_of_query, comp_queries, comp_tables, selected, sel_q, sel_t,
                     n_components=len(comp_queries))


def needed_views(corpus: Corpus, sel_q: list) -> set:
    need_v, stack = set(), [ref for k in sel_q for ref in corpus.rows[k]["refs"]]
    while stack:
        o = stack.pop()
        if o in corpus.vbody and o not in need_v:
            need_v.add(o)
            stack.extend(corpus.views.get(o, ()))
    return need_v


def files_to_extract(corpus: Corpus, sel: Selection, need_v: set, pilot_owners) -> list:
    """Zip members the historical run had on disk when it built the database."""
    names = {corpus.ent[b][0] for b in sel.sel_t}
    alldeps = set()
    for o in need_v:
        alldeps |= {(a.lower(), b.lower()) for a, b in REF.findall(corpus.vbody.get(o, ""))} - {o}
    for k in sel.sel_q:
        alldeps |= set(corpus.rows[k]["refs"])
    names |= {corpus.ent[x][0] for x in alldeps if x in corpus.ent}
    pilot = {p.lower() for p in pilot_owners}
    for fname, _ in corpus.infos:
        if fname.split("/")[2].lower() in pilot:
            names.add(fname)
    return sorted(names)


def _crc(path: Path) -> int:
    crc = 0
    with open(path, "rb") as fh:
        while True:
            b = fh.read(1 << 22)
            if not b:
                return crc
            crc = zlib.crc32(b, crc)


def extract(zip_path: Path, members: list, dest: Path) -> list:
    """Extract members to short, safe paths and return the manifest.

    Every member keeps a record of its original name, the name the historical Windows
    extraction gave it, and where it lives now. Local names are derived from a hash of the
    member name, so they are stable and never collide or hit Windows name rules.
    """
    dest.mkdir(parents=True, exist_ok=True)
    manifest = []
    t0 = last = time.perf_counter()
    with zipfile.ZipFile(zip_path) as z:
        sizes = {i.filename: i.file_size for i in z.infolist()}
        crcs = {i.filename: i.CRC for i in z.infolist()}
        for n, name in enumerate(members):
            owner, fname = name.split("/")[2], name.split("/")[3]
            disk_name = windows_name(fname)
            ext = disk_name.rsplit(".", 1)[-1].lower() if "." in disk_name else ""
            ext = re.sub(r"[^a-z0-9]", "", ext)[:8] or "dat"
            digest = hashlib.sha1(name.encode("utf-8", "surrogatepass")).hexdigest()[:16]
            local = dest / ("o_" + re.sub(r"[^A-Za-z0-9_.-]", "_", owner)[:40]) / f"{digest}.{ext}"
            if not (local.exists() and local.stat().st_size == sizes[name] and _crc(local) == crcs[name]):
                local.parent.mkdir(parents=True, exist_ok=True)
                tmp = local.with_suffix(".tmp")
                try:
                    with z.open(name) as src, open(tmp, "wb") as out:
                        while True:
                            b = src.read(1 << 22)
                            if not b:
                                break
                            out.write(b)
                        out.flush()
                        os.fsync(out.fileno())
                except zipfile.BadZipFile as e:
                    raise RuntimeError(f"the release zip is damaged ({name}: {e}). Delete {zip_path} "
                                       f"and run again to download it fresh.") from e
                replace_file(tmp, local)
            manifest.append({"zip_name": name, "owner": owner, "file": fname,
                             "disk_name": disk_name, "local": str(local), "bytes": sizes[name]})
            now = time.perf_counter()
            if now - last > 30:
                last = now
                LOG.info("extracted %d of %d files (%.0f s)", n + 1, len(members), now - t0)
    return manifest
