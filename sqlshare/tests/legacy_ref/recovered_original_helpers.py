"""Verbatim recovered definitions; dependencies/global state must be supplied by caller.
Historical code, not corrected or a standalone reproduction driver.
"""
import numpy as np


# Source: frames/OPERON-278ea99c/02-qpricing/notebook.ipynb, cell 65, execution 136
def build_view2(o):
    m = HDR2.match(vbody[o])
    if not m: return None
    own, name, cols, sel = m.group(1), m.group(2), m.group(3), m.group(4)
    duck = transpile_tsql(sel)
    colspec = ""
    if cols:
        cl = [c.strip() for c in _re.findall(r"\[([^\]]+)\]", cols)]
        if cl: colspec = "(" + ",".join(f'"{c}"' for c in cl) + ")"
    return f'CREATE OR REPLACE VIEW "{own}"."{name}" {colspec} AS {duck}'


# Source: frames/OPERON-278ea99c/03-qpricing/notebook.ipynb, cell 6, execution 195
fp_sql = lambda q: f"SELECT count(*) AS n, sum(hash(to_json(qq))::HUGEINT) AS h FROM ({q}) qq"


# Source: frames/OPERON-278ea99c/03-qpricing/notebook.ipynb, cell 8, execution 197
def delete_one(con, tab, rng):
    """Delete a single sampled row by value match. Returns True if exactly one went."""
    cols = [d[0] for d in con.execute(f'SELECT * FROM "{tab[0]}"."{tab[1]}" LIMIT 0').description]
    row = con.execute(f'SELECT * FROM "{tab[0]}"."{tab[1]}" USING SAMPLE 1 ROWS').fetchone()
    if row is None: return False
    preds = []
    for c, v in zip(cols, row):
        if v is None: preds.append(f'"{c}" IS NULL')
        else:
            lit = str(v).replace("'", "''")
            preds.append(f'"{c}" IS NOT DISTINCT FROM \'{lit}\'')
    n = con.execute(f'SELECT count(*) FROM "{tab[0]}"."{tab[1]}" WHERE {" AND ".join(preds)}').fetchone()[0]
    if n != 1: return False
    con.execute(f'DELETE FROM "{tab[0]}"."{tab[1]}" WHERE {" AND ".join(preds)}')
    return True


# Source: frames/OPERON-278ea99c/03-qpricing/notebook.ipynb, cell 11, execution 200
def sqlshare_plan_all(pv):
    cost = pv.copy(); psize = np.ones(len(keys1), int); vok=vbad=0
    pm = {k: float(pv[ki[k]]) for k in keys1}
    for k, c_ in contrib_map.items():
        cc_, plan = min_cost_row_cover(c_, nrows_map[k], pm)
        if plan and cc_ < pv[ki[k]] - 1e-9:
            if verify_row_cover(ans1, k, plan):
                cost[ki[k]] = cc_; psize[ki[k]] = len(plan); vok+=1
            else: vbad+=1
    return cost, psize, vok, vbad


# Source: frames/OPERON-278ea99c/03-qpricing/notebook.ipynb, cell 51, execution 255
def qirana_from_edges(edges, n):
    cov = np.array([float(len(e)) for e in edges]); frac = cov/n
    sh = np.zeros(len(edges))
    for i,kk in enumerate(cov.astype(int)):
        probs = ([ (n-kk)/n ] if n-kk>0 else []) + [1.0/n]*int(kk)
        p = np.array(probs); sh[i] = float(-(p*np.log2(p)).sum())
    return {"qirana_weighted_coverage":cov, "qirana_uniform_gain":frac, "qirana_shannon":sh}
