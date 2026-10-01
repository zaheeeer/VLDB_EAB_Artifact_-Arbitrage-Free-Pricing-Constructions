"""Port of the revenue-maximization pricing algorithms of Chawla et al., VLDB 2019.

The paper reduces query pricing to revenue maximization with single-minded buyers and
unlimited supply. Each buyer i wants one query vector with valuation v_i; the query is
mapped to a hyperedge e_i, the conflict set of that query against a support set S, and
the seller picks a monotone subadditive price over subsets of S. Revenue is

    R(p) = sum over i with v_i >= p(e_i) of p(e_i).

Five algorithms are implemented, from Sections 5.1 and 5.2 of the paper. Two are folklore
or prior results the paper restates (UBP, UIP); three are the paper's own experimental
set (LPIP, CIP, Layering), and XOS is their combination rule.

Note on what this port shares with the rest of the harness: the hyperedges consumed here
are the same conflict sets our QIRANA port computes, so the two systems are priced on
identical inputs. That is the point of the substrate.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

FIDELITY = {
    "chawla_ubp": {
        "source": "Chawla et al., VLDB 2019, Section 5.1 (folklore, restated there)",
        "faithful": "optimal uniform bundle price by a linear pass over valuations sorted descending",
        "deviation": "none",
    },
    "chawla_uip": {
        "source": "Chawla et al., VLDB 2019, Section 5.2, after Guruswami et al.",
        "faithful": "uniform item price; sweep w = v_e/|e| over all hyperedges, keep the best revenue",
        "deviation": "none",
    },
    "chawla_lpip": {
        "source": "Chawla et al., VLDB 2019, Section 5.2",
        "faithful": "one LP per hyperedge over the edges at least as valuable, maximizing revenue "
                    "subject to every such edge remaining sold; best solution across all LPs",
        "deviation": "the paper solves one LP per hyperedge; we cap the sweep at the K most valuable "
                     "hyperedges when m is large, and report K",
    },
    "chawla_cip": {
        "source": "Chawla et al., VLDB 2019, Section 5.2, after Cheung and Swamy",
        "faithful": "primal-dual item pricing under a uniform capacity k, swept geometrically",
        "deviation": "we solve the welfare LP and read item prices from the duals; the paper's "
                     "presentation is in the limited-supply setting and extends to unlimited supply",
    },
    "chawla_layering": {
        "source": "Chawla et al., VLDB 2019, Algorithm 1",
        "faithful": "greedy layering by minimal set cover; unique item in each covering edge takes "
                    "that edge's valuation",
        "deviation": "minimal set cover is computed greedily, which the paper's O(Bm) bound assumes",
    },
    "chawla_xos": {
        "source": "Chawla et al., VLDB 2019, Section 5.2",
        "faithful": "maximum of the LPIP and CIP item prices, evaluated per hyperedge",
        "deviation": "inherits the deviations of its two inputs",
    },
}


def revenue(prices, vals):
    """R(p) = sum of p(e) over edges whose valuation covers the price."""
    p = np.asarray(prices, float)
    v = np.asarray(vals, float)
    return float(np.sum(np.where(p <= v, p, 0.0)))


def ubp(edges, vals):
    """Uniform bundle pricing. Every hyperedge is sold at one price P."""
    v = np.asarray(vals, float)
    order = np.argsort(-v)
    best_p, best_r = 0.0, 0.0
    for idx in order:
        P = v[idx]
        r = P * np.count_nonzero(v >= P)
        if r > best_r:
            best_p, best_r = float(P), float(r)
    return np.full(len(edges), best_p), {"P": best_p}


def _edge_price(edges, w):
    return np.array([float(w[list(e)].sum()) if len(e) else 0.0 for e in edges])


def uip(edges, vals, n_items):
    """Uniform item pricing. Every item carries the same weight w."""
    v = np.asarray(vals, float)
    sizes = np.array([max(len(e), 1) for e in edges], float)
    q = v / sizes
    best_w, best_r = 0.0, -1.0
    for cand in np.unique(q):
        w = np.full(n_items, cand)
        r = revenue(_edge_price(edges, w), v)
        if r > best_r:
            best_w, best_r = float(cand), r
    w = np.full(n_items, best_w)
    return _edge_price(edges, w), {"w": best_w, "weights": w}


def lpip(edges, vals, n_items, max_lps=60):
    """LP item pricing: one LP per hyperedge over the edges at least as valuable.

    The sweep must span the whole valuation range, not its top. For a high-valuation
    edge the constraint set F_e is small and the LP returns a high-price, low-volume
    solution; the revenue-bearing solutions sit at low thresholds where most edges stay
    sold. When m exceeds the budget we therefore take evenly spaced thresholds across
    the sorted valuations rather than truncating to the largest.
    """
    v = np.asarray(vals, float)
    srt = np.argsort(-v)
    order = srt if len(srt) <= max_lps else srt[np.linspace(0, len(srt) - 1, max_lps).astype(int)]
    best_w, best_r = None, -1.0
    for idx in order:
        F = [i for i in range(len(edges)) if v[i] >= v[idx]]
        if not F:
            continue
        # maximize sum over F of price(e), subject to price(e) <= v_e for every e in F.
        c = np.zeros(n_items)
        rowsA, b = [], []
        for i in F:
            ind = np.zeros(n_items)
            ind[list(edges[i])] = 1.0
            c += ind
            rowsA.append(ind)
            b.append(v[i])
        res = linprog(-c, A_ub=np.array(rowsA), b_ub=np.array(b),
                      bounds=[(0, None)] * n_items, method="highs")
        if not res.success:
            continue
        w = np.maximum(res.x, 0.0)
        r = revenue(_edge_price(edges, w), v)
        if r > best_r:
            best_w, best_r = w, r
    if best_w is None:
        best_w = np.zeros(n_items)
    return _edge_price(edges, best_w), {"weights": best_w, "n_lps": len(order)}


def cip(edges, vals, n_items, eps=0.5, k_max=None):
    """Capacity item pricing. Duals of a capacitated welfare LP give the item prices."""
    v = np.asarray(vals, float)
    m = len(edges)
    k_max = k_max or max(m, 2)
    best_w, best_r = np.zeros(n_items), -1.0
    k = 1.0
    while k <= k_max:
        # max sum v_e x_e  s.t.  for each item j: sum_{e ∋ j} x_e <= k ;  0 <= x_e <= 1
        rowsA, b = [], []
        for j in range(n_items):
            ind = np.array([1.0 if j in e else 0.0 for e in edges])
            if ind.any():
                rowsA.append(ind)
                b.append(k)
        if not rowsA:
            break
        res = linprog(-v, A_ub=np.array(rowsA), b_ub=np.array(b),
                      bounds=[(0, 1)] * m, method="highs")
        if res.success and res.ineqlin is not None:
            duals = np.abs(res.ineqlin.marginals)
            w = np.zeros(n_items)
            jj = 0
            for j in range(n_items):
                if any(j in e for e in edges):
                    w[j] = duals[jj]
                    jj += 1
            r = revenue(_edge_price(edges, w), v)
            if r > best_r:
                best_w, best_r = w, r
        k *= (1.0 + eps)
    return _edge_price(edges, best_w), {"weights": best_w}


def layering(edges, vals, n_items):
    """Algorithm 1. Greedy layering; each covering edge's unique item takes its valuation."""
    v = np.asarray(vals, float)
    remaining = set(range(len(edges)))
    w = np.zeros(n_items)
    best_layer, best_rev = [], -1.0
    while remaining:
        universe = set().union(*[edges[i] for i in remaining]) if remaining else set()
        if not universe:
            break
        uncovered, layer = set(universe), []
        pool = sorted(remaining, key=lambda i: -len(set(edges[i]) & uncovered))
        for i in pool:
            gain = set(edges[i]) & uncovered
            if gain:
                layer.append(i)
                uncovered -= gain
            if not uncovered:
                break
        if not layer:
            break
        # The paper requires a MINIMAL (irredundant) cover: every edge in it must hold an
        # item no other edge in it holds, which is what makes the unique-item pricing
        # step well defined. Greedy coverage is not minimal, so prune redundant edges.
        pruned = True
        while pruned and len(layer) > 1:
            pruned = False
            for i in list(layer):
                others = set().union(*[set(edges[j]) for j in layer if j != i])
                if set(edges[i]) <= others:
                    layer.remove(i)
                    pruned = True
                    break
        rev = float(sum(v[i] for i in layer))
        if rev > best_rev:
            best_layer, best_rev = list(layer), rev
        remaining -= set(layer)
    for i in best_layer:
        others = set().union(*[set(edges[j]) for j in best_layer if j != i]) if len(best_layer) > 1 else set()
        uniq = set(edges[i]) - others
        if uniq:
            w[min(uniq)] = v[i]
    return _edge_price(edges, w), {"weights": w, "layer_size": len(best_layer)}


def xos(edges, vals, n_items, **kw):
    """XOS: the larger of the LPIP and CIP prices, per hyperedge."""
    p1, i1 = lpip(edges, vals, n_items, **{k: x for k, x in kw.items() if k == "max_lps"})
    p2, i2 = cip(edges, vals, n_items)
    return np.maximum(p1, p2), {"lpip": i1, "cip": i2}


ALGORITHMS = {"chawla_ubp": ubp, "chawla_uip": uip, "chawla_lpip": lpip,
              "chawla_cip": cip, "chawla_layering": layering, "chawla_xos": xos}
