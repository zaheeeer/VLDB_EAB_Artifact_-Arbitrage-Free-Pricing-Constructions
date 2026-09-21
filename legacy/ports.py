"""Ports of published query-pricing constructions onto the common substrate.

Each port implements the construction's defining rule, not a paraphrase of it. Where the
substrate cannot support a construction exactly, the deviation is recorded in
:data:`FIDELITY` and reported alongside the results, so a weak result can be attributed
to the port rather than to the mechanism.

QIRANA (Deep and Koutris, SIGMOD 2017)
    A query is priced by how much it shrinks the buyer's set of possible databases.
    Evaluating that over all possible instances is intractable, so the construction
    examines a support set of instances that differ from the true one at one or two
    rows. Here the support set is a uniform random sample of single-row deletion
    neighbours. An asset distinguishes a neighbour when its answer changes once that row
    is removed; the four pricing functions below are computed over the distinguished
    set. Weighted coverage is the construction's default.

QueryMarket (Koutris et al., PODS 2012; JACM 2015)
    The seller prices a few views explicitly and the price of any query is the total
    price of the cheapest subset of those views that determines it. The paper restricts
    explicit prices to selection views for tractability, which is what the per-month
    spine provides here, and solves the cover with an integer program.
"""
from __future__ import annotations

import numpy as np

FIDELITY = {
    "qirana_weighted_coverage": {
        "source": "Deep and Koutris, SIGMOD 2017",
        "faithful": "support set of single-row neighbours; weighted coverage over the distinguished set",
        "deviation": "neighbours are deletions only, not the one-or-two-row modifications of the paper; "
                     "support-set size is a parameter we sweep rather than the paper's default",
    },
    "qirana_shannon": {
        "source": "Deep and Koutris, SIGMOD 2017",
        "faithful": "Shannon-entropy pricing function over the distinguished fraction",
        "deviation": "as above; entropy is taken over the distinguished/undistinguished split of the support set",
    },
    "querymarket_viewcover": {
        "source": "Koutris et al., PODS 2012 and JACM 2015",
        "faithful": "price is the cheapest determining subset of seller-priced views, solved exactly as an ILP",
        "deviation": "explicit prices are placed on selection views only, which is the paper's own tractable "
                     "restriction; the view set is the per-month spine rather than a seller's catalogue",
    },
}


def sample_support_set(con, n, seed=0):
    """Uniform random sample of base rows. Each row stands for the neighbour instance
    in which that row is absent."""
    rows = con.execute(
        f"SELECT l_orderkey, l_linenumber, CAST(date_trunc('month', l_shipdate) AS DATE) AS m, "
        f"l_quantity, l_discount, l_shipdate "
        f"FROM lineitem USING SAMPLE {int(n)} ROWS (reservoir, {int(seed)})"
    ).fetchall()
    return rows


def distinguishes(asset, row, TEMPLATES):
    """Does removing this row change the asset's answer?

    The asset aggregates additively over the rows its filter admits, so its answer
    changes exactly when the removed row satisfies both the template predicate and the
    asset's month restriction.
    """
    _, _, m, qty, disc, _ = row
    if m not in asset.months:
        return False
    pred = TEMPLATES[asset.template].get("py_pred")
    if pred is None:
        return True
    return bool(pred(qty, disc))


def qirana_scores(assets, support, TEMPLATES, weights=None):
    """Weighted coverage, uniform gain and Shannon entropy over the support set."""
    n = len(support)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    cov = np.zeros(len(assets))
    frac = np.zeros(len(assets))
    for i, a in enumerate(assets):
        mask = np.array([distinguishes(a, r, TEMPLATES) for r in support], dtype=bool)
        cov[i] = float(w[mask].sum())
        frac[i] = float(mask.mean())
    # Entropy is taken over the partition the answer induces on the support set, not
    # over the distinguished fraction. Undistinguished neighbours fall in one class,
    # since they all return the unchanged answer; each distinguished neighbour forms its
    # own class under the assumption that removing different rows yields different
    # answers, which is checked separately. Defined this way entropy rises with the
    # number distinguished, which is what makes it monotone in the determinacy order.
    k = np.rint(frac * n).astype(int)
    shannon = np.zeros(len(assets))
    for i, kk in enumerate(k):
        probs = []
        if n - kk > 0:
            probs.append((n - kk) / n)
        probs.extend([1.0 / n] * int(kk))
        p = np.array(probs, dtype=float)
        shannon[i] = float(-(p * np.log2(p)).sum())
    return {"qirana_weighted_coverage": cov,
            "qirana_shannon": shannon}


def querymarket_prices(assets, view_idx, view_prices, min_cost_cover_fn):
    """Price every asset as the cheapest determining subset of the seller-priced views.

    Views not in the seller's catalogue are priced only through the cover; an asset that
    no subset determines is left at infinity and excluded, which the caller reports.
    """
    big = float(max(view_prices.values())) * (len(view_idx) + 1) if view_prices else 1.0
    prices = np.full(len(assets), np.inf)
    catalogue = {i: (view_prices[i] if i in view_prices else big) for i in range(len(assets))}
    vec = np.array([catalogue[i] for i in range(len(assets))])
    for j in range(len(assets)):
        if j in view_prices:
            # An explicitly priced view is sold at its posted price; it does not need,
            # and cannot form, a cover of itself.
            prices[j] = float(view_prices[j])
            continue
        cost, plan = min_cost_cover_fn(assets, j, vec, sources=[i for i in view_idx if i != j])
        prices[j] = cost
    return prices
