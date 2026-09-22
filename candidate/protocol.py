"""New candidate helpers for validation; not the code that produced V8.

These helpers are deliberately separate from the unchanged historical modules.
They do not establish equivalence to any published pricing construction.
"""
from collections import defaultdict
import numpy as np


def _nonnegative_vector(value, name):
    out = np.asarray(value, dtype=float)
    if out.ndim != 1 or not np.all(np.isfinite(out)) or np.any(out < 0):
        raise ValueError(f"{name} must be a finite nonnegative vector")
    return out


def popularity(k, rng, skew=1.0):
    if k < 1 or not np.isfinite(skew) or skew < 0:
        raise ValueError("positive asset count and finite nonnegative skew required")
    weights = (rng.permutation(k) + 1.0) ** (-skew)
    return weights / weights.sum()


def sample_buyers(n, info, demand, rng, gamma=0.7, sigma=0.6, scale=100.0):
    info = _nonnegative_vector(info, "info")
    demand = _nonnegative_vector(demand, "demand")
    if n < 0 or len(info) != len(demand) or not len(info):
        raise ValueError("invalid buyer count or demand dimension")
    if not np.isclose(demand.sum(), 1.0, rtol=0, atol=1e-12):
        raise ValueError("demand must sum to one")
    if not all(np.isfinite(x) and x >= 0 for x in (gamma, sigma, scale)):
        raise ValueError("invalid value-distribution parameter")
    targets = rng.choice(len(info), size=n, p=demand)
    values = scale * rng.lognormal(0.0, sigma, size=n) * info[targets] ** gamma
    return targets, values


def buyer_split(info, seed, n_train=800, n_test=800, shift=False):
    """Independent streams; one demand vector unless shift is explicitly requested."""
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    train_demand = popularity(len(info), streams[0])
    test_demand = popularity(len(info), streams[3]) if shift else train_demand.copy()
    train = sample_buyers(n_train, info, train_demand, streams[1])
    test = sample_buyers(n_test, info, test_demand, streams[2])
    return {"train": train, "test": test, "train_demand": train_demand,
            "test_demand": test_demand, "shift": bool(shift)}


def fit_scale_breakpoints(targets, values, scores):
    """Maximize training revenue at value/score breakpoints, without a fixed cap.

    Zero scores stay free. Inclusive purchase ties match ordinary price <= value.
    Adjacent lower floats address division/multiplication rounding at a breakpoint.
    Direct objective evaluation is O(number_of_buyers squared), suitable for the
    small calibration fixtures here. This is not a scalability implementation.
    """
    scores = _nonnegative_vector(scores, "scores")
    values = _nonnegative_vector(values, "values")
    targets = np.asarray(targets)
    if targets.ndim != 1 or len(targets) != len(values):
        raise ValueError("targets and values must align")
    if not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("targets must be integer indices")
    if np.any(targets < 0) or np.any(targets >= len(scores)):
        raise ValueError("target out of range")
    observed = scores[targets]
    positive = observed > 0
    if not positive.any():
        return 0.0
    with np.errstate(over="ignore", divide="ignore"):
        breaks = values[positive] / observed[positive]
    if not np.all(np.isfinite(breaks)):
        raise ValueError("breakpoint overflow; rescale scores before fitting")
    candidates = np.unique(np.r_[0.0, breaks, np.nextafter(breaks, 0.0)])
    best_scale, best_revenue = 0.0, -1.0
    for alpha in candidates:
        with np.errstate(over="ignore"):
            cost = alpha * observed
        revenue = float(cost[cost <= values].sum())
        if revenue > best_revenue:
            best_scale, best_revenue = float(alpha), revenue
    return best_scale


def partition_entropy(answer_labels, weights=None):
    """Shannon entropy of supplied answer classes, NOT just changed/not-changed flags.

    Callers must construct collision-safe equivalence labels respecting their
    chosen bag, NULL, type, and numerical semantics. This helper does not do so.
    """
    labels = list(answer_labels)
    if not labels:
        raise ValueError("at least one support point required")
    weights = (np.ones(len(labels)) if weights is None
               else _nonnegative_vector(weights, "weights"))
    if len(weights) != len(labels) or weights.sum() <= 0:
        raise ValueError("positive total weight and one weight per label required")
    groups = defaultdict(float)
    for label, weight in zip(labels, weights):
        groups[label] += float(weight)
    probabilities = np.array([w for w in groups.values() if w > 0]) / weights.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def verified_effective_prices(posted, cover_costs, verified):
    """Only explicitly verified cheaper plans can alter prices or arbitrage metrics."""
    posted = _nonnegative_vector(posted, "posted")
    cover_costs = np.asarray(cover_costs, dtype=float)
    if cover_costs.shape != posted.shape or len(verified) != len(posted):
        raise ValueError("one cost and verification outcome per asset required")
    if np.isnan(cover_costs).any() or np.any(cover_costs < 0):
        raise ValueError("cover costs must be nonnegative or positive infinity")
    valid = np.array([isinstance(v, (bool, np.bool_)) and bool(v) for v in verified])
    valid &= np.isfinite(cover_costs) & (cover_costs < posted)
    out = posted.copy()
    out[valid] = cover_costs[valid]
    return out


def delete_one_seeded(con, table, rng):
    """DuckDB base-table fixture sampler: delete exactly one physical row.

    The caller owns BEGIN/ROLLBACK. Seed reproducibility is for the same fixed
    physical table snapshot. Durable row IDs are still needed for production
    rebuilds. A real column named rowid is rejected to avoid alias ambiguity.
    """
    def quote(identifier):
        return '"' + str(identifier).replace('"', '""') + '"'
    if len(table) != 2:
        raise ValueError("table must be (schema, table_name)")
    name = ".".join(quote(x) for x in table)
    cols = [d[0].lower() for d in con.execute(f"SELECT * FROM {name} LIMIT 0").description]
    if "rowid" in cols:
        raise ValueError("explicit rowid column requires a dedicated stable-ID strategy")
    count = int(con.execute(f"SELECT count(*) FROM {name}").fetchone()[0])
    if count == 0:
        return None
    offset = int(rng.integers(count))
    row_id = con.execute(f"SELECT rowid FROM {name} ORDER BY rowid LIMIT 1 OFFSET ?",
                         [offset]).fetchone()[0]
    affected = con.execute(f"DELETE FROM {name} WHERE rowid = ?", [row_id]).fetchone()[0]
    if affected != 1:
        raise RuntimeError(f"expected one deleted row, observed {affected}")
    return int(row_id)
