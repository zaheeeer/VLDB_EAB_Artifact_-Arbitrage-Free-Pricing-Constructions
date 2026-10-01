"""Stress tests for the TPC-H derivability oracle.

Three checks, each of which can invalidate a headline number if it fails.

1. Confusion matrix of the structural candidate rule against brute-force verification.
   Every ordered pair gets a best-effort reconstruction attempt, whether or not the rule
   nominates it. This yields recall as well as precision: a rule that is sound but narrow
   understates arbitrage, and the paper must report which.

2. Negative control on the verifier. Pairs whose reconstruction is attempted and whose
   answers genuinely differ must be rejected. If the comparison were permissive the
   reported precision would be vacuous.

3. Tolerance calibration. A verified source answer is perturbed by a relative epsilon and
   the smallest epsilon the verifier rejects is recorded. A tolerance looser than the
   perturbations arising from re-aggregation order would accept wrong answers.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd


def derive_forced(answers, assets, i, j, TEMPLATES, ATTR=None):
    """Best-effort roll-up reconstruction of asset j from asset i, ignoring the rule.

    Returns None when the reconstruction is structurally impossible, which is itself a
    negative result rather than an error.
    """
    src, tgt = assets[i], assets[j]
    if src.template != tgt.template:
        return None
    df = answers[src]
    gcols_src = [g for g in TEMPLATES[src.template]["group_dims"] if g in src.group]
    gcols_tgt = [g for g in TEMPLATES[tgt.template]["group_dims"] if g in tgt.group]
    mcols = list(tgt.measures)

    if not set(gcols_tgt) <= set(gcols_src):
        return None
    if tgt.months != src.months:
        if "shipmonth" not in gcols_src:
            return None
        want = pd.to_datetime(sorted(tgt.months))
        keep = pd.to_datetime(df["shipmonth"]).isin(want)
        df = df[keep.to_numpy()]
        if df.empty:
            return None
    if not set(mcols) <= set(df.columns):
        return None
    if gcols_tgt:
        out = df.groupby(gcols_tgt, dropna=False, as_index=False)[mcols].sum()
        out = out.sort_values(gcols_tgt).reset_index(drop=True)
    else:
        out = pd.DataFrame([df[mcols].sum()])
    return out.loc[:, gcols_tgt + mcols]


def confusion(answers, assets, determines_fn, match_fn, TEMPLATES, limit=None):
    n = len(assets) if limit is None else min(limit, len(assets))
    tp = fp = fn = tn = 0
    fn_examples, fp_examples = [], []
    t0 = time.perf_counter()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            cand = determines_fn(assets[i], assets[j])
            got = derive_forced(answers, assets, i, j, TEMPLATES)
            ver = got is not None and match_fn(got, answers[assets[j]])
            if cand and ver:
                tp += 1
            elif cand and not ver:
                fp += 1
                if len(fp_examples) < 5:
                    fp_examples.append((i, j))
            elif not cand and ver:
                fn += 1
                if len(fn_examples) < 5:
                    fn_examples.append((i, j))
            else:
                tn += 1
    return {
        "n_assets": n,
        "pairs_tested": n * (n - 1),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "seconds": time.perf_counter() - t0,
        "fp_examples": fp_examples,
        "fn_examples": fn_examples,
    }


def tolerance_curve(answers, assets, verified_pairs, derive_fn, match_fn, n_pairs=20,
                    epsilons=(0.0, 1e-12, 1e-10, 1e-9, 1e-8, 1e-6, 1e-4, 1e-2)):
    """Smallest relative perturbation the verifier rejects, over sampled verified pairs."""
    rows = []
    for (i, j) in verified_pairs[:n_pairs]:
        src, tgt = assets[i], assets[j]
        base = answers[src]
        for eps in epsilons:
            pert = base.copy()
            num = [c for c in pert.columns if pd.api.types.is_numeric_dtype(pert[c])]
            if not num:
                continue
            pert[num] = pert[num] * (1.0 + eps)
            got = derive_fn(pert, src, tgt)
            rows.append({"pair": f"{i}->{j}", "eps": eps,
                         "accepted": bool(got is not None and match_fn(got, answers[tgt]))})
    return pd.DataFrame(rows)
