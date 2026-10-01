"""Draw Figures 1 and 2 of the paper from the data that reproduce/make_tables.py writes.

    python reproduce/make_tables.py      # first: writes reproduce/out/*.csv
    python reproduce/make_figures.py     # then: writes reproduce/out/fig_taxonomy.pdf
                                         #       and reproduce/out/fig_results.pdf

fig_taxonomy.pdf  Figure 1, the taxonomy of the evaluated mechanisms (no data).
fig_results.pdf   Figure 2. Panel (a) averages the TPC-H and SQLShare values of Table 4,
                  rounded as printed; panel (b) is the per-market data of Table 8
                  (figure2b_markets.csv); panel (c) is Table 7.
"""
from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
ASSETS = OUT

# Okabe-Ito colorblind-safe palette; shape and fill always carry the distinction too.
BLUE, VERMILLION, ORANGE, GREY = "#0072B2", "#D55E00", "#E69F00", "#6E6E6E"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

COLUMN_IN, TEXT_IN = 3.33, 7.0


def _rows(name):
    with open(os.path.join(OUT, name), newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_cross():
    """Panel (a): the mean of the TPC-H and SQLShare values of Table 4, as printed."""
    rows = {}
    for r in _rows("table4_cross.csv"):
        name = r["mechanism"] + ("\u2020" if r["mechanism"].startswith("QueryMarket") else "")
        rev_t, rev_s, wel_t, wel_s = (round(float(r[k]), 3) for k in
                                      ("revenue_tpch", "revenue_sqlshare", "welfare_tpch",
                                       "welfare_sqlshare"))
        rows[name] = ((rev_t + rev_s) / 2, (wel_t + wel_s) / 2)
    return rows


def read_sampling():
    """Panel (c): priceable assets and price levels under the two samplers (Table 7)."""
    rows = {r["row"].split(" (")[0]: r for r in _rows("table7_sampling.csv")}
    total = int(next(k for k in (r["row"] for r in _rows("table7_sampling.csv"))
                     if k.startswith("priceable")).split("of ")[1].rstrip(")"))
    names = ["size proportional (as specified)", "readership weighted"]
    cols = ["size_proportional", "readership_weighted"]
    return [{"name": names[i], "priced": int(float(rows["priceable assets"][cols[i]])),
             "total": total, "levels": int(float(rows["price levels"][cols[i]])),
             "build": float(rows["build time"][cols[i]])} for i in range(2)]


def read_markets():
    """Panel (b): per market, the better published construction against the better flat price."""
    return [{"market": f"rank {r['rank']}",
             "welfare_baseline": float(r["best_flat_welfare"]),
             "welfare_published": float(r["best_published_welfare"]),
             "revenue_baseline": float(r["best_flat_revenue"]),
             "revenue_published": float(r["best_published_revenue"])}
            for r in _rows("figure2b_markets.csv")]


def taxonomy():
    width_pt, height_pt = 240.0, 162.0
    fig = plt.figure(figsize=(width_pt / 72, height_pt / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_pt)
    ax.set_ylim(0, height_pt)
    ax.axis("off")

    groups = [
        (BLUE, "#E6F0F8", "Information-proportional pricing (4)",
         "price grows with what the answer reveals",
         ["Conflict-set pricing (Qirana): weighted coverage,",
          "uniform gain, entropy",
          "View-cover pricing (QueryMarket): cheapest",
          "determining set of priced views"]),
        (ORANGE, "#FDF3E1", "Revenue-maximizing pricing (6)",
         "price fitted to revealed buyer valuations",
         ["Chawla et al.: UBP, UIP, LPIP, CIP, layering, XOS"]),
        (GREY, "#F0F0F0", "Baselines (5)",
         "no attribution; fitted on the same buyers",
         ["flat fee, size-proportional, compute-metered,",
          "per-asset monopoly, monotone fitted"]),
    ]
    step, pad, gap = 9.5, 5.0, 6.0
    top = height_pt - 2.0
    for edge, face, title, subtitle, lines in groups:
        n = 2 + len(lines)
        h = n * step + pad
        y0 = top - h
        ax.add_patch(FancyBboxPatch((3, y0), width_pt - 6, h,
                                    boxstyle="round,pad=0,rounding_size=4",
                                    linewidth=1.0, edgecolor=edge, facecolor=face))
        y = top - pad / 2 - step / 2 - 1.0
        ax.text(9, y, title, ha="left", va="center", fontsize=7.5, fontweight="bold")
        y -= step
        ax.text(9, y, subtitle, ha="left", va="center", fontsize=7, style="italic",
                color="#333333")
        for line in lines:
            y -= step
            ax.text(15, y, line, ha="left", va="center", fontsize=7)
        top = y0 - gap

    fig.savefig(os.path.join(ASSETS, "fig_taxonomy.pdf"), bbox_inches=None)
    plt.close(fig)


def results():
    cross = read_cross()
    markets = read_markets()
    sampling = read_sampling()

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_IN, 2.45),
                             gridspec_kw={"width_ratios": [1.55, 1.0, 1.0], "wspace": 0.42})

    # (a) revenue and welfare per mechanism, sorted by revenue.
    ax = axes[0]
    items = sorted(cross.items(), key=lambda kv: kv[1][0])
    for i, (name, (rev, wel)) in enumerate(items):
        ax.plot([rev, wel], [i, i], color="#BBBBBB", linewidth=0.8, zorder=1)
        ax.scatter(rev, i, s=16, facecolors="white", edgecolors=VERMILLION,
                   linewidths=1.0, zorder=3, label="revenue" if i == 0 else None)
        ax.scatter(wel, i, s=16, marker="s", color=BLUE, zorder=3,
                   label="welfare" if i == 0 else None)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([name for name, _ in items])
    ax.set_xlim(0, 1.7)
    ax.set_ylim(-0.7, len(items) - 0.3)
    ax.set_xlabel("normalized by per-asset monopoly revenue")
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False,
              handletextpad=0.3, columnspacing=1.2)

    # (b) per-market best published construction against the best flat price.
    ax = axes[1]
    xw = [m["welfare_baseline"] for m in markets]
    yw = [m["welfare_published"] for m in markets]
    xr = [m["revenue_baseline"] for m in markets]
    yr = [m["revenue_published"] for m in markets]
    hi = 1.05 * max(xw + yw + xr + yr)
    ax.plot([0, hi], [0, hi], linestyle="--", color=GREY, linewidth=0.7, zorder=1)
    ax.scatter(xw, yw, s=16, marker="s", color=BLUE, zorder=3, label="welfare")
    ax.scatter(xr, yr, s=16, facecolors="white", edgecolors=VERMILLION, linewidths=1.0,
               zorder=3, label="revenue")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("flat fee or UBP, better of the two")
    ax.set_ylabel("Qirana w.c. or QueryMarket, better")
    ax.text(0.05 * hi, 0.95 * hi, "published higher\nabove the diagonal",
            ha="left", va="top", fontsize=6.5, color="#333333")
    ax.legend(loc="lower right", frameon=False, handletextpad=0.3, borderaxespad=0.2)

    # (c) support-set sampling: priceable assets and distinct price levels.
    ax = axes[2]
    names = ["size-\nproportional", "readership-\nweighted"]
    xs = [0, 1]
    width = 0.36
    priced = [s["priced"] for s in sampling]
    levels = [s["levels"] for s in sampling]
    total = sampling[0]["total"]
    b1 = ax.bar([x - width / 2 for x in xs], priced, width, color=BLUE, edgecolor=BLUE,
                label="priceable assets")
    b2 = ax.bar([x + width / 2 for x in xs], levels, width, color="white", edgecolor=ORANGE,
                hatch="////", linewidth=0.8, label="price levels")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4,
                    f"{int(bar.get_height())}", ha="center", va="bottom", fontsize=7)
    ax.axhline(total, linestyle=":", color=GREY, linewidth=0.8)
    ax.text(-0.45, total + 5, f"{total} assets in the market", ha="left", va="bottom",
            fontsize=6.5, color="#333333")
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 255)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylabel("count")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.25), ncol=2, frameon=False,
              handlelength=1.4, handletextpad=0.3, columnspacing=0.8, fontsize=6.5)

    for label, ax in zip("abc", axes):
        ax.text(-0.02, 1.04, f"({label})", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, fontweight="bold")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.savefig(os.path.join(ASSETS, "fig_results.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(ASSETS, exist_ok=True)
    taxonomy()
    results()
    print(f"wrote {os.path.join(ASSETS, 'fig_taxonomy.pdf')} and fig_results.pdf")
