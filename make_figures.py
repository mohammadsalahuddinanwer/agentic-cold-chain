#!/usr/bin/env python3
"""generate Fig. 2 and Fig. 3 for the LNCS paper.


"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent / "outputs"
DEST = Path(__file__).resolve().parent / "figures"

BLUE, ORANGE = "#4472c4", "#ed7d31"
GRID, INK, MUTED = "#e4e4e4", "#000000", "#5a5a5a"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.edgecolor": INK,
    "axes.linewidth": 1.1,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

# ── Fig. 2 — detection lead time by failure mode (Monitor at 0.80) ───────
def fig2():
    # from run_ablation_study.py console output at threshold 0.80, 10 seeds
    rows = [
        ("power\noutage",        0,  100),
        ("compressor\nfailure",  5,  100),
        ("door left\nopen",      12,  95),
        ("driver\nerror",        12, 100),
        ("gps\nblackout",        25,  39),
        ("extreme\nweather",     31,  99),
    ]
    labels = [r[0] for r in rows]
    lead   = [r[1] for r in rows]
    alarm  = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(9.195, 3.50), dpi=200)
    x = range(len(rows))
    ax.bar(x, lead, width=0.62, color=BLUE, edgecolor=INK, linewidth=0.9, zorder=3)

    ax.axhline(30, color="#c00000", linestyle="--", linewidth=1.3, zorder=2)
    ax.text(-0.42, 30.9, "30 min target", color="#c00000", fontsize=13,
            va="bottom", ha="left")

    for xi, v in zip(x, lead):
        ax.text(xi, v + 0.9, f"{v} min", ha="center", va="bottom",
                fontsize=14, color=INK)
    # the axes extend below zero; the "% alarmed" row lives in that band,
    # inside the axes and above the bottom spine, as in the original figure
    for xi, a in zip(x, alarm):
        ax.text(xi, -2.9, f"{a}% alarmed", ha="center", va="center",
                fontsize=13.5, color=MUTED)

    ax.set_ylabel("Lead before breach (min)", fontsize=15)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=15)
    ax.tick_params(axis="x", length=6, width=1.1, pad=8)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylim(-5.4, 36)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_yticks([0, 10, 20, 30])
    ax.yaxis.grid(True, color=GRID, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.subplots_adjust(left=0.085, right=0.995, top=0.97, bottom=0.255)
    fig.savefig(DEST / "fig2_lead_time.png", dpi=200, facecolor="white")
    plt.close(fig)


# ── Fig. 3 — normalised disruption cost by configuration ─────────────────
def fig3():
    a = pd.read_csv(OUT / "baselines_v3_summary.csv").set_index("baseline")
    b = pd.read_csv(OUT / "baselines_v3_classifier_t0.86_k1_summary.csv").set_index("baseline")
    keys = ["B1_Reactive_Rule", "B2_Centralised_Opt", "B3_Detection_Only",
            "B4_MAS_No_LLM", "B5_Full_System"]
    labels = ["B1\nreactive\n(manual)", "B2\ncentralised\n(type known)",
              "B3\ndetection\nonly", "B4\nrule +\nkeywords", "B5\nfull\nsystem"]

    def vals(df):
        m = [df.loc[k, "cost_mean"] for k in keys]
        e = [(df.loc[k, "cost_ci_high"] - df.loc[k, "cost_ci_low"]) / 2 for k in keys]
        return m, e

    m80, e80 = vals(a)
    m86, e86 = vals(b)
    print("  Fig. 3 values read from the output files:")
    for k, v1, v2 in zip(keys, m80, m86):
        print(f"    {k:22s} 0.80 {v1:6.3f} -> {v1:.1f}   0.86 {v2:6.3f} -> {v2:.1f}")

    fig, ax = plt.subplots(figsize=(9.195, 4.30), dpi=200)
    x = range(len(keys))
    w = 0.38
    bar_kw = dict(edgecolor=INK, linewidth=0.9, zorder=3,
                  error_kw=dict(ecolor=INK, elinewidth=2.0, capsize=5, capthick=2.0, zorder=4))
    ax.bar([i - w / 2 for i in x], m80, w, yerr=e80, color=BLUE,
           label="Monitor at 0.80 (design)", **bar_kw)
    ax.bar([i + w / 2 for i in x], m86, w, yerr=e86, color=ORANGE,
           label="Monitor at 0.86 (selected)", **bar_kw)

    for i, (v, e) in enumerate(zip(m80, e80)):
        ax.text(i - w / 2, v + e + 0.9, f"{v:.1f}", ha="center", va="bottom", fontsize=15)
    for i, (v, e) in enumerate(zip(m86, e86)):
        ax.text(i + w / 2, v + e + 0.9, f"{v:.1f}", ha="center", va="bottom", fontsize=15)

    ax.set_ylabel("Normalised cost", fontsize=16)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=15)
    ax.tick_params(axis="x", length=0, pad=10)
    ax.tick_params(axis="y", labelsize=15)
    ax.set_ylim(0, 43)
    ax.set_xlim(-0.6, len(keys) - 0.4)
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.yaxis.grid(True, color=GRID, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.015), ncol=2,
              frameon=False, fontsize=15.5, handlelength=1.6,
              columnspacing=2.4, handletextpad=0.7)

    fig.subplots_adjust(left=0.095, right=0.995, top=0.875, bottom=0.235)
    fig.savefig(DEST / "fig3_disruption_cost.png", dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    DEST.mkdir(exist_ok=True)
    fig2()
    fig3()
    from PIL import Image
    for n, want in (("fig2_lead_time.png", (1839, 700)), ("fig3_disruption_cost.png", (1839, 860))):
        got = Image.open(DEST / n).size
        print(f"  {n}: {got}   original {want}   {'MATCH' if got == want else 'DIFFERENT'}")
