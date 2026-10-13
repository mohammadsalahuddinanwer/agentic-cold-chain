"""
sensitivity_analysis.py
=======================
Re-costs a saved ablation run (no re-running) to show how the conclusions
depend on two modelling choices:
  1. what a false alarm triggers (escalation, driver check, or log only), and
  2. the share of normal (anomaly-free) segments in the case mix.
It also reports the break-even points at which the Monitor starts to pay off
against the reactive rule (A0 vs A3).

Anomalous-case costs are taken from the run unchanged; only the cost of false
alarms on normal segments and the mix are varied.


"""
import argparse

import numpy as np
import pandas as pd

from run_ablation_study import ACTION_COSTS, B1_RAW, COST_WEIGHTS, RESOLVER_LATENCY_S, ci95

FA_RESPONSES = ["escalate_to_human", "notify_driver", "monitor_only"]
NORMAL_SHARES = [0.50, 0.70, 0.85]


def false_alarm_cost(action):
    """Normalised cost of one false alarm handled with `action` (no spoilage)."""
    carbon = 0.0 if action == "monitor_only" else 0.05
    raw = COST_WEIGHTS @ np.array([0.0, ACTION_COSTS[action],
                                   min(1.0, RESOLVER_LATENCY_S / 600.0), carbon])
    return 100.0 * raw / B1_RAW


def main():
    ap = argparse.ArgumentParser(description="Sensitivity of ablation costs")
    ap.add_argument("--cases", default="outputs/ablation_v3_cases.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.cases)
    conds = [c for c in ["A0_Full_System", "A1_Remove_RAG", "A2_Remove_EvoOpt",
                         "A3_Remove_Monitor", "A4_Replace_LLM_RuleBased"]
             if c in set(df.condition)]
    g = df.groupby(["condition", "seed"])
    anom = df[df.has_anomaly].groupby(["condition", "seed"]).cost.mean()
    fa = df[~df.has_anomaly].groupby(["condition", "seed"]).alarmed.mean()
    seeds = sorted(df.seed.unique())

    print("=" * 78)
    print(f"SENSITIVITY ANALYSIS  ({args.cases}, {len(seeds)} seeds)")
    print("=" * 78)
    print("False-alarm rate on normal segments:  " +
          ", ".join(f"{c.split('_')[0]} {fa[c].mean():.1%}" for c in conds))
    print("Cost of ONE false alarm:  " +
          ", ".join(f"{a} {false_alarm_cost(a):.1f}" for a in FA_RESPONSES))

    # consistency check against the run itself (escalation, 50/50 mix)
    check = g.cost.mean()["A0_Full_System"].mean()
    recon = np.mean([0.5 * anom["A0_Full_System"][s]
                     + 0.5 * fa["A0_Full_System"][s] * false_alarm_cost("escalate_to_human")
                     for s in seeds])
    print(f"Check: A0 cost in run {check:.2f}, reconstructed {recon:.2f} (should match)")

    rows = []
    for action in FA_RESPONSES:
        c_fa = false_alarm_cost(action)
        print(f"\nFalse alarm -> {action}  (cost {c_fa:.1f} per false alarm)")
        print(f"  {'condition':<26}" + "".join(f"{'normal share ' + format(p, '.0%'):>26}"
                                             for p in NORMAL_SHARES))
        for cond in conds:
            cells = []
            for p in NORMAL_SHARES:
                per_seed = [(1 - p) * anom[cond][s] + p * fa[cond][s] * c_fa for s in seeds]
                base = [(1 - p) * anom["A0_Full_System"][s]
                        + p * fa["A0_Full_System"][s] * c_fa for s in seeds]
                m, lo, hi = ci95(per_seed)
                pct = ci95([100 * (x - b) / b for x, b in zip(per_seed, base)])
                cells.append(f"{m:6.1f}" + ("" if cond == "A0_Full_System"
                                            else f" ({pct[0]:+5.1f}% vs A0)"))
                rows.append(dict(false_alarm_response=action, normal_share=p, condition=cond,
                                 cost=m, cost_ci_low=lo, cost_ci_high=hi, pct_vs_A0=pct[0],
                                 pct_ci_low=pct[1], pct_ci_high=pct[2]))
            print(f"  {cond:<26}" + "".join(f"{c:>26}" for c in cells))

        a0, a3 = anom["A0_Full_System"].mean(), anom["A3_Remove_Monitor"].mean()
        n0 = fa["A0_Full_System"].mean() * c_fa
        if a3 <= a0:
            print("  Break-even: the Monitor never pays off (no saving on anomalies).")
        else:
            p_star = (a3 - a0) / (a3 - a0 + n0) if n0 > 0 else 1.0
            print(f"  Break-even: the Monitor pays off while normal segments are below "
                  f"{p_star:.0%} of cases")
    a0, a3 = anom["A0_Full_System"].mean(), anom["A3_Remove_Monitor"].mean()
    f0 = fa["A0_Full_System"].mean()
    if a3 > a0 and f0 > 0:
        print(f"\nAt a 50/50 mix, the Monitor pays off if one false alarm costs less than "
              f"{(a3 - a0) / f0:.1f} points (saving per anomaly {a3 - a0:.1f}, "
              f"false-alarm rate {f0:.1%}).")

    out = args.cases.replace("_cases.csv", "_sensitivity.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
