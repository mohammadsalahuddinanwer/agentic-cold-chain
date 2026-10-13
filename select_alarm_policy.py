"""
select_alarm_policy.py
======================
Chooses the Monitor's alarm policy on VALIDATION shipments (seeds 2000+), which
are separate from both the training data (seed 42) and the test shipments used
by run_ablation_study.py (seeds 1000+). The chosen policy is then tested once.

Candidate policies
  trigger    classifier : probability >= threshold
             forecast   : +30 min temperature forecast > product's critical limit
             both       : classifier AND forecast
             either     : classifier OR forecast
  persist    alarm only after the trigger holds for k consecutive scans (5 min apart)

Selection criterion (fixed before looking at any test result)
  lowest mean cost of the full system (A0) on validation shipments, with the same
  cost model and case mix as run_ablation_study.py. Costs less than 0.1 point
  apart count as ties, and ties go to the simpler policy (fewer scans first).


"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_ablation_study import (Config, DEFAULT_POLICY, DIAG_ACCURACY, RESOLVER_LATENCY_S,
                                apply_policy, build_cases, detect, load_monitor,
                                load_training_scaling, policy_tag, run_condition)

VAL_BASE_SEED = 2000


def candidate_policies():
    grid = []
    for t in [0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94]:
        for k in [1, 2, 3]:
            grid.append({"trigger": "classifier", "threshold": t, "persist": k})
    for k in [1, 2, 3]:
        grid.append({"trigger": "forecast", "threshold": 0.80, "persist": k})
    for t in [0.80, 0.85, 0.90]:
        for k in [1, 2]:
            grid.append({"trigger": "both", "threshold": t, "persist": k})
            grid.append({"trigger": "either", "threshold": t, "persist": k})
    return grid


def complexity(p):
    """Simpler first: fewer scans, then classifier < forecast < both < either."""
    return (p["persist"], ["classifier", "forecast", "both", "either"].index(p["trigger"]))


def main():
    ap = argparse.ArgumentParser(description="Select the Monitor alarm policy on validation seeds")
    ap.add_argument("--val_seeds", type=int, default=5)
    ap.add_argument("--n_anomalous", type=int, default=100)
    ap.add_argument("--n_normal", type=int, default=100)
    args = ap.parse_args()

    out = Path("outputs")
    out.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ws = Config.MONITOR_WINDOW_SIZE
    scaler, y_mean, y_std, n_feat = load_training_scaling()
    model = load_monitor(n_feat, device)

    print("=" * 70)
    print(f"ALARM POLICY SELECTION on validation seeds "
          f"{VAL_BASE_SEED}-{VAL_BASE_SEED + args.val_seeds - 1}")
    print("=" * 70)
    per_seed_cases = []
    for seed in range(VAL_BASE_SEED, VAL_BASE_SEED + args.val_seeds):
        cases = detect(build_cases(seed, args.n_anomalous, args.n_normal, ws),
                       model, scaler, ws, y_mean, y_std, device)
        per_seed_cases.append(cases)
        print(f"seed {seed}: {sum(c['has_anomaly'] for c in cases)} events + "
              f"{sum(not c['has_anomaly'] for c in cases)} normal segments scored")

    rule_cost = np.mean([pd.DataFrame(run_condition(
        cs, "rule", DIAG_ACCURACY["rag"], RESOLVER_LATENCY_S)).cost.mean() for cs in per_seed_cases])

    rows = []
    for p in candidate_policies():
        seed_costs, frames = [], []
        for cs in per_seed_cases:
            d = pd.DataFrame(run_condition(apply_policy(cs, p), "monitor",
                                           DIAG_ACCURACY["rag"], RESOLVER_LATENCY_S))
            seed_costs.append(d.cost.mean())
            frames.append(d)
        d = pd.concat(frames)
        br, nm = d[d.breaching], d[~d.has_anomaly]
        lead = br.loc[br.alarmed, "lead_before_breach_min"]
        rows.append(dict(policy=policy_tag(p), **p, cost=np.mean(seed_costs),
                         cost_anomalous=d[d.has_anomaly].cost.mean(), cost_normal=nm.cost.mean(),
                         false_alarm_rate=nm.alarmed.mean(), breach_alarmed=br.alarmed.mean(),
                         alarmed_before_breach=(lead > 0).sum() / max(len(br), 1),
                         median_lead_min=lead.median()))
    res = pd.DataFrame(rows)
    res["simplicity"] = [complexity(p) for p in candidate_policies()]
    res["cost_rounded"] = res.cost.round(1)       # < 0.1 point apart counts as a tie
    res = res.sort_values(["cost_rounded", "simplicity"]).reset_index(drop=True)
    res.drop(columns=["simplicity", "cost_rounded"]).to_csv(out / "alarm_policy_validation.csv", index=False)

    best = res.iloc[0]
    chosen = {"trigger": best.trigger, "threshold": float(best.threshold),
              "persist": int(best.persist)}
    (out / "alarm_policy.json").write_text(json.dumps(chosen, indent=2))

    show = res.head(10)
    default_row = res[res.policy == policy_tag(DEFAULT_POLICY)]
    print(f"\n{'policy':<26}{'cost':>7}{'anom.':>7}{'normal':>8}{'false al.':>10}"
          f"{'breach al.':>11}{'before br.':>11}{'lead med':>9}")
    print("-" * 89)
    for r in pd.concat([show, default_row]).drop_duplicates("policy").itertuples():
        mark = "  <- chosen" if r.policy == best.policy else (
               "  <- thesis default" if r.policy == policy_tag(DEFAULT_POLICY) else "")
        print(f"{r.policy:<26}{r.cost:7.1f}{r.cost_anomalous:7.1f}{r.cost_normal:8.1f}"
              f"{r.false_alarm_rate:10.1%}{r.breach_alarmed:11.1%}{r.alarmed_before_breach:11.1%}"
              f"{r.median_lead_min:9.1f}{mark}")
    print(f"\nReference, no Monitor (A3, reactive rule): cost {rule_cost:.1f}")
    print(f"\nChosen policy: {policy_tag(chosen)}  ->  saved to {out/'alarm_policy.json'}")
    print(f"Full table: {out/'alarm_policy_validation.csv'}")
    print("Next: test it ONCE on seeds 1000+ with run_ablation_study.py --policy_file ...")


if __name__ == "__main__":
    main()
