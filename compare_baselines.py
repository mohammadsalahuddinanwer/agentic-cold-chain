"""
compare_baselines.py  (v3: same cases, cost model and random draws as the ablation)
===================================================================================
Evaluates baselines B1-B5 on the SAME event-based test shipments as
run_ablation_study.py (ColdChainDataGenerator, seeds 1000+), with the same cost
model and the same per-case random draw. B5 is identical to ablation A0, which
is checked at the end.

Baselines (measured components wherever a measurement exists)
  B1 Reactive rule     : critical-limit rule alarms at the breach; alert handled
                         manually (human response, no automated diagnosis/resolution)
  B2 Centralised opt.  : rule alarm; anomaly type assumed KNOWN (optimistic
                         reference, as stated in the thesis); automated response
  B3 Detection-only    : Monitor alarm + LLM/RAG diagnosis; no Resolver, so the
                         response is handled manually
  B4 MAS without LLM   : rule alarm; keyword classifier (accuracy measured on the
                         120-case diagnosis test set); automated response
  B5 Full system       : Monitor alarm + LLM/RAG diagnosis + automated Resolver


"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from run_ablation_study import (ACTION_MAP, BASE_SEED, Config, DEFAULT_POLICY, DIAG_ACCURACY,
                                FALSE_ALARM_ACTION, RESOLVER_LATENCY_S, apply_policy,
                                build_cases, ci95, compute_cost, detect, load_monitor,
                                load_training_scaling, measure_rule_accuracy, policy_tag)

HUMAN_LATENCY_S = 600.0   # manual handling; the delay term is capped at 600 s,
                          # so any human response of 10 min or more costs the same


def run_baseline(cases, detector, accuracy, automated):
    """detector: 'monitor' or 'rule'; accuracy: probability the diagnosis is
    correct (1.0 = type known); automated: Resolver acts, otherwise a human."""
    rows = []
    for c in cases:
        alarm = c["mon_alarm"] if detector == "monitor" else c["rule_alarm"]
        if alarm is None:
            action, latency = "no_action", 0.0
        elif not automated:
            action, latency = "escalate_to_human", HUMAN_LATENCY_S
        elif not c["has_anomaly"]:
            action, latency = FALSE_ALARM_ACTION, RESOLVER_LATENCY_S
        elif c["u"] < accuracy:                    # same draw as in the ablation
            action, latency = ACTION_MAP[c["anomaly_type"]], RESOLVER_LATENCY_S
        else:
            action, latency = "escalate_to_human", RESOLVER_LATENCY_S
        cost, lead = compute_cost(c, alarm, action, latency)
        rows.append(dict(case_id=c["case_id"], has_anomaly=c["has_anomaly"],
                         anomaly_type=c["anomaly_type"], breaching=c["breach"] is not None,
                         alarmed=alarm is not None, lead_before_breach_min=lead,
                         action=action, cost=cost))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Baseline comparison (v3)")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n_anomalous", type=int, default=100)
    ap.add_argument("--n_normal", type=int, default=100)
    ap.add_argument("--diag_test_set", type=str, default=None,
                    help="held-out diagnosis test set, needed for B4; "
                         "use diagnosis_test_set_100.json (120 cases)")
    ap.add_argument("--policy_file", type=str, default=None,
                    help="Monitor alarm policy JSON (default: classifier, 0.80, 1 scan)")
    args = ap.parse_args()

    out = Path("outputs")
    out.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ws = Config.MONITOR_WINDOW_SIZE
    policy = (json.loads(Path(args.policy_file).read_text()) if args.policy_file
              else DEFAULT_POLICY)
    tag = "" if args.policy_file is None else "_" + policy_tag(policy)

    print("=" * 70)
    print("BASELINE COMPARISON v3: same cases and cost model as the ablation")
    print("=" * 70)
    scaler, y_mean, y_std, n_feat = load_training_scaling()
    model = load_monitor(n_feat, device)
    print(f"Monitor alarm policy: {policy_tag(policy)}, device={device}")

    baselines = {
        "B1_Reactive_Rule":   dict(detector="rule", accuracy=0.0, automated=False),
        "B2_Centralised_Opt": dict(detector="rule", accuracy=1.0, automated=True),
        "B3_Detection_Only":  dict(detector="monitor", accuracy=DIAG_ACCURACY["rag"],
                                   automated=False),
        "B5_Full_System":     dict(detector="monitor", accuracy=DIAG_ACCURACY["rag"],
                                   automated=True),
    }
    if args.diag_test_set:
        acc, n, _, _ = measure_rule_accuracy(args.diag_test_set)
        print(f"B4 keyword classifier accuracy: {acc:.4f} on {n} cases (measured)")
        baselines["B4_MAS_No_LLM"] = dict(detector="rule", accuracy=acc, automated=True)
    else:
        print("B4 skipped: pass --diag_test_set to measure the keyword classifier.")
    order = [b for b in ["B1_Reactive_Rule", "B2_Centralised_Opt", "B3_Detection_Only",
                         "B4_MAS_No_LLM", "B5_Full_System"] if b in baselines]

    rows = []
    for seed in range(BASE_SEED, BASE_SEED + args.seeds):
        cases = build_cases(seed, args.n_anomalous, args.n_normal, ws)
        cases = apply_policy(detect(cases, model, scaler, ws, y_mean, y_std, device), policy)
        print(f"seed {seed}: {sum(c['has_anomaly'] for c in cases)} events + "
              f"{sum(not c['has_anomaly'] for c in cases)} normal segments")
        for name in order:
            for r in run_baseline(cases, **baselines[name]):
                rows.append(dict(seed=seed, baseline=name, **r))
    df = pd.DataFrame(rows)
    df.to_csv(out / f"baselines_v3{tag}_cases.csv", index=False)

    per_seed = pd.DataFrame({
        "cost": df.groupby(["baseline", "seed"]).cost.mean(),
        "cost_anomalous": df[df.has_anomaly].groupby(["baseline", "seed"]).cost.mean(),
        "cost_normal": df[~df.has_anomaly].groupby(["baseline", "seed"]).cost.mean(),
    }).reset_index()
    b1 = per_seed[per_seed.baseline == "B1_Reactive_Rule"].set_index("seed")["cost"]
    per_seed["reduction_vs_B1"] = [100.0 * (b1[s] - c) / b1[s]
                                   for c, s in zip(per_seed.cost, per_seed.seed)]

    summary = []
    for name in order:
        s = per_seed[per_seed.baseline == name]
        m, lo, hi = ci95(s.cost)
        r, rlo, rhi = ci95(s.reduction_vs_B1)
        summary.append(dict(baseline=name, cost_mean=m, cost_ci_low=lo, cost_ci_high=hi,
                            reduction_vs_B1=r, reduction_ci_low=rlo, reduction_ci_high=rhi,
                            cost_anomalous=s.cost_anomalous.mean(),
                            cost_normal=s.cost_normal.mean(), n_seeds=len(s)))
    summary = pd.DataFrame(summary)
    summary.to_csv(out / f"baselines_v3{tag}_summary.csv", index=False)

    print(f"\nBASELINE RESULTS ({args.seeds} seeds; mean [95% CI] across seeds)")
    print(f"{'Baseline':<22}{'Cost':>24}{'Reduction vs B1':>30}{'Anom.':>8}{'Normal':>8}")
    print("-" * 92)
    for r in summary.itertuples():
        red = ("baseline" if r.baseline == "B1_Reactive_Rule"
               else f"{r.reduction_vs_B1:+6.1f}% [{r.reduction_ci_low:+.1f}, "
                    f"{r.reduction_ci_high:+.1f}]")
        print(f"{r.baseline:<22}{r.cost_mean:8.1f} [{r.cost_ci_low:5.1f}, {r.cost_ci_high:5.1f}]"
              f"{red:>30}{r.cost_anomalous:8.1f}{r.cost_normal:8.1f}")

    # Built-in check: B5 must equal ablation A0 for the same policy
    abl = out / f"ablation_v3{tag}_summary.csv"
    if abl.exists():
        a0 = pd.read_csv(abl).set_index("condition").loc["A0_Full_System", "cost_mean"]
        b5 = summary.set_index("baseline").loc["B5_Full_System", "cost_mean"]
        print(f"\nCheck: B5 {b5:.2f} vs ablation A0 {a0:.2f} "
              f"({'match' if abs(a0 - b5) < 0.01 else 'MISMATCH: re-run the ablation first'})")

    print("\nASSUMPTIONS (report these with the results)")
    print("  same shipments, cases, cost model and paired draws as run_ablation_study.py")
    print(f"  manual handling (B1, B3): escalate_to_human, delay term at its cap "
          f"({HUMAN_LATENCY_S:.0f} s)")
    print("  B2: anomaly type assumed known (optimistic); no optimiser was implemented,")
    print(f"      so B2 uses the measured Resolver latency ({RESOLVER_LATENCY_S} s)")
    print(f"  Monitor alarm policy: {policy_tag(policy)}")
    print(f"\nSaved: outputs/baselines_v3{tag}_cases.csv, _summary.csv")


if __name__ == "__main__":
    main()
