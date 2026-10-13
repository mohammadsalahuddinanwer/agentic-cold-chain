"""
run_ablation_study.py  (v3: paired, multi-seed, event-based)
============================================================

Shipments come from ColdChainDataGenerator (generate_cold_chain_training_data.py),
the generator the Monitor was trained on, with NEW seeds (1000+) so no evaluation
shipment was used in training. Each anomaly is a proper event: one type, a start,
a duration and a type-specific temperature rise.

Design
------
1. Monitor input matches training: training-split RobustScaler, threshold 0.80.
2. Normal cases: anomaly-free stretches, so false alarms carry a cost.
3. Lead time is MEASURED as minutes between the first alarm and the moment the
   temperature breaches the product's critical limit. The reactive rule (A3)
   alarms at the breach by definition, so its lead time is ~0.
4. Resolver latency is the MEASURED median (57.9 s). A2 runs only if a latency
   measured with random weights is supplied (--a2_latency).
5. Common random numbers: one uniform draw per case, reused in every condition.
6. Several seeds; mean and 95% CI across seeds.
7. A4: keyword-classifier accuracy measured on the 120-case diagnosis test set.


"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).parent))
from config import Config                                   # noqa: E402
from generate_cold_chain_training_data import (                # noqa: E402
    ColdChainDataGenerator, PRODUCT_SPECS, PRODUCT_WEIGHTS)
from monitor_agent.informer_model import Informer           # noqa: E402


# ════════════════════════════════════════════════════════════════════════
# SETTINGS: every modelling assumption lives here and is reported
# ════════════════════════════════════════════════════════════════════════
BASE_SEED           = 1000     # training data used seed 42; evaluation uses 1000+
MONITOR_THRESHOLD   = 0.80     # operating threshold (validation-selected)
DEFAULT_POLICY      = {"trigger": "classifier", "threshold": MONITOR_THRESHOLD, "persist": 1}
RESOLVER_LATENCY_S  = 57.9     # measured median, evaluate_resolver.py
# Measured on the 120-case balanced test set (diagnosis_test_set_100.json),
# from outputs/diagnosis_results_WITH_RAG_ts120.csv and ..._WITHOUT_RAG_ts120.csv.
# These are the accuracies reported in Sect. 5.2 of the paper.
# NOTE: the earlier values 0.7833 / 0.7000 came from the superseded 60-case run
# whose answers were dominated by a hardcoded keyword fallback (see the
# Implementation provenance paragraph in Sect. 6.2). Do not reinstate them.
DIAG_ACCURACY       = {"rag": 0.7917, "no_rag": 0.7583}

LOOKBACK_MIN        = 30       # scan starts 30 min before the event onset
LOOKAHEAD_MIN       = 120      # and ends at most 120 min after it (max event length),
ALARM_GRACE_MIN     = 30       # or 30 min after the event ends, or at the next event
SCAN_STRIDE_MIN     = 5        # Monitor evaluated every 5 minutes (as in training)
NORMAL_SPAN_MIN     = 120      # length of each normal segment that is scanned
MAX_PER_SHIPMENT    = 3        # cap cases per shipment so no shipment dominates
FALSE_ALARM_ACTION  = "escalate_to_human"   # low-confidence path (Sect. 3.6)
EXCLUDED_TYPES      = {"sensor_malfunction"}  # not in the six-class taxonomy

# Cost model (researcher-designed)
COST_WEIGHTS = np.array([0.40, 0.30, 0.20, 0.10])   # spoilage, rerouting, delay, carbon
ACTION_COSTS = {
    "dispatch_backup": 0.25, "notify_driver": 0.10, "repair_on_site": 0.30,
    "escalate_to_human": 0.70, "no_action": 0.90, "monitor_only": 0.0,
    "reroute_fastest": 0.20, "reroute_safest": 0.22,
}
ACTION_MAP = {
    "compressor_failure": "dispatch_backup", "door_left_open": "notify_driver",
    "power_outage": "dispatch_backup",       "driver_error": "notify_driver",
    "extreme_weather": "notify_driver",      "gps_blackout": "monitor_only",
}
# Normaliser: a missed breaching anomaly = 100. A missed breach is never
# resolved in transit, so its delay term is at the cap (1.0).
MISSED_DELAY = 1.0
B1_RAW = (COST_WEIGHTS[0] * 1.0 + COST_WEIGHTS[1] * ACTION_COSTS["no_action"]
          + COST_WEIGHTS[2] * MISSED_DELAY + COST_WEIGHTS[3] * 0.05)


# ════════════════════════════════════════════════════════════════════════
# MONITOR AGENT
# ════════════════════════════════════════════════════════════════════════
class InformerWithClassifier(nn.Module):
    """Same architecture as train_on_synthetic.py (256-128-64-1 head)."""
    def __init__(self, n_steps, n_features, d_model=256, n_heads=8,
                 n_layers=3, d_ff=512, factor=5, dropout=0.1, device="cpu"):
        super().__init__()
        self.backbone = Informer(n_steps=n_steps, n_features=n_features,
                                 d_model=d_model, n_heads=n_heads,
                                 n_layers=n_layers, d_ff=d_ff,
                                 factor=factor, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),      nn.LayerNorm(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1))
        self.to(device)

    def forward(self, x):
        enc = self.backbone.position_encoding(self.backbone.embedding(x))
        for layer in self.backbone.encoder:
            enc = layer(enc)
        context = enc[:, -1, :]
        pred_30 = self.backbone.prediction_heads["horizon_30"](context).squeeze(-1)
        return pred_30, self.classifier(context).squeeze(-1)


def load_training_scaling():
    """Rebuild the exact scaling used in training (verified by the
    regression check in val_selected_threshold_eval.py)."""
    d = Config.SYNTHETIC_DATA_DIR
    X, y_r = np.load(d / "cold_chain_X.npy"), np.load(d / "cold_chain_y_reg.npy")
    np.random.seed(42)
    idx = np.random.choice(len(X), min(70000, len(X)), replace=False)
    X, y_r = X[idx], y_r[idx]
    train_end = int(len(X) * 0.80)
    scaler = RobustScaler().fit(X.reshape(-1, X.shape[-1])[:train_end * X.shape[1]])
    y_mean, y_std = y_r[:train_end].mean(), y_r[:train_end].std() + 1e-8
    return scaler, float(y_mean), float(y_std), X.shape[-1]


def load_monitor(n_features, device):
    ckpt = Config.CHECKPOINT_DIR / "monitor_synthetic_best.pt"
    state = torch.load(str(ckpt), map_location=device)
    model = InformerWithClassifier(Config.MONITOR_WINDOW_SIZE, n_features, device=device)
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    model.eval()
    return model


def monitor_scores(model, scaler, feats, ends, ws, y_mean, y_std, device):
    """Classifier probability and +30 min forecast (°C) for windows ending at `ends`."""
    wins = np.stack([feats[e - ws:e] for e in ends])
    wins = scaler.transform(wins.reshape(-1, wins.shape[-1])).reshape(wins.shape)
    with torch.no_grad():
        pred, logit = model(torch.tensor(wins, dtype=torch.float32, device=device))
    return (torch.sigmoid(logit).cpu().numpy().reshape(-1),
            pred.cpu().numpy().reshape(-1) * y_std + y_mean)


# ════════════════════════════════════════════════════════════════════════
# CASES: anomaly events and normal segments from the training generator
# ════════════════════════════════════════════════════════════════════════
def shipment_features(s):
    """(n, 5) array in training order: T, RH, vibration, door, CO2."""
    return np.stack([s["temperature"], s["humidity"], s["vibration"],
                     s["door_open"], s["co2"]], axis=1).astype(np.float32)


def events(labels):
    """(start, end) of each contiguous anomalous run; end is exclusive."""
    d = np.diff(np.r_[0, labels.astype(int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def build_cases(seed, n_anom, n_norm, ws):
    """Anomalous cases: events with a clean history of ws + LOOKBACK minutes.
    Normal cases: anomaly-free stretches of ws + NORMAL_SPAN minutes."""
    gen = ColdChainDataGenerator(seed=seed)       # seeds global random / np.random
    rng = np.random.default_rng(seed)             # our own draws
    products = list(PRODUCT_SPECS)
    anom, norm, k = [], [], 0

    while (len(anom) < n_anom or len(norm) < n_norm) and k < 50 * (n_anom + n_norm):
        k += 1
        product = random.choices(products, weights=PRODUCT_WEIGHTS)[0]
        s = gen.generate_shipment_series(f"E{seed}_{k:05d}", product)
        f, lab, n = shipment_features(s), s["anomaly_labels"], s["n_points"]
        crit = PRODUCT_SPECS[product]["critical_breach"]
        ev = events(lab)

        taken = 0
        for i, (a, b) in enumerate(ev):
            if len(anom) >= n_anom or taken >= MAX_PER_SHIPMENT:
                break
            next_onset = ev[i + 1][0] if i + 1 < len(ev) else n
            typ = s["anomaly_types"][a]
            if typ in EXCLUDED_TYPES or a < ws + LOOKBACK_MIN or lab[a - ws - LOOKBACK_MIN:a].any():
                continue
            over = np.flatnonzero(f[a:b, 0] > crit)
            anom.append(dict(case_id=f"{s['shipment_id']}_ev{a}", has_anomaly=True,
                             anomaly_type=typ, product=product, feats=f, onset=a, end=b,
                             scan_end=min(n, next_onset, b + ALARM_GRACE_MIN, a + LOOKAHEAD_MIN),
                             breach=(a + over[0]) if len(over) else None,
                             crit=crit, u=rng.random()))
            taken += 1

        taken, prev_end = 0, 0
        for a, b in ev + [(n, n)]:                     # clean stretches between events
            need = ws + NORMAL_SPAN_MIN
            if a - prev_end >= need and len(norm) < n_norm and taken < MAX_PER_SHIPMENT:
                start = prev_end + int(rng.integers(0, a - prev_end - need + 1))
                norm.append(dict(case_id=f"{s['shipment_id']}_norm{start}", has_anomaly=False,
                                 anomaly_type="none", product=product, feats=f,
                                 seg=(start, start + need), breach=None, crit=crit,
                                 u=rng.random()))
                taken += 1
            prev_end = b
    return anom + norm


def detect(cases, model, scaler, ws, y_mean, y_std, device):
    """Scores every scanned window once (probability and +30 min forecast) and
    finds the first rule alarm (T > critical limit). Monitor alarms are then
    derived from the stored scores by apply_policy()."""
    for c in cases:
        f = c["feats"]
        if c["has_anomaly"]:
            lo = c["onset"] - LOOKBACK_MIN + 1
            hi = c["scan_end"]                 # never runs into the next event
        else:
            lo, hi = c["seg"][0] + ws, c["seg"][1]
        ends = np.arange(lo, hi + 1, SCAN_STRIDE_MIN)
        c["ends"] = ends[(ends >= ws) & (ends <= len(f))]
        c["prob"], c["fc"] = monitor_scores(model, scaler, f, c["ends"], ws,
                                            y_mean, y_std, device)
        over = np.flatnonzero(f[lo - 1:hi, 0] > c["crit"])
        c["rule_alarm"] = int(lo - 1 + over[0]) if len(over) else None
    return cases


def apply_policy(cases, policy):
    """Sets c["mon_alarm"]: the reading index at which the Monitor alarms under
    `policy` = {trigger, threshold, persist}. The alarm fires when the trigger
    has held for `persist` consecutive scans (5 min apart)."""
    trig, thr, k = policy["trigger"], policy["threshold"], policy["persist"]
    for c in cases:
        clf, fc = c["prob"] >= thr, c["fc"] > c["crit"]
        fired = {"classifier": clf, "forecast": fc,
                 "both": clf & fc, "either": clf | fc}[trig]
        run, alarm = 0, None
        for i, x in enumerate(fired):
            run = run + 1 if x else 0
            if run >= k:
                alarm = int(c["ends"][i] - 1)
                break
        c["mon_alarm"] = alarm
    return cases


def policy_tag(policy):
    if policy["trigger"] == "forecast":            # threshold not used
        return f"forecast_k{policy['persist']}"
    return f"{policy['trigger']}_t{policy['threshold']:.2f}_k{policy['persist']}"


# ════════════════════════════════════════════════════════════════════════
# COST AND CONDITIONS
# ════════════════════════════════════════════════════════════════════════
def compute_cost(case, alarm, action, latency_s):
    """Normalised disruption cost (missed breaching anomaly = 100).
    Spoilage applies only to anomalies that breach the critical limit and
    depends on the lead time = breach minute - alarm minute."""
    breaching = case["breach"] is not None
    if alarm is None:
        if not breaching:
            return 0.0, np.nan                                    # nothing happened / harmless
        spoil, action, delay = 1.0, "no_action", MISSED_DELAY     # missed breach, unresolved
        lead = np.nan
    else:
        lead = float(case["breach"] - alarm) if breaching else np.nan
        spoil = float(np.clip(0.80 - (lead / 60.0) * 0.80, 0.0, 1.0)) if breaching else 0.0
        delay = min(1.0, latency_s / 600.0)
    carbon = 0.15 if action == "dispatch_backup" else (0.0 if action == "monitor_only" else 0.05)
    raw = COST_WEIGHTS @ np.array([spoil, ACTION_COSTS[action], delay, carbon])
    return round(100.0 * raw / B1_RAW, 2), lead


def run_condition(cases, detector, accuracy, latency_s):
    rows = []
    for c in cases:
        alarm = c["mon_alarm"] if detector == "monitor" else c["rule_alarm"]
        if alarm is None:
            action = "no_action"
        elif not c["has_anomaly"]:
            action = FALSE_ALARM_ACTION
        elif c["u"] < accuracy:                   # same draw in every condition
            action = ACTION_MAP[c["anomaly_type"]]
        else:
            action = "escalate_to_human"
        cost, lead = compute_cost(c, alarm, action, latency_s)
        rows.append(dict(case_id=c["case_id"], has_anomaly=c["has_anomaly"],
                         anomaly_type=c["anomaly_type"], product=c["product"],
                         breaching=c["breach"] is not None, alarmed=alarm is not None,
                         alarm_after_onset_min=(alarm - c["onset"]) if (alarm is not None and c["has_anomaly"]) else np.nan,
                         lead_before_breach_min=lead, action=action, cost=cost))
    return rows


# ════════════════════════════════════════════════════════════════════════
# A4: measure the keyword classifier on the diagnosis test set
# ════════════════════════════════════════════════════════════════════════
KEYWORDS = {
    "compressor_failure": ["compressor", "cooling unit", "refrigerant", "mechanical", "fan", "seized"],
    "door_left_open":     ["door", "open", "ajar", "seal", "gap"],
    "power_outage":       ["power", "electrical", "generator", "outage", "battery", "fuse"],
    "driver_error":       ["driver", "operator", "human", "ignored", "mistake", "negligence"],
    "extreme_weather":    ["weather", "storm", "heatwave", "snow", "ambient", "external"],
    "gps_blackout":       ["gps", "signal", "navigation", "satellite", "tracking"],
}


def keyword_classify(text):
    text = text.lower()
    scores = {t: sum(k in text for k in kws) for t, kws in KEYWORDS.items()}
    return max(scores, key=scores.get) if any(scores.values()) else "unknown"


def measure_rule_accuracy(path):
    """Accuracy of the keyword classifier on the held-out diagnosis cases.

    Pass data/knowledge_base/diagnosis_test_set_100.json (120 cases), which is
    the set the paper reports. The older diagnosis_test_set.json holds 60.
    """
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = next(v for v in data.values() if isinstance(v, list))
    text_keys  = ["description", "text", "narrative", "case_description", "input"]
    label_keys = ["anomaly_type", "label", "true_label", "ground_truth", "expected_type"]
    tk = next((k for k in text_keys if k in data[0]), None)
    lk = next((k for k in label_keys if k in data[0]), None)
    if tk is None or lk is None:
        raise KeyError(f"Could not find text/label fields; keys are {list(data[0])}")
    norm = lambda s: str(s).strip().lower().replace(" ", "_")
    correct = sum(keyword_classify(str(d[tk])) == norm(d[lk]) for d in data)
    return correct / len(data), len(data), tk, lk


# ════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════
def ci95(values):
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return v.mean(), np.nan, np.nan
    half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), v.mean() - half, v.mean() + half


def print_diagnostics(df):
    """Detection behaviour of the Monitor (A0) and the reactive rule (A3)."""
    print("\nDETECTION DIAGNOSTICS (pooled over seeds)")
    for name, label in [("A0_Full_System", "Monitor"), ("A3_Remove_Monitor", "Rule")]:
        d = df[df.condition == name]
        a, n, br = d[d.has_anomaly], d[~d.has_anomaly], d[d.breaching]
        lead = br.loc[br.alarmed, "lead_before_breach_min"]
        print(f"  {label:8s} events alarmed {a.alarmed.mean():6.1%} | breaching events alarmed "
              f"{br.alarmed.mean():6.1%} | false alarms on normal {n.alarmed.mean():6.1%}")
        print(f"  {'':8s} lead before breach: median {lead.median():5.1f} min "
              f"(IQR {lead.quantile(.25):.1f} to {lead.quantile(.75):.1f}); "
              f"alarmed before breach {(lead > 0).mean():6.1%}")
    d = df[(df.condition == "A0_Full_System") & df.has_anomaly]
    print("\n  Monitor by anomaly type:")
    print(f"    {'type':<20}{'events':>7}{'alarmed':>9}{'min after onset':>17}"
          f"{'breach %':>10}{'lead before breach':>20}")
    for t, g in d.groupby("anomaly_type"):
        gb = g[g.breaching & g.alarmed]
        print(f"    {t:<20}{len(g):>7}{g.alarmed.mean():>9.0%}"
              f"{g.alarm_after_onset_min.median():>17.1f}{g.breaching.mean():>10.0%}"
              f"{gb.lead_before_breach_min.median():>20.1f}")


def main():
    ap = argparse.ArgumentParser(description="Paired multi-seed ablation (v3, event-based)")
    ap.add_argument("--seeds", type=int, default=10, help="number of seeds (1000, 1001, ...)")
    ap.add_argument("--n_anomalous", type=int, default=100)
    ap.add_argument("--n_normal", type=int, default=100)
    ap.add_argument("--diag_test_set", type=str, default=None,
                    help="JSON of the held-out diagnosis cases, enables A4; "
                         "use diagnosis_test_set_100.json (120 cases)")
    ap.add_argument("--a2_latency", type=float, default=None,
                    help="resolver latency (s) MEASURED with random weights (enables A2)")
    ap.add_argument("--policy_file", type=str, default=None,
                    help="JSON alarm policy chosen by select_alarm_policy.py "
                         "(default: classifier, threshold 0.80, 1 scan, as in Sect. 3.6)")
    args = ap.parse_args()

    Config.create_dirs()
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ws = Config.MONITOR_WINDOW_SIZE

    print("=" * 70)
    print("ABLATION STUDY v3: paired, multi-seed, event-based")
    print("=" * 70)
    scaler, y_mean, y_std, n_feat = load_training_scaling()
    model = load_monitor(n_feat, device)
    policy = (json.loads(Path(args.policy_file).read_text()) if args.policy_file
              else DEFAULT_POLICY)
    tag = "" if args.policy_file is None else "_" + policy_tag(policy)
    print(f"Monitor: training scaler, alarm policy {policy_tag(policy)}, device={device}")

    conditions = {
        "A0_Full_System":    dict(detector="monitor", accuracy=DIAG_ACCURACY["rag"],
                                  latency_s=RESOLVER_LATENCY_S),
        "A1_Remove_RAG":     dict(detector="monitor", accuracy=DIAG_ACCURACY["no_rag"],
                                  latency_s=RESOLVER_LATENCY_S),
        "A3_Remove_Monitor": dict(detector="rule", accuracy=DIAG_ACCURACY["rag"],
                                  latency_s=RESOLVER_LATENCY_S),
    }
    if args.a2_latency is not None:
        conditions["A2_Remove_EvoOpt"] = dict(detector="monitor",
                                              accuracy=DIAG_ACCURACY["rag"],
                                              latency_s=args.a2_latency)
    else:
        print("A2 skipped: no latency measured with random weights (--a2_latency).")
    if args.diag_test_set:
        acc, n, tk, lk = measure_rule_accuracy(args.diag_test_set)
        print(f"A4 keyword classifier accuracy: {acc:.4f} on {n} cases "
              f"(fields '{tk}' / '{lk}')")
        conditions["A4_Replace_LLM_RuleBased"] = dict(detector="monitor", accuracy=acc,
                                                      latency_s=RESOLVER_LATENCY_S)
    else:
        print("A4 skipped: pass --diag_test_set to measure the keyword classifier.")
    order = [c for c in ["A0_Full_System", "A1_Remove_RAG", "A2_Remove_EvoOpt",
                         "A3_Remove_Monitor", "A4_Replace_LLM_RuleBased"] if c in conditions]

    all_rows = []
    for seed in range(BASE_SEED, BASE_SEED + args.seeds):
        cases = build_cases(seed, args.n_anomalous, args.n_normal, ws)
        cases = detect(cases, model, scaler, ws, y_mean, y_std, device)
        cases = apply_policy(cases, policy)
        n_a = sum(c["has_anomaly"] for c in cases)
        n_b = sum(c["breach"] is not None for c in cases)
        print(f"seed {seed}: {n_a} anomaly events ({n_b} breaching) + "
              f"{len(cases) - n_a} normal segments")
        for name in order:
            for r in run_condition(cases, **conditions[name]):
                all_rows.append(dict(seed=seed, condition=name, **r))

    df = pd.DataFrame(all_rows)
    df.to_csv(out / f"ablation_v3{tag}_cases.csv", index=False)
    print_diagnostics(df)

    # ── Per-seed means and paired differences vs A0 ──────────────────────
    g = df.groupby(["condition", "seed"])
    per_seed = pd.DataFrame({
        "cost":           g.cost.mean(),
        "cost_anomalous": df[df.has_anomaly].groupby(["condition", "seed"]).cost.mean(),
        "cost_normal":    df[~df.has_anomaly].groupby(["condition", "seed"]).cost.mean(),
    }).reset_index()
    base = per_seed[per_seed.condition == "A0_Full_System"].set_index("seed")["cost"]
    per_seed["pct_vs_A0"] = [100.0 * (c - base[s]) / base[s]
                             for c, s in zip(per_seed.cost, per_seed.seed)]
    per_seed.to_csv(out / f"ablation_v3{tag}_seeds.csv", index=False)

    summary = []
    for name in order:
        s = per_seed[per_seed.condition == name]
        m, lo, hi = ci95(s.cost)
        pm, plo, phi = ci95(s.pct_vs_A0)
        summary.append(dict(condition=name, cost_mean=m, cost_ci_low=lo, cost_ci_high=hi,
                            pct_vs_A0=pm, pct_ci_low=plo, pct_ci_high=phi,
                            cost_anomalous=s.cost_anomalous.mean(),
                            cost_normal=s.cost_normal.mean(), n_seeds=len(s)))
    summary = pd.DataFrame(summary)
    summary.to_csv(out / f"ablation_v3{tag}_summary.csv", index=False)

    print(f"\nABLATION RESULTS ({args.seeds} seeds; mean [95% CI] across seeds)")
    print(f"{'Condition':<26}{'Cost':>24}{'% vs A0':>28}{'Anom.':>8}{'Normal':>8}")
    print("-" * 94)
    for r in summary.itertuples():
        pct = ("baseline" if r.condition == "A0_Full_System"
               else f"{r.pct_vs_A0:+6.1f}% [{r.pct_ci_low:+.1f}, {r.pct_ci_high:+.1f}]")
        print(f"{r.condition:<26}{r.cost_mean:8.1f} [{r.cost_ci_low:5.1f}, {r.cost_ci_high:5.1f}]"
              f"{pct:>28}{r.cost_anomalous:8.1f}{r.cost_normal:8.1f}")

    print("\nASSUMPTIONS (report these with the results)")
    print(f"  shipments: ColdChainDataGenerator, seeds {BASE_SEED}-{BASE_SEED + args.seeds - 1} "
          f"(training used seed 42); sensor_malfunction events excluded")
    print(f"  cost weights {COST_WEIGHTS.tolist()}; action costs as in ACTION_COSTS")
    print("  spoilage only if the event breaches the product's critical limit:")
    print("    0.80 if alarmed at the breach, 0 if >= 60 min before, linear; 1.0 if missed")
    print("  a missed breach is unresolved: delay term at its cap; missed breach = 100")
    print("  diagnosis correctness sampled from measured accuracy (paired draws)")
    print(f"  resolver latency {RESOLVER_LATENCY_S} s (measured median)")
    print(f"  false alarm -> '{FALSE_ALARM_ACTION}'; gps_blackout -> 'monitor_only' (no cost)")
    print(f"  normal segments: {NORMAL_SPAN_MIN} min scanned every {SCAN_STRIDE_MIN} min; "
          f"events scanned from {LOOKBACK_MIN} min before onset to {ALARM_GRACE_MIN} min after the event ends"
          f" (never into the next event)")
    print(f"  Monitor alarm policy: {policy_tag(policy)}")
    print(f"\nSaved: outputs/ablation_v3{tag}_cases.csv, _seeds.csv, _summary.csv")


if __name__ == "__main__":
    main()
