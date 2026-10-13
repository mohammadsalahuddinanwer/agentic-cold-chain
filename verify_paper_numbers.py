"""
verify_paper_numbers.py
=======================
Cross-checks every quantitative claim in Agentic_ColdChain_Springer_LNCS_v5.docx
against the CSV / JSON files your own scripts produced.

"""
import json
import sys
from pathlib import Path

import pandas as pd

OUT = Path("outputs")
TOL = 0.051          # absolute; see chk_rel for large-magnitude values          # tolerance for a figure quoted to one decimal place
results = []

def chk(label, expected, actual, tol=TOL, source=""):
    if actual is None:
        results.append(("SKIP", label, f"expected {expected}", source)); return
    ok = abs(float(actual) - float(expected)) <= tol
    results.append(("PASS" if ok else "FAIL", label,
                    f"paper {expected}  |  file {actual:.4g}", source))

def chk_str(label, expected, actual, source=""):
    if actual is None:
        results.append(("SKIP", label, f"expected {expected}", source)); return
    ok = str(actual).strip() == str(expected).strip()
    results.append(("PASS" if ok else "FAIL", label,
                    f"paper {expected!r}  |  file {actual!r}", source))

MISSING = []

def load(name, **kw):
    """Read an output file, recording its absence so no check disappears silently."""
    p = OUT / name
    if not p.exists():
        MISSING.append(name)
        return None
    return pd.read_csv(p, **kw) if p.suffix == ".csv" else json.loads(p.read_text())

# ══════════════════════════════════════════════════════════════════════
# 1. DIAGNOSIS (paper Sect. 5.2, Table 4)
# ══════════════════════════════════════════════════════════════════════
S = "diagnosis_results_WITH_RAG_ts120.csv / WITHOUT_RAG_ts120.csv"
a = load("diagnosis_results_WITH_RAG_ts120.csv")
b = load("diagnosis_results_WITHOUT_RAG_ts120.csv")
if a is not None and b is not None:
    chk("5.2  accuracy WITH RAG (%)",    79.17, 100 * a.correct.mean(), source=S)
    chk("5.2  accuracy WITHOUT RAG (%)", 75.83, 100 * b.correct.mean(), source=S)
    chk("5.2  test-set size",            120,   len(a), tol=0, source=S)
    chk("5.2  abstentions WITH RAG",      3, (a.predicted == "unknown").sum(), tol=0, source=S)
    chk("5.2  abstentions WITHOUT RAG",   0, (b.predicted == "unknown").sum(), tol=0, source=S)
    VALID = {"compressor_failure","power_outage","door_left_open",
             "driver_error","extreme_weather","gps_blackout"}
    inv_a = (~a.predicted.isin(VALID) & (a.predicted != "unknown")).sum()
    inv_b = (~b.predicted.isin(VALID) & (b.predicted != "unknown")).sum()
    chk("5.2  invented labels WITH RAG",    0, inv_a, tol=0, source=S)
    chk("5.2  invented labels WITHOUT RAG", 0, inv_b, tol=0, source=S)
    if "answer_source" in a.columns:
        chk("5.2  answers from the LLM, WITH RAG", 111,
            (a.answer_source != "keyword_fallback").sum(), tol=0, source=S)
        chk("5.2  answers from the LLM, WITHOUT RAG", 120,
            (b.answer_source != "keyword_fallback").sum(), tol=0, source=S)
    # paired McNemar recomputed from the two files
    A = a.sort_values("case_id").reset_index(drop=True)
    B = b.sort_values("case_id").reset_index(drop=True)
    if (A.text.values == B.text.values).all():
        bb = int(( A.correct & ~B.correct).sum())
        cc = int((~A.correct &  B.correct).sum())
        chk("5.2  McNemar b (RAG only correct)", 14, bb, tol=0, source=S)
        chk("5.2  McNemar c (no-RAG only)",      10, cc, tol=0, source=S)
        chk("5.2  discordant pairs",             24, bb + cc, tol=0, source=S)
        try:
            from scipy.stats import binomtest
            p = binomtest(min(bb, cc), n=bb + cc, p=0.5,
                          alternative="two-sided").pvalue
            chk("5.2  McNemar p", 0.541, p, tol=0.002, source=S)
        except ImportError:
            results.append(("SKIP", "5.2  McNemar p", "scipy unavailable", S))
    else:
        results.append(("FAIL", "5.2  arms line up case-for-case",
                        "the two CSVs are not paired", S))
    # keyword classifier parity (paper Sect. 5.5 / 6.1)
    KW = {"compressor_failure": ["compressor","cooling unit","refrigerant","mechanical","fan","seized"],
          "door_left_open":     ["door","open","ajar","seal","gap"],
          "power_outage":       ["power","electrical","generator","outage","battery","fuse"],
          "driver_error":       ["driver","operator","human","ignored","mistake","negligence"],
          "extreme_weather":    ["weather","storm","heatwave","snow","ambient","external"],
          "gps_blackout":       ["gps","signal","navigation","satellite","tracking"]}
    def kwc(t):
        t = str(t).lower()
        sc = {k: sum(w in t for w in ws) for k, ws in KW.items()}
        return max(sc, key=sc.get) if any(sc.values()) else "unknown"
    rule = a.text.map(kwc) == a.ground_truth
    chk("5.5  keyword classifier accuracy (%)", 79.17, 100 * rule.mean(), source=S)
    bb2 = int(( a.correct & ~rule).sum()); cc2 = int((~a.correct &  rule).sum())
    chk("5.5  RAG vs rule, b", 13, bb2, tol=0, source=S)
    chk("5.5  RAG vs rule, c", 13, cc2, tol=0, source=S)

# per-class F1 quoted in Sect. 5.2 — recomputed from the same two CSVs
if a is not None and b is not None:
    try:
        from sklearn.metrics import f1_score
    except ImportError:
        print("NOTE: scikit-learn not installed; per-class F1 checks skipped")
    else:
        VALID = ["compressor_failure", "door_left_open", "driver_error",
                 "extreme_weather", "gps_blackout", "power_outage"]
        kwf = dict(labels=VALID, average=None, zero_division=0)
        fw = f1_score(a.ground_truth, a.predicted, **kwf)
        fo = f1_score(b.ground_truth, b.predicted, **kwf)
        # paper Sect. 5.2: without -> with retrieval
        F1_EXPECTED = {"compressor_failure": (0.75, 0.80), "door_left_open": (0.88, 0.92),
                       "driver_error": (0.79, 0.86), "extreme_weather": (0.76, 0.67),
                       "gps_blackout": (0.52, 0.71), "power_outage": (0.78, 0.83)}
        for cls, (e_o, e_w) in F1_EXPECTED.items():
            i = VALID.index(cls)
            chk(f"5.2  F1 {cls} without RAG", e_o, fo[i], tol=0.005, source=S)
            chk(f"5.2  F1 {cls} with RAG",    e_w, fw[i], tol=0.005, source=S)
        chk("5.2  classes improved by retrieval", 5,
            int(sum(w > o for w, o in zip(fw, fo))), tol=0, source=S)
        chk("5.2  classes degraded by retrieval", 1,
            int(sum(w < o for w, o in zip(fw, fo))), tol=0, source=S)

# ══════════════════════════════════════════════════════════════════════
# 2. BASELINES (paper Sect. 5.4, Table 5) — validation-selected policy
# ══════════════════════════════════════════════════════════════════════
S = "baselines_v3_classifier_t0.86_k1_summary.csv"
bs = load(S)
if bs is not None:
    bs = bs.set_index("baseline")
    EXP = {"B1_Reactive_Rule": (23.2, None), "B2_Centralised_Opt": (12.6, 45.9),
           "B3_Detection_Only": (30.1, -30.6), "B4_MAS_No_LLM": (13.6, 41.4),
           "B5_Full_System": (14.5, 37.3)}
    for k, (cost, red) in EXP.items():
        if k in bs.index:
            chk(f"5.4  {k} cost", cost, bs.loc[k, "cost_mean"], source=S)
            if red is not None:
                chk(f"5.4  {k} reduction vs B1 (%)", red,
                    bs.loc[k, "reduction_vs_B1"], source=S)
    if {"B3_Detection_Only", "B5_Full_System"} <= set(bs.index):
        b3, b5 = bs.loc["B3_Detection_Only","cost_mean"], bs.loc["B5_Full_System","cost_mean"]
        chk("5.4  B5 vs B3 saving (%)  [paper: 51.8]", 51.8, 100*(b3-b5)/b3, source=S)

# design threshold 0.80 column of Table 5
S80 = "baselines_v3_summary.csv"
bs80 = load(S80)
if bs80 is not None:
    bs80 = bs80.set_index("baseline")
    for k, cost in {"B3_Detection_Only": 37.8, "B4_MAS_No_LLM": 13.6,
                    "B5_Full_System": 17.8}.items():
        if k in bs80.index:
            chk(f"5.4  {k} cost at 0.80", cost, bs80.loc[k, "cost_mean"], source=S80)

# ══════════════════════════════════════════════════════════════════════
# 3. ABLATION (paper Sect. 5.5, Table 6)
# ══════════════════════════════════════════════════════════════════════
S = "ablation_v3_classifier_t0.86_k1_summary.csv"
ab = load(S)
if ab is not None:
    ab = ab.set_index("condition")
    EXP = {"A0_Full_System": (14.5, None), "A1_Remove_RAG": (14.7, 1.2),
           "A3_Remove_Monitor": (13.6, -6.3),
           "A4_Replace_LLM_RuleBased": (14.5, 0.0)}
    for k, (cost, pct) in EXP.items():
        if k in ab.index:
            chk(f"5.5  {k} cost", cost, ab.loc[k, "cost_mean"], source=S)
            if pct is not None:
                chk(f"5.5  {k} % vs A0", pct, ab.loc[k, "pct_vs_A0"], source=S)
S80 = "ablation_v3_summary.csv"
ab80 = load(S80)
if ab80 is not None:
    ab80 = ab80.set_index("condition")
    if "A3_Remove_Monitor" in ab80.index:
        chk("5.5  A3 % vs A0 at the design threshold", -23.4,
            ab80.loc["A3_Remove_Monitor", "pct_vs_A0"], source=S80)

# cross-check the paper relies on: B5 == A0
if bs is not None and ab is not None:
    if "B5_Full_System" in bs.index and "A0_Full_System" in ab.index:
        chk("consistency  B5 == ablation A0",
            bs.loc["B5_Full_System","cost_mean"],
            ab.loc["A0_Full_System","cost_mean"], tol=0.01,
            source="both summary files")

# ══════════════════════════════════════════════════════════════════════
# 4. A2 COST WEIGHTS (paper Sect. 5.6, Table 7)
# ══════════════════════════════════════════════════════════════════════
S = "a2_summary.csv"
a2 = load(S)
if a2 is not None:
    a2 = a2.set_index("condition")
    for k, (gbp, pct, sp) in {"researcher": (1672, None, 0.272),
                              "ga":        (1339, -19.0, 0.089),
                              "random":    (2024, 22.5, 0.350)}.items():
        if k in a2.index:
            chk(f"5.6  {k} GBP/case", gbp, a2.loc[k, "gbp"], tol=2.0, source=S)
            chk(f"5.6  {k} spoiled fraction", sp, a2.loc[k, "spoiled"],
                tol=0.002, source=S)
            if pct is not None:
                chk(f"5.6  {k} % vs researcher", pct,
                    a2.loc[k, "pct_vs_researcher"], tol=0.15, source=S)
    print("NOTE: a2_summary.csv is overwritten by each run. The values above are "
          "the SECONDS reading;\n      if you last ran --action_time_unit minutes, "
          "these will legitimately differ.\n")
S = "a2_price_grid.csv"
pg = load(S)
if pg is not None and {"cargo_gbp", "pct"} <= set(pg.columns):
    for cargo, exp in ((2000, 68.4), (10000, -20.0), (50000, -56.2)):
        sel = pg[pg.cargo_gbp == cargo]
        if len(sel):
            chk(f"5.6  GA vs researcher at GBP {cargo:,} cargo (%)", exp,
                sel.pct.mean(), tol=0.6, source=S)

# ══════════════════════════════════════════════════════════════════════
# 5. SENSITIVITY (paper Sect. 5.7)
# ══════════════════════════════════════════════════════════════════════
for name, fa in (("ablation_v3_classifier_t0.86_k1_cases.csv", 11.1),
                 ("ablation_v3_cases.csv", 39.1)):
    c = load(name)
    if c is not None and {"has_anomaly", "alarmed"} <= set(c.columns):
        a0 = c[c.condition == "A0_Full_System"] if "condition" in c.columns else c
        norm = a0[~a0.has_anomaly]
        if len(norm):
            chk(f"5.7  false-alarm rate on normal segments (%) [{name.split('_v3')[1][:12]}]",
                fa, 100 * norm.alarmed.mean(), tol=0.6, source=name)

# ══════════════════════════════════════════════════════════════════════
# 6. RETRIEVAL AND ALARM POLICY
# ══════════════════════════════════════════════════════════════════════
pol = load("alarm_policy.json")
if pol is not None:
    chk("4.2  selected alarm threshold", 0.86, pol.get("threshold"), tol=0.001,
        source="alarm_policy.json")
    chk_str("4.2  selected trigger", "classifier", pol.get("trigger"),
            source="alarm_policy.json")
    chk("4.2  consecutive confirmations", 1, pol.get("persist"), tol=0,
        source="alarm_policy.json")

S = "alarm_policy_validation.csv"
val = load(S)
if val is not None:
    chk("4.2  candidate policies searched", 39, len(val), tol=0, source=S)
    chk("4.2  distinct triggers", 4, val.trigger.nunique(), tol=0, source=S)
    chk("4.2  distinct thresholds", 9, val.threshold.nunique(), tol=0, source=S)
    chk("4.2  max consecutive confirmations", 3, val.persist.max(), tol=0, source=S)

# ══════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════
print("=" * 96)
print("PAPER NUMBER VERIFICATION  —  Agentic_ColdChain_Springer_LNCS_v5.docx")
print("=" * 96)
w = max(len(l) for _, l, _, _ in results) if results else 40
for status, label, detail, source in results:
    mark = {"PASS": "  ok  ", "FAIL": " FAIL ", "SKIP": " skip "}[status]
    print(f"[{mark}] {label:<{w}}  {detail}")
    if status == "FAIL":
        print(f"{'':>{w+11}}  source: {source}")
if MISSING:
    print("-" * 96)
    print("SOURCE FILES NOT FOUND — the checks that depend on them did NOT run:")
    for m in dict.fromkeys(MISSING):
        print(f"    outputs/{m}")
    print("  Re-run the script that produces each, or confirm those paper numbers by hand.")
n_pass = sum(1 for s, *_ in results if s == "PASS")
n_fail = sum(1 for s, *_ in results if s == "FAIL")
n_skip = sum(1 for s, *_ in results if s == "SKIP")
print("-" * 96)
print(f"{n_pass} passed, {n_fail} FAILED, {n_skip} skipped, "
      f"{len(dict.fromkeys(MISSING))} source file(s) missing")
if n_fail == 0 and not MISSING:
    print("Every paper number that can be checked against a file matches it.")
print("""
CHECK MANUALLY — these are printed by their scripts rather than saved:
  Monitor window-level: precision 0.800, recall 0.802, F1 0.801, RMSE 1.525 degC
      source: val_selected_threshold_eval.py console output
  Monitor event-level at 0.80: 100% of breaching events alarmed, 81.6% before the
      breach, median lead 10 min, 89.3% of all events
      source: run_ablation_study.py console output WITHOUT --policy_file
  Monitor event-level at 0.86: 99.1% breaching, 74.1% before, median 6 min,
      81.4% of all events, 11.1% false alarms on normal segments
      source: run_ablation_study.py console output WITH --policy_file
  Resolver: 50/50 within 120 s, median 57.9 s
      source: evaluate_resolver.py console output / resolver_evaluation_results.csv
  Retrieval similarity 0.618 for the walk-through query (thesis only, not in the paper)
      source: the snippet that printed sim=0.618 cos=0.692
""")
sys.exit(1 if (n_fail or MISSING) else 0)
