"""
audit_diagnosis_fallback.py
===========================
Counts how many diagnoses in a saved evaluation came from the LLM and how many
came from the hardcoded keyword fallback in
ReActAgent._fallback_diagnosis_from_logs.
    

No GPU and no LLM: it reads the CSVs the evaluation already wrote.

"""
import argparse
import collections
from pathlib import Path

import pandas as pd

FALLBACK_CONFIDENCE = 0.7


def keyword_rule(text: str) -> str:
    """Reproduces ReActAgent._fallback_diagnosis_from_logs, in its if/elif order."""
    t = str(text).lower()
    if "compressor" in t or "cooling" in t or "refrigerat" in t or "cool" in t:
        return "compressor_failure"
    if "door" in t or "seal" in t or "ajar" in t:
        return "door_left_open"
    if "gps" in t or "signal" in t or "satellite" in t:
        return "gps_blackout"
    if "driver" in t or "operator" in t:
        return "driver_error"
    if ("weather" in t or "snow" in t or "rain" in t or "storm" in t
            or "heat" in t):
        return "extreme_weather"
    if "power" in t or "electrical" in t or "outage" in t:
        return "power_outage"
    if "sensor" in t:
        return "sensor_malfunction"
    return "unknown"


def audit(path: Path, label: str):
    if not path.exists():
        print(f"  {path} not found — skipped")
        return None
    d = pd.read_csv(path)
    d["keyword_rule"] = d.text.map(keyword_rule)

    if "answer_source" in d.columns and d.answer_source.notna().any():
        is_fb = d.answer_source.astype(str) == "keyword_fallback"
        basis = "recorded answer_source column"
    else:
        is_fb = ((d.confidence.round(4) == FALLBACK_CONFIDENCE)
                 & (d.predicted == d.keyword_rule))
        basis = "inferred (confidence 0.70 AND prediction == keyword rule)"
    llm = ~is_fb

    print(f"=== {label}    n = {len(d)}    reported accuracy = {d.correct.mean():.4f}")
    print(f"    provenance basis: {basis}")
    print(f"    from the LLM              : {llm.sum():>3}/{len(d)} = {llm.mean():>6.1%}"
          + (f"   accuracy {d[llm].correct.mean():.4f}" if llm.sum() else ""))
    if "answer_source" in d.columns:
        trunc = d.answer_source.astype(str) == "llm_truncated"
        if int(trunc.sum()):
            print(f"      of which recovered from a truncated completion: "
                  f"{int(trunc.sum())}   accuracy {d[trunc].correct.mean():.4f}")
            print("      (still the model's own answer: its JSON was cut off by the")
            print("       token limit and the fields were read back from its text)")
    print(f"    from the keyword fallback : {is_fb.sum():>3}/{len(d)} = {is_fb.mean():>6.1%}"
          + (f"   accuracy {d[is_fb].correct.mean():.4f}" if is_fb.sum() else ""))
    print(f"    keyword rule alone, same cases : "
          f"{(d.keyword_rule == d.ground_truth).mean():.4f}")
    if "iterations" in d.columns and d.iterations.notna().any():
        print(f"    iterations: mean {d.iterations.mean():.2f}, "
              f"max {int(d.iterations.max())}, "
              f"hit the limit {int((d.iterations == d.iterations.max()).sum())} times")
    if is_fb.mean() > 0.25:
        print("    ⚠️  over a quarter of answers did not come from the LLM — the")
        print("        accuracy figure is substantially the keyword rule's, and the")
        print("        RAG comparison is confounded (the fallback ignores retrieval).")
    print()
    d["source"] = ["keyword_fallback" if f else "llm" for f in is_fb]
    d["mode_label"] = label
    return d


def mcnemar_from_frames(frames, tag):
    """Recompute the exact McNemar test from the two saved frames."""
    if len(frames) != 2:
        return
    try:
        from scipy.stats import binomtest
    except ImportError:
        print("scipy not available — skipping the significance recomputation.")
        return
    a, b = (f.sort_values("case_id").reset_index(drop=True) for f in frames)
    if len(a) != len(b) or not (a.text.values == b.text.values).all():
        print("The two files do not line up case-for-case; skipping McNemar.")
        return
    bb = int((a.correct & ~b.correct).sum())
    cc = int((~a.correct & b.correct).sum())
    n = bb + cc
    print("=" * 70)
    print("EXACT McNEMAR, recomputed from the saved files")
    print("=" * 70)
    print(f"  WITH RAG only correct (b)    : {bb}")
    print(f"  WITHOUT RAG only correct (c) : {cc}")
    print(f"  discordant pairs             : {n}")
    if n == 0:
        print("  the conditions never disagreed — no test possible.")
        return
    p = binomtest(min(bb, cc), n=n, p=0.5, alternative="two-sided").pvalue
    floor = 2 * 0.5 ** n
    print(f"  two-sided exact p            : {p:.4f}")
    print(f"  smallest p attainable at n={n} : {floor:.3g}")
    if p < 0.05:
        print("  → statistically significant (p < 0.05)")
    elif floor >= 0.05:
        print(f"  → NOT significant and CANNOT be: with {n} discordant pairs the")
        print(f"    floor is {floor:.4f}. At least 6 are needed to reach p < 0.05.")
        print("    Report this as a power limitation, not as evidence of no effect.")
    else:
        print("  → not significant; the discordant pairs are too evenly split.")
        print(f"    This is NOT a power limitation: with {n} discordant pairs the")
        print(f"    attainable floor is {floor:.3g}, so a real effect of the")
        print("    hypothesised size would have been detected. Report it as a")
        print("    properly powered null result.")


def baseline_comparisons(frames):
    """Each arm against the keyword rule on the same cases.

    This is the comparison that carries the positive result: it asks whether the
    LLM agent beats rule-based classification, rather than whether retrieval beats
    no retrieval.
    """
    try:
        from scipy.stats import binomtest
    except ImportError:
        return
    print("=" * 70)
    print("EACH ARM vs THE KEYWORD RULE (paired, same cases)")
    print("=" * 70)
    for d in frames:
        label = d.mode_label.iloc[0]
        rule = d.text.map(keyword_rule) == d.ground_truth
        bb = int((d.correct & ~rule).sum())
        cc = int((~d.correct & rule).sum())
        n = bb + cc
        if n == 0:
            print(f"  {label:<12} identical to the rule on every case")
            continue
        p = binomtest(min(bb, cc), n=n, p=0.5, alternative="two-sided").pvalue
        print(f"  {label:<12} {d.correct.mean():.4f} vs rule {rule.mean():.4f}"
              f"  ({100*(d.correct.mean()-rule.mean()):+.1f} pp)"
              f"   b={bb} c={cc} n={n}  p={p:.4f}"
              f"{'  SIGNIFICANT' if p < 0.05 else ''}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="",
                    help="filename suffix used on the run, e.g. _kb128_ts120")
    ap.add_argument("--dir", default="outputs")
    args = ap.parse_args()

    out = Path(args.dir)
    files = [(out / f"diagnosis_results_WITH_RAG{args.tag}.csv", "WITH RAG"),
             (out / f"diagnosis_results_WITHOUT_RAG{args.tag}.csv", "WITHOUT RAG")]

    print("=" * 70)
    print("DIAGNOSIS PROVENANCE AUDIT — how many answers came from the LLM?")
    print("=" * 70)
    frames = [f for f in (audit(p, l) for p, l in files) if f is not None]
    if not frames:
        print("No evaluation CSVs found. Run evaluate_diagnosis_accuracy.py first,")
        print(f"or pass --tag if the run used one (looked in {out}/).")
        return

    mcnemar_from_frames(frames, args.tag)
    baseline_comparisons(frames)

    combined = pd.concat(frames, ignore_index=True)
    dest = out / f"diagnosis_provenance_audit{args.tag}.csv"
    combined.to_csv(dest, index=False)
    print(f"\nPer-case provenance written to {dest}")


if __name__ == "__main__":
    main()
