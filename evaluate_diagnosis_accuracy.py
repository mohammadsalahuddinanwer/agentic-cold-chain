"""Evaluate Diagnosis Agent accuracy on a labelled test set.

Runs TWO evaluations back-to-back:
  1. WITH RAG  (standard mode — uses FAISS knowledge base)
  2. WITHOUT RAG (LLM-only mode — RAG retriever disabled)

This produces a direct comparison for RQ2:
  "Does RAG-enhanced LLM diagnosis improve anomaly classification accuracy
   compared to standalone LLMs?"


"""
import sys
import json
import argparse
import time
from pathlib import Path
from difflib import SequenceMatcher

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from core.llm_wrapper import get_llm
from diagnosis_agent.rag_retriever import RAGRetriever
from diagnosis_agent.re_act_agent import ReActAgent
from diagnosis_agent.diagnosis_agent import DiagnosisAgent
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np


# ================================================================
# SYNONYM MAPPING FOR FUZZY MATCHING  (unchanged)
# ================================================================
SYNONYM_MAP = {
    "door_left_open": [
        "door_left_ajar", "door_open", "door_ajar", "door_not_closed",
        "door_seal_failure", "door_seal_degradation", "loading_issue",
        "unloading_delay", "door_failure"
    ],
    "compressor_failure": [
        "cooling_unit_failure", "cooling_system_failure", "refrigeration_failure",
        "refrigerant_leak", "compressor_issue", "compressor_overheating",
        "condenser_blockage", "ice_build_up", "cooling_failure"
    ],
    "power_outage": [
        "power_failure", "electrical_failure", "generator_failure",
        "power_cut", "power_surge", "battery_failure", "power_fluctuation"
    ],
    "driver_error": [
        "human_error", "operator_error", "driver_negligence",
        "driver_misread_thermostat", "driver_communication_failure",
        "temperature_setting_issue"
    ],
    "extreme_weather": [
        "weather_issue", "environmental_delay", "heatwave", "storm",
        "heavy_rainfall", "extreme_cold", "temperature_fluctuation",
        "temperature_exposure"
    ],
    "gps_blackout": [
        "communication_failure", "gps_signal_loss", "satellite_signal_loss",
        "signal_loss", "gps_issue", "navigation_failure"
    ],
    "sensor_malfunction": [
        "sensor_failure", "sensor_issue", "faulty_sensor", "erratic_sensor"
    ],
    "vibration_anomaly": ["vibration_issue", "rough_terrain", "cargo_shift"],
    "route_deviation":   ["off_route", "route_change", "detour"],
    "fuel_leak":         ["fuel_issue", "fuel_consumption", "fuel_problem"],
    "traffic_congestion":["traffic_delay", "congestion", "road_delay"],
    "battery_failure":   ["battery_issue", "battery_depleted", "low_battery"],
}


def is_correct_prediction(predicted: str, ground_truth: str,
                           similarity_threshold: float = 0.6) -> bool:
    """Check if predicted label matches ground truth using fuzzy matching."""
    if predicted.lower() == ground_truth.lower():
        return True
    if ground_truth in SYNONYM_MAP:
        for synonym in SYNONYM_MAP[ground_truth]:
            if predicted.lower() == synonym.lower():
                return True
            if synonym.lower() in predicted.lower() or predicted.lower() in synonym.lower():
                return True
    if ground_truth.lower() in predicted.lower():
        return True
    if predicted.lower() in ground_truth.lower():
        return True
    similarity = SequenceMatcher(None, predicted.lower(), ground_truth.lower()).ratio()
    if similarity >= similarity_threshold:
        return True
    return False


def load_test_set(test_set_path: str = "data/knowledge_base/diagnosis_test_set.json"):
    with open(test_set_path, 'r') as f:
        test_cases = json.load(f)
    return test_cases


# ================================================================
# CORE EVALUATION FUNCTION  ← CHANGED: accepts agent directly
# ================================================================
def run_evaluation(diagnosis_agent, test_cases, label: str) -> pd.DataFrame:
    """
    Run the given diagnosis_agent on all test_cases and return results DataFrame.

    Args:
        diagnosis_agent : DiagnosisAgent instance (with or without RAG)
        test_cases      : list of test case dicts
        label           : human-readable label for printing ("WITH RAG" / "WITHOUT RAG")
    """
    results = []
    print(f"\n{'='*60}")
    print(f"🔍 Evaluating — {label}  ({len(test_cases)} cases)")
    print(f"{'='*60}")

    for i, case in enumerate(test_cases):
        print(f"\n[{i+1}/{len(test_cases)}] {case['text'][:60]}...")

        context = {
            "monitor_prediction": {
                "30": {"temperature": 3.5, "is_anomaly": True, "confidence": 0.9}
            },
            "sensor_data": [],
            "operational_logs": [case["text"]]
        }

        t_case = time.time()
        try:
            diagnosis_result = diagnosis_agent.diagnose(context)
            predicted_type = diagnosis_result.get("diagnosis", {}).get("anomaly_type", "unknown")
            confidence     = diagnosis_result.get("confidence", 0.0)
            # Provenance: did the LLM answer, or did the keyword fallback?
            answer_source  = diagnosis_result.get("answer_source", "unknown")
            used_fallback  = diagnosis_result.get("used_fallback", None)
            n_iterations   = diagnosis_result.get("iterations", None)
        except Exception as e:
            print(f"❌ Error: {e}")
            predicted_type = "error"
            confidence     = 0.0
            answer_source  = "error"
            used_fallback  = None
            n_iterations   = None

        elapsed = time.time() - t_case
        ground_truth = case["ground_truth"]
        is_correct   = is_correct_prediction(predicted_type, ground_truth)

        results.append({
            "case_id":       i + 1,
            "text":          case["text"],
            "ground_truth":  ground_truth,
            "predicted":     predicted_type,
            "confidence":    confidence,
            "correct":       is_correct,
            "mode":          label,
            "answer_source": answer_source,
            "used_fallback": used_fallback,
            "iterations":    n_iterations,
            "seconds":       round(elapsed, 1),
        })

        status = "✅" if is_correct else "❌"
        src = "LLM" if answer_source == "llm" else answer_source
        print(f"   GT: {ground_truth:25s} → Pred: {predicted_type:25s} "
              f"(conf: {confidence:.2f}, via {src}, {elapsed:.0f}s) {status}")

    df = pd.DataFrame(results)

    # Timing: lets a 5-case smoke run predict the cost of the full experiment,
    # instead of having to guess it.
    if "seconds" in df and len(df):
        per = df.seconds.mean()
        print(f"\n   Timing — {label}: {per:.0f}s per case "
              f"(min {df.seconds.min():.0f}, max {df.seconds.max():.0f}), "
              f"{df.seconds.sum()/60:.1f} min for {len(df)} cases")
        for n, name in ((60, "60-case set"), (120, "120-case set")):
            print(f"     projected: {n} cases x 2 arms = {2*n*per/3600:.1f} h "
                  f"({name})")
        if "iterations" in df and df.iterations.notna().any():
            print(f"     mean LLM calls per case: {df.iterations.mean():.1f} "
                  f"(each call is one loop iteration)")

    # Provenance summary: this is the number that was invisible before, and the
    # reason the original 78.33% was largely a keyword rule rather than the model.
    if "used_fallback" in df and df.used_fallback.notna().any():
        fb = df.used_fallback.fillna(False).astype(bool)
        print(f"\n   Answer provenance — {label}")
        print(f"     LLM              : {(~fb).sum()}/{len(df)}"
              f"   accuracy {df[~fb].correct.mean():.4f}" if (~fb).sum() else
              f"     LLM              : 0/{len(df)}")
        print(f"     keyword fallback : {fb.sum()}/{len(df)}"
              f"   accuracy {df[fb].correct.mean():.4f}" if fb.sum() else
              f"     keyword fallback : 0/{len(df)}")
        if fb.mean() > 0.25:
            print("     ⚠️  more than a quarter of answers did not come from the LLM;"
                  " report this rate with the accuracy figure.")

    return df


# ================================================================
# PAIRED SIGNIFICANCE TEST (exact McNemar)
# ================================================================
def mcnemar_exact(df_a, df_b, label_a="WITH RAG", label_b="WITHOUT RAG"):
    """Exact (binomial) McNemar test on the paired per-case outcomes.

    McNemar looks ONLY at the cases where the two conditions disagree:
      b = A correct, B wrong      c = A wrong, B correct
    Under the null hypothesis the two are equally likely, so b ~ Binomial(b+c, 0.5)
    and the two-sided exact p-value is 2 * P(X <= min(b, c)).

    Why the exact test and not the chi-square version: with only a handful of
    discordant pairs the chi-square approximation is unreliable, and the exact
    test has a hard floor that is worth knowing about -- with n = b + c discordant
    pairs the smallest attainable two-sided p-value is 2 * 0.5^n. That means:

        n = 4  ->  p >= 0.125     cannot reach 0.05 however lopsided
        n = 5  ->  p >= 0.0625    cannot reach 0.05
        n = 6  ->  p >= 0.03125   can reach significance, but only at 6-0
        n = 8  ->  p >= 0.0078    8-0 or 7-1 significant
        n = 10 ->  p >= 0.00195   10-0, 9-1 significant; 8-2 is p = 0.109

    So a non-significant result on few discordant pairs is a statement about the
    sample size, not evidence that the two conditions are equivalent.
    """
    from scipy.stats import binomtest

    a = df_a.sort_values("case_id").reset_index(drop=True)
    b_ = df_b.sort_values("case_id").reset_index(drop=True)
    if len(a) != len(b_) or not (a.text.values == b_.text.values).all():
        print("⚠️  the two conditions do not line up case-for-case; "
              "McNemar needs paired outcomes on identical cases.")
        return None

    both  = int(( a.correct &  b_.correct).sum())
    b_cnt = int(( a.correct & ~b_.correct).sum())     # A right, B wrong
    c_cnt = int((~a.correct &  b_.correct).sum())     # A wrong, B right
    neither = int((~a.correct & ~b_.correct).sum())
    n_disc = b_cnt + c_cnt

    print(f"\n{'='*60}")
    print("PAIRED SIGNIFICANCE TEST — exact McNemar")
    print(f"{'='*60}")
    print(f"  cases                              : {len(a)}")
    print(f"  both correct                       : {both}")
    print(f"  {label_a} only correct   (b)       : {b_cnt}")
    print(f"  {label_b} only correct   (c)       : {c_cnt}")
    print(f"  both wrong                         : {neither}")
    print(f"  discordant pairs (b + c)           : {n_disc}")

    if n_disc == 0:
        print("  the two conditions never disagreed; no test is possible.")
        return None

    p = binomtest(min(b_cnt, c_cnt), n=n_disc, p=0.5, alternative="two-sided").pvalue
    floor = 2 * 0.5 ** n_disc
    print(f"  two-sided exact p                  : {p:.4f}")
    print(f"  smallest p attainable with {n_disc} pairs : {floor:.4f}")
    if p < 0.05:
        print("  → difference is statistically significant (p < 0.05)")
    elif floor >= 0.05:
        print(f"  → NOT significant, and it CANNOT be: with only {n_disc} discordant")
        print(f"    pairs the floor is {floor:.4f}. At least 6 discordant pairs are")
        print("    needed before p < 0.05 is reachable at all. Report this as a")
        print("    power limitation, not as evidence of no effect.")
    else:
        print("  → not significant at 0.05; the discordant pairs are too evenly split.")
    return {"n": len(a), "both": both, "b": b_cnt, "c": c_cnt, "neither": neither,
            "n_discordant": n_disc, "p_value": p, "p_floor": floor}


# ================================================================
# PRINT SUMMARY FOR ONE MODE
# ================================================================
def print_summary(df: pd.DataFrame, label: str) -> float:
    """Print accuracy summary and return fuzzy accuracy."""
    accuracy       = df["correct"].mean()
    exact_accuracy = (df["predicted"] == df["ground_truth"]).mean()

    print(f"\n{'='*60}")
    print(f"📊 RESULTS — {label}")
    print(f"{'='*60}")
    print(f"   Total cases         : {len(df)}")
    print(f"   Correct (fuzzy)     : {df['correct'].sum()}")
    print(f"   Fuzzy accuracy      : {accuracy:.2%}")
    print(f"   Exact match accuracy: {exact_accuracy:.2%}")

    print("\n📋 Classification Report:")
    print(classification_report(df["ground_truth"], df["predicted"],
                                zero_division=0))

    print("📊 Confusion Matrix:")
    labels = sorted(df["ground_truth"].unique())
    cm     = confusion_matrix(df["ground_truth"], df["predicted"], labels=labels)
    cm_df  = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)

    return accuracy


# ================================================================
# MAIN EVALUATION  ← KEY CHANGE: two passes, one with RAG, one without
# ================================================================
def evaluate_diagnosis_agent(test_cases, max_cases: int = None,
                              run_rag: bool = True, run_no_rag: bool = True,
                              kb_files=None, tag: str = ""):
    """
    Run evaluation with RAG and/or without RAG.

    The key change vs the original file is that we run TWO evaluations:
      Pass 1  —  DiagnosisAgent(llm, rag=RAGRetriever)   → WITH RAG
      Pass 2  —  DiagnosisAgent(llm, rag=None)            → WITHOUT RAG

    Pass 2 is where the single line change happens:
        diagnosis_agent_no_rag = DiagnosisAgent(llm, rag_retriever=None)
    This disables RAG retrieval so the LLM must classify from the prompt alone.
    """
    print("Initialising Diagnosis Agent...")
    Config.create_dirs()

    print("Loading Llama 3.1...")
    llm = get_llm(**Config.get_llama_kwargs())

    # ── Build RAG index ──────────────────────────────────────────
    rag = RAGRetriever()
    # The knowledge base is chosen by --kb_file so the corpus is explicit and
    # recorded, rather than whatever a glob happens to pick up. Default: the 60
    # curated cases. The test set must never appear here.
    if kb_files is None:
        kb_files = [Config.KNOWLEDGE_BASE_DIR / "anomaly_cases.json"]
    knowledge_files = [Path(f) for f in kb_files]
    missing = [f for f in knowledge_files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"knowledge base file(s) not found: {missing}")

    # Guard: refuse to index anything that carries ground-truth labels.
    for f in knowledge_files:
        entries = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(entries, list) and any("ground_truth" in e for e in entries
                                             if isinstance(e, dict)):
            raise ValueError(
                f"{f.name} contains a 'ground_truth' field — that is a labelled "
                "test set and must not be indexed. Refusing to continue.")

    rag.build_index(knowledge_files)
    n_docs = len(rag.documents)
    print(f"RAG index: {n_docs} documents from "
          f"{[f.name for f in knowledge_files]}")

    # Guard: the test cases must not appear in the index.
    kb_texts = {d["text"].strip() for d in rag.documents}
    overlap = kb_texts & {c["text"].strip() for c in test_cases}
    if overlap:
        raise ValueError(f"{len(overlap)} test case text(s) also appear in the "
                         f"knowledge base — leakage. Refusing to continue.")
    print(f"Leakage check: 0 of {len(test_cases)} test texts appear in the index ✅")

    # Limit test cases if requested
    if max_cases:
        test_cases = test_cases[:max_cases]

    results_rag    = None
    results_no_rag = None
    acc_rag        = None
    acc_no_rag     = None

    # ── Pass 1: WITH RAG ─────────────────────────────────────────
    if run_rag:
        agent_with_rag = DiagnosisAgent(llm, rag_retriever=rag)   # standard
        results_rag    = run_evaluation(agent_with_rag, test_cases, "WITH RAG")
        acc_rag        = print_summary(results_rag, "WITH RAG")
        p_rag = f"outputs/diagnosis_results_WITH_RAG{tag}.csv"
        results_rag.to_csv(p_rag, index=False)
        print(f"💾 Saved to {p_rag}")

    # ── Pass 2: WITHOUT RAG ──────────────────────────────────────
    # ┌─────────────────────────────────────────────────────────────┐
    # │  THE ONLY LINE THAT CHANGES vs the original file:          │
    # │  rag_retriever=None  disables retrieval completely.         │
    # │  The LLM still runs its ReAct loop but the                 │
    # │  retrieve_similar_cases tool returns [] every time,         │
    # │  so the LLM must classify using only the prompt context.   │
    # └─────────────────────────────────────────────────────────────┘
    if run_no_rag:
        agent_no_rag = DiagnosisAgent(llm, rag_retriever=None)    # ← KEY CHANGE
        results_no_rag = run_evaluation(agent_no_rag, test_cases, "WITHOUT RAG")
        acc_no_rag     = print_summary(results_no_rag, "WITHOUT RAG")
        p_norag = f"outputs/diagnosis_results_WITHOUT_RAG{tag}.csv"
        results_no_rag.to_csv(p_norag, index=False)
        print(f"💾 Saved to {p_norag}")

    # ── Comparison summary ───────────────────────────────────────
    if run_rag and run_no_rag:
        improvement = acc_rag - acc_no_rag
        print(f"\n{'='*60}")
        print(f"📝 COMPARISON SUMMARY — FOR THESIS RQ2")
        print(f"{'='*60}")
        print(f"   WITH RAG    accuracy : {acc_rag:.2%}")
        print(f"   WITHOUT RAG accuracy : {acc_no_rag:.2%}")
        print(f"   Improvement from RAG : {improvement:+.2%} percentage points")
        if improvement > 0:
            print(f"   ✅ RAG improves accuracy by {improvement*100:.1f} pp → H2 supported")
        elif improvement == 0:
            print(f"   ⚠️  RAG makes no difference → H2 not supported")
        else:
            print(f"   ⚠️  RAG hurts accuracy → investigate retrieval quality")
        print(f"{'='*60}")

        # Paired significance test on identical cases
        stats = mcnemar_exact(results_rag, results_no_rag)

        # Save combined results
        combined = pd.concat([results_rag, results_no_rag], ignore_index=True)
        p_comb = f"outputs/diagnosis_results_COMBINED{tag}.csv"
        combined.to_csv(p_comb, index=False)
        print(f"💾 Combined results saved to {p_comb}")
        if stats:
            import json as _j
            p_stats = f"outputs/diagnosis_mcnemar{tag}.json"
            Path(p_stats).write_text(_j.dumps(
                {**stats, "acc_rag": acc_rag, "acc_no_rag": acc_no_rag,
                 "n_kb_documents": n_docs}, indent=2))
            print(f"💾 McNemar result saved to {p_stats}")

    return results_rag, results_no_rag, acc_rag, acc_no_rag


# ================================================================
# ENTRY POINT
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Diagnosis Agent with and without RAG"
    )
    parser.add_argument("--max_cases", type=int, default=None,
                        help="Limit number of test cases (for quick testing)")
    # ── NEW FLAGS ────────────────────────────────────────────────
    parser.add_argument("--rag_only", action="store_true",
                        help="Only run the WITH-RAG evaluation")
    parser.add_argument("--no_rag_only", action="store_true",
                        help="Only run the WITHOUT-RAG evaluation")
    parser.add_argument("--test_set", type=str,
                        default="data/knowledge_base/diagnosis_test_set.json",
                        help="Test set JSON. Use diagnosis_test_set_100.json for "
                             "the 120-case balanced set.")
    parser.add_argument("--kb_file", type=str, nargs="+", default=None,
                        help="Knowledge base file(s) to index. Default: "
                             "anomaly_cases.json (60 cases). The test set is "
                             "rejected automatically.")
    parser.add_argument("--tag", type=str, default="",
                        help="Suffix for the output filenames, e.g. _120cases, so "
                             "a new run does not overwrite the previous one.")
    # ────────────────────────────────────────────────────────────
    args = parser.parse_args()

    run_rag    = not args.no_rag_only
    run_no_rag = not args.rag_only

    test_set_path = args.test_set
    test_cases    = load_test_set(test_set_path)
    print(f"📂 Loaded {len(test_cases)} test cases from {test_set_path}")

    # Duplicate check: McNemar treats cases as independent, and a repeated test
    # case is counted twice, which distorts both the accuracy and the p-value.
    texts = [c["text"].strip() for c in test_cases]
    n_dup = len(texts) - len(set(texts))
    if n_dup:
        print(f"⚠️  {n_dup} of {len(texts)} test cases are exact duplicates of an "
              f"earlier case ({len(set(texts))} unique). Duplicates break the "
              f"independence McNemar assumes — prefer a test set with none.")

    evaluate_diagnosis_agent(
        test_cases,
        max_cases=args.max_cases,
        run_rag=run_rag,
        run_no_rag=run_no_rag,
        kb_files=args.kb_file,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
