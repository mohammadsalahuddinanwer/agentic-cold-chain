"""
verify_retrieval_similarity.py
==============================
Checks what the RAG retriever really returns, so that Section 5.6 of the thesis
quotes a measured similarity instead of a number that appeared inside an
LLM-generated "Observation:" block.


What it does
  1. builds the index exactly as evaluate_diagnosis_accuracy.py does
     (same file filter, so the 60-case knowledge base, test set excluded);
  2. runs the Section 5.6 demo query and a few variants;
  3. prints, for every hit, the similarity the code computes AND the cosine
     similarity it corresponds to, so the two are never confused again;
  4. prints the ceiling: the similarity when the query IS a knowledge-base
     entry, word for word. Nothing can score higher than that;
  5. reports how many of those hits would survive the 0.75 threshold that
     the retriever used to default to before consolidation.

Why the two numbers differ
  The code does NOT compute cosine similarity. FAISS IndexFlatL2 returns the
  SQUARED L2 distance, and the code maps it with

      similarity = 1 / (1 + d^2)

  all-MiniLM-L6-v2 L2-normalises its output, so d^2 = 2 - 2*cos and therefore

      similarity = 1 / (3 - 2*cos)      cos = (3 - 1/similarity) / 2

  which bottoms out at 1/3 for unrelated text and only reaches 1.0 for an exact
  duplicate. Reference points:

      cos 0.30 -> 0.417     cos 0.70 -> 0.625     cos 0.90 -> 0.833
      cos 0.50 -> 0.500     cos 0.80 -> 0.714     cos 1.00 -> 1.000
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

from config import Config
from diagnosis_agent.rag_retriever import RAGRetriever   # the 0.30 module used in the evaluation

# The query the Section 5.6 walk-through describes, plus variants, so the
# reported figure does not hang on one exact wording.
QUERIES = [
    "rapid temperature rise refrigeration unit unusual noise",
    "Temperature rose from 4.0C to 11.0C in 20 minutes. No door open detected. "
    "Refrigeration unit making unusual noise.",
    "temperature rising rapidly no door event grinding noise from cooling unit",
    "snowstorm temperature rise external weather",
    "main power outage total electrical loss",
]

DEPLOYED_THRESHOLD = 0.75      # the original default, before rag_retriever_new.py was removed


def cosine_from_similarity(sim: float) -> float:
    """Invert similarity = 1 / (3 - 2*cos) for unit-norm embeddings."""
    return (3.0 - 1.0 / sim) / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.30,
                    help="threshold used in the evaluated configuration")
    args = ap.parse_args()

    # Same knowledge-base selection as evaluate_diagnosis_accuracy.py
    excluded = {"diagnosis_test_set.json", "diagnosis_test_set_100.json",
                "anomaly_cases _bkp.json", "operational_logs.json",
                "resolution_strategies.json", "anomaly_cases_59.json",
                "anomaly_cases_100.json"}
    knowledge_files = [f for f in sorted(Config.KNOWLEDGE_BASE_DIR.glob("*.json"))
                       if f.name not in excluded]
    print("Knowledge files:", [f.name for f in knowledge_files])

    rag = RAGRetriever()
    rag.build_index(knowledge_files)          # <-- the argument the earlier snippet was missing
    print(f"Documents indexed: {len(rag.documents)}\n")

    print("=" * 78)
    print("MEASURED RETRIEVAL  (similarity = 1/(1+d^2); cos is the equivalent cosine)")
    print("=" * 78)
    all_sims = []
    for q in QUERIES:
        hits = rag.retrieve(q, top_k=args.top_k, similarity_threshold=args.threshold)
        print(f"\nQuery: {q[:72]}{'...' if len(q) > 72 else ''}")
        if not hits:
            print("  no hits above threshold")
            continue
        for h in hits:
            sim = h["similarity"]
            all_sims.append(sim)
            print(f"  sim={sim:.3f}  cos={cosine_from_similarity(sim):.3f}  "
                  f"type={h['metadata'].get('type', '?'):<22}{h['text'][:56]}...")
        kept = sum(s >= DEPLOYED_THRESHOLD for s in (h['similarity'] for h in hits))
        print(f"  -> would survive the deployed {DEPLOYED_THRESHOLD} threshold: {kept}/{len(hits)}")

    # Ceiling: query is a knowledge-base entry verbatim
    verbatim = rag.documents[0]["text"]
    top = rag.retrieve(verbatim, top_k=1, similarity_threshold=0.0)
    print("\n" + "=" * 78)
    print("CEILING: query is a knowledge-base entry, word for word")
    print("=" * 78)
    if top:
        print(f"  sim={top[0]['similarity']:.3f}  cos={cosine_from_similarity(top[0]['similarity']):.3f}")
        print("  Nothing retrieved from a real sensor description can exceed this.")

    if all_sims:
        print(f"\nAcross the demo queries: min {min(all_sims):.3f}, "
              f"max {max(all_sims):.3f}, mean {sum(all_sims)/len(all_sims):.3f}")
    print("\nQuote the MEASURED figure in Section 5.6, and call it by its name: "
          "\n  'similarity score (1/(1+squared L2 distance)) over MiniLM embeddings',"
          "\n  not 'cosine similarity', unless you report the cos column instead.")

    out = Path("outputs"); out.mkdir(exist_ok=True)
    (out / "retrieval_similarity_check.json").write_text(json.dumps(
        {"n_documents": len(rag.documents),
         "threshold_evaluated": args.threshold,
         "threshold_deployed": DEPLOYED_THRESHOLD,
         "similarities": all_sims,
         "ceiling": top[0]["similarity"] if top else None}, indent=2))
    print("\nSaved outputs/retrieval_similarity_check.json")


if __name__ == "__main__":
    main()
