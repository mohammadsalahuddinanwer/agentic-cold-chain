# Save as test_retrieval.py in cold_chain_agent/
import sys
sys.path.insert(0, '.')
from diagnosis_agent.rag_retriever import RAGRetriever
from config import Config
from pathlib import Path

rag = RAGRetriever()
knowledge_files = [Config.KNOWLEDGE_BASE_DIR / "anomaly_cases.json"]
rag.build_index(knowledge_files)

# Test queries matching your failing cases
test_queries = [
    "snowstorm temperature rise external weather",
    "main power outage total electrical loss",
    "electrical fault cooling system shutdown",
    "GPS lost temperature stable",
]

print("\n=== RETRIEVAL TEST ===")
for q in test_queries:
    results = rag.retrieve(q, top_k=3)
    print(f"\nQuery: '{q}'")
    if results:
        for r in results:
            print(f"   Found: type={r['metadata']['type']} sim={r['similarity']:.3f}")
            print(f"     Text: {r['text'][:80]}...")
    else:
        print(f"   No results — threshold still too high")