"""Main Diagnosis Agent interface combining RAG and ReAct."""
from typing import Dict, Any, Optional
from .rag_retriever import RAGRetriever
from .re_act_agent import ReActAgent
from .llm_finetune import LlamaFineTuner


class DiagnosisAgent:
    """
    Main Diagnosis Agent that orchestrates anomaly classification
    using LLM + RAG + ReAct reasoning.
    """

    def __init__(self, llm, rag_retriever: Optional[RAGRetriever] = None):
        self.llm = llm
        self.rag = rag_retriever or RAGRetriever()
        self.react_agent = ReActAgent(llm, self.rag)

    def diagnose(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Diagnose an anomaly using ReAct reasoning.

        Args:
            context: Contains monitor_prediction, sensor_data, operational_logs

        Returns:
            Diagnosis with anomaly_type, severity, root_cause, confidence
        """
        return self.react_agent.diagnose(context)

    def classify_with_rag(self, query: str) -> Dict[str, Any]:
        """
        Direct classification using RAG retrieval without full ReAct loop.
        Useful for quick classification of known anomaly types.
        """
        # Retrieve similar cases
        similar = self.rag.retrieve(query, top_k=3)

        if not similar:
            return {
                "anomaly_type": "unknown",
                "confidence": 0.5,
                "similar_cases": []
            }

        # Extract most common type from retrieved cases
        types = [case.get("metadata", {}).get("type", "unknown") for case in similar]
        from collections import Counter
        most_common = Counter(types).most_common(1)[0][0]

        # Use LLM to refine classification
        prompt = f"""
Based on the following similar anomaly cases:
{similar}

Classify the anomaly described by: {query}

Return a JSON with:
- anomaly_type: the most likely type
- confidence: 0-1 confidence score
- reasoning: brief explanation
"""
        response = self.llm.generate(prompt)

        try:
            import json
            result = json.loads(response)
            return result
        except:
            return {
                "anomaly_type": most_common,
                "confidence": 0.7,
                "raw_response": response
            }

    def set_knowledge_base(self, knowledge_files):
        """Build or rebuild the RAG index from knowledge files."""
        self.rag.build_index(knowledge_files)