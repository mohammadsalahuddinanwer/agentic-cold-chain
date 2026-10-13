"""RAG retriever with FAISS vector store and sentence transformers."""
import json
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path

class RAGRetriever:
    """
    Retrieval-augmented generation vector store for cold-chain anomalies.
    """
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(embedding_model)
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        self.index = None
        self.documents = []

    def build_index(self, knowledge_files: List[Path]):
        """
        Build FAISS index from knowledge base files.
        Each file should contain JSON entries with "text" and "metadata".
        """
        texts = []
        metadata_list = []

        for file_path in knowledge_files:
            with open(file_path, 'r') as f:
                if file_path.suffix == '.json':
                    data = json.load(f)
                    if isinstance(data, list):
                        for entry in data:
                            texts.append(entry.get("text", ""))
                            metadata_list.append(entry.get("metadata", {}))
                    elif isinstance(data, dict):
                        texts.append(data.get("text", ""))
                        metadata_list.append(data.get("metadata", {}))
                elif file_path.suffix == '.txt':
                    text = f.read()
                    chunks = self._chunk_text(text, chunk_size=500)
                    for i, chunk in enumerate(chunks):
                        texts.append(chunk)
                        metadata_list.append({"source": str(file_path), "chunk": i})

        if not texts:
            print("No documents found to build index.")
            return

        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(np.array(embeddings))
        self.documents = [{"text": t, "metadata": m} for t, m in zip(texts, metadata_list)]
        print(f"Built index with {len(self.documents)} documents.")

    def retrieve(self, query: str, top_k: int = 3,
                 # ── CHANGED: lowered from 0.75 to 0.30 ──────────────
                 # 0.75 was too strict — valid matches were being rejected
                 # 0.30 returns all meaningful semantic matches
                 similarity_threshold: float = 0.30) -> List[Dict]:
        """
        Retrieve most relevant documents for query.
        """
        if self.index is None:
            return []

        query_embedding = self.encoder.encode([query])
        distances, indices = self.index.search(query_embedding, top_k)

        # Convert L2 distances to similarity scores
        similarities = 1 / (1 + distances[0])

        results = []
        for idx, sim in zip(indices[0], similarities):
            if sim >= similarity_threshold and idx < len(self.documents):
                result = self.documents[idx].copy()
                result["similarity"] = float(sim)
                results.append(result)

        return results

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks for better retrieval."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def save_index(self, path: str):
        """Save FAISS index and documents."""
        if self.index is not None:
            faiss.write_index(self.index, f"{path}_index.faiss")
            with open(f"{path}_docs.json", 'w') as f:
                json.dump(self.documents, f)

    def load_index(self, path: str):
        """Load saved FAISS index and documents."""
        self.index = faiss.read_index(f"{path}_index.faiss")
        with open(f"{path}_docs.json", 'r') as f:
            self.documents = json.load(f)
