from __future__ import annotations

from .base import BaseSplitter


class SemanticSplitter(BaseSplitter):
    """
    Split text at semantic boundaries using sentence embeddings.

    Sentences whose cosine similarity to the next sentence drops below
    *threshold* are treated as split points. Lazy-imports sentence-transformers.
    """

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        threshold: float = 0.5,
        min_chunk_size: int = 50,
    ) -> None:
        self.model_name = model
        self.threshold = threshold
        self.min_chunk_size = min_chunk_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers required: pip install synapsekit[semantic]"
                ) from None
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def split(self, text: str) -> list[str]:
        import numpy as np

        text = text.strip()
        if not text:
            return []

        # Split into sentences (simple heuristic)
        sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
        if len(sentences) <= 1:
            return [text]

        model = self._get_model()
        embeddings = model.encode(sentences)
        embeddings = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embeddings = embeddings / norms

        # Find split points where consecutive similarity drops below threshold
        chunks: list[str] = []
        current_sentences: list[str] = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = float(np.dot(embeddings[i - 1], embeddings[i]))
            if sim < self.threshold and len(". ".join(current_sentences)) >= self.min_chunk_size:
                chunks.append(". ".join(current_sentences) + ".")
                current_sentences = [sentences[i]]
            else:
                current_sentences.append(sentences[i])

        if current_sentences:
            remainder = ". ".join(current_sentences)
            if not remainder.endswith("."):
                remainder += "."
            chunks.append(remainder)

        return chunks
