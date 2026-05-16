"""Builder to extract Knowledge Graph structures from text."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...llm.base import BaseLLM
    from .backends import BaseGraphStore

_EXTRACT_ENTITIES_PROMPT = """\
Extract the key entities (people, places, concepts, organizations, etc.) from the following text.
Return only a JSON array of strings, nothing else.

Text: {text}
"""

_EXTRACT_TRIPLES_PROMPT = """\
Extract knowledge graph triples from the following text.
For each relationship found, provide the subject, predicate, object, and a confidence score between 0.0 and 1.0.

Return the result STRICTLY as a JSON array of objects with the keys "subject", "predicate", "object", and "confidence".
Do not include any other text or markdown formatting.

Text: {text}
"""


class KnowledgeGraphBuilder:
    """Builder to extract entities and triples from documents using an LLM."""

    def __init__(self, llm: BaseLLM, store: BaseGraphStore) -> None:
        self._llm = llm
        self._store = store

    async def extract_entities(self, text: str) -> list[str]:
        """Extract a list of entities from the given text."""
        prompt = _EXTRACT_ENTITIES_PROMPT.format(text=text)
        response = await self._llm.generate(prompt)
        try:
            # Clean up markdown formatting if the LLM adds it
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]

            entities = json.loads(cleaned.strip())
            if isinstance(entities, list):
                return [str(e) for e in entities]
        except json.JSONDecodeError:
            # Fallback if the LLM doesn't return valid JSON
            raw = response.replace("[", "").replace("]", "").replace("\n", ",")
            rows = list(csv.reader([raw], skipinitialspace=True))
            if rows:
                return [e.strip() for e in rows[0] if e.strip()]
            return []
        return []

    async def extract_triples(self, text: str) -> list[dict]:
        """Extract relationship triples with confidence scores."""
        prompt = _EXTRACT_TRIPLES_PROMPT.format(text=text)
        response = await self._llm.generate(prompt)
        try:
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]

            triples = json.loads(cleaned.strip())
            if isinstance(triples, list):
                return triples
        except json.JSONDecodeError:
            pass
        return []

    async def build_from_documents(self, docs: list[str], doc_ids: list[str] | None = None) -> None:
        """Process documents, extract triples, and store them in the graph backend."""
        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(docs))]

        if len(docs) != len(doc_ids):
            raise ValueError("Length of docs and doc_ids must match.")

        for text, doc_id in zip(docs, doc_ids, strict=True):
            triples = await self.extract_triples(text)
            for triple in triples:
                if not isinstance(triple, dict):
                    continue
                subject = triple.get("subject")
                predicate = triple.get("predicate")
                obj = triple.get("object")
                confidence = float(triple.get("confidence", 1.0))

                if subject and predicate and obj:
                    self._store.add_triple(subject, predicate, obj, confidence)
                    self._store.add_document_link(subject, doc_id)
                    self._store.add_document_link(obj, doc_id)
