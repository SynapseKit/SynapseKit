"""Federated retrieval across multiple sources (local or remote)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from ..loaders.web import _validate_url


@dataclass
class _Result:
    text: str
    score: float | None
    metadata: dict | None
    source: str | None


class FederatedRetriever:
    """Fan out a query to multiple sources and fuse results.

    Sources can be:
    - local retrievers (any object with async retrieve/retrieve_with_scores)
    - remote HTTP endpoints that implement the retrieve protocol
    """

    def __init__(
        self,
        sources: list[dict],
        fusion: str = "rrf",
        top_k: int = 10,
        timeout_ms: int = 3000,
        rrf_k: int = 60,
        dedup_threshold: float = 0.92,
    ) -> None:
        if not sources:
            raise ValueError("At least one source is required.")
        self._sources = sources
        self._fusion = fusion
        self._top_k = top_k
        self._timeout_ms = timeout_ms
        self._rrf_k = rrf_k
        self._dedup_threshold = dedup_threshold

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: dict | None = None,
    ) -> list[str]:
        results = await self.retrieve_with_scores(
            query, top_k=top_k, metadata_filter=metadata_filter
        )
        return [r["text"] for r in results]

    async def retrieve_with_scores(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        k = top_k or self._top_k

        tasks = [self._fetch_source(source, query, k, metadata_filter) for source in self._sources]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)

        all_results: list[list[_Result]] = []
        for out in gathered:
            if isinstance(out, BaseException):
                continue
            all_results.append(out)

        fused = self._fuse(all_results, k)
        return [
            {
                "text": r.text,
                "score": r.score,
                "metadata": r.metadata or {},
                "source": r.source,
            }
            for r in fused
        ]

    async def _fetch_source(
        self,
        source: dict,
        query: str,
        top_k: int,
        metadata_filter: dict | None,
    ) -> list[_Result]:
        name = source.get("name")
        timeout_s = self._source_timeout(source)

        if "retriever" in source:
            retriever = source["retriever"]
            coro = self._fetch_local(retriever, query, top_k, metadata_filter, name)
            return await asyncio.wait_for(coro, timeout=timeout_s)

        if "url" in source:
            url = source["url"]
            api_key = source.get("api_key")
            coro = self._fetch_remote(url, api_key, query, top_k, metadata_filter, name)
            return await asyncio.wait_for(coro, timeout=timeout_s)

        raise ValueError("Each source must define either 'retriever' or 'url'.")

    def _source_timeout(self, source: dict) -> float:
        """Return timeout seconds for a source (defaults to instance timeout)."""
        return (source.get("timeout_ms") or self._timeout_ms) / 1000

    async def _fetch_local(
        self,
        retriever: Any,
        query: str,
        top_k: int,
        metadata_filter: dict | None,
        name: str | None,
    ) -> list[_Result]:
        if hasattr(retriever, "retrieve_with_scores"):
            raw = await retriever.retrieve_with_scores(
                query,
                top_k=top_k,
                metadata_filter=metadata_filter,
            )
            return [
                _Result(
                    text=r.get("text", ""),
                    score=r.get("score"),
                    metadata=r.get("metadata") or {},
                    source=name,
                )
                for r in raw
                if r.get("text")
            ]

        raw = await retriever.retrieve(query, top_k=top_k, metadata_filter=metadata_filter)
        return [_Result(text=t, score=None, metadata={}, source=name) for t in raw]

    async def _fetch_remote(
        self,
        url: str,
        api_key: str | None,
        query: str,
        top_k: int,
        metadata_filter: dict | None,
        name: str | None,
    ) -> list[_Result]:
        _validate_url(url)
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required: pip install synapsekit[web]") from None

        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "query": query,
            "top_k": top_k,
            "metadata_filter": metadata_filter or {},
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        results = data.get("results", data)
        out: list[_Result] = []
        for r in results:
            text = r.get("text") if isinstance(r, dict) else None
            if not text:
                continue
            out.append(
                _Result(
                    text=text,
                    score=r.get("score") if isinstance(r, dict) else None,
                    metadata=r.get("metadata") if isinstance(r, dict) else {},
                    source=name,
                )
            )
        return out

    def _fuse(self, results: list[list[_Result]], top_k: int) -> list[_Result]:
        if self._fusion == "interleave":
            return self._interleave(results, top_k)
        if self._fusion == "score":
            return self._score_fusion(results, top_k)
        return self._rrf(results, top_k)

    def _rrf(self, results: list[list[_Result]], top_k: int) -> list[_Result]:
        scores: dict[str, float] = {}
        best: dict[str, _Result] = {}
        for group in results:
            for rank, r in enumerate(group):
                key = self._norm(r.text)
                scores[key] = scores.get(key, 0.0) + 1.0 / (self._rrf_k + rank + 1)
                best[key] = self._choose_best(best.get(key), r)

        ordered = sorted(scores, key=lambda k: scores[k], reverse=True)
        return self._dedup([best[k] for k in ordered], top_k)

    def _score_fusion(self, results: list[list[_Result]], top_k: int) -> list[_Result]:
        scores: dict[str, float] = {}
        best: dict[str, _Result] = {}

        for group in results:
            raw_scores = [
                (r.score if r.score is not None else 1.0 / (idx + 1)) for idx, r in enumerate(group)
            ]
            if not raw_scores:
                continue
            min_s, max_s = min(raw_scores), max(raw_scores)
            denom = max_s - min_s

            for idx, r in enumerate(group):
                base = r.score if r.score is not None else 1.0 / (idx + 1)
                norm = 1.0 if denom == 0 else (base - min_s) / denom
                key = self._norm(r.text)
                scores[key] = scores.get(key, 0.0) + norm
                best[key] = self._choose_best(best.get(key), r)

        ordered = sorted(scores, key=lambda k: scores[k], reverse=True)
        return self._dedup([best[k] for k in ordered], top_k)

    def _interleave(self, results: list[list[_Result]], top_k: int) -> list[_Result]:
        buckets = [list(group) for group in results]
        output: list[_Result] = []
        seen: list[_Result] = []

        while len(output) < top_k and any(buckets):
            for bucket in buckets:
                if not bucket or len(output) >= top_k:
                    continue
                candidate = bucket.pop(0)
                seen.append(candidate)
                if not self._is_duplicate(candidate.text, output):
                    output.append(candidate)
        return output

    def _dedup(self, items: list[_Result], top_k: int) -> list[_Result]:
        output: list[_Result] = []
        for r in items:
            if self._is_duplicate(r.text, output):
                continue
            output.append(r)
            if len(output) >= top_k:
                break
        return output

    def _is_duplicate(self, text: str, existing: list[_Result]) -> bool:
        key = self._norm(text)
        for r in existing:
            other = self._norm(r.text)
            if key == other:
                return True
            if SequenceMatcher(None, key, other).ratio() >= self._dedup_threshold:
                return True
        return False

    @staticmethod
    def _norm(text: str) -> str:
        return " ".join(text.lower().split())

    @staticmethod
    def _choose_best(current: _Result | None, candidate: _Result) -> _Result:
        if current is None:
            return candidate
        if (candidate.score or 0.0) > (current.score or 0.0):
            return candidate
        return current
