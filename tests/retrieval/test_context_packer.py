from __future__ import annotations

from synapsekit.retrieval.context_packer import ContextPacker
from synapsekit.retrieval.token_counting import TokenCounter


def _word_counter() -> TokenCounter:
    return TokenCounter(count_fn=lambda text: len([t for t in text.split() if t]))


def test_relevance_lost_in_middle_ordering_places_best_edges():
    packer = ContextPacker(
        max_tokens=100,
        strategy="relevance",
        ordering="lost-in-middle",
        token_counter=_word_counter(),
    )

    chunks = [
        {"text": "alpha", "score": 1.0},
        {"text": "beta", "score": 0.9},
        {"text": "gamma", "score": 0.8},
        {"text": "delta", "score": 0.7},
    ]

    packed = packer.pack(chunks)
    texts = [c["text"] for c in packed]

    assert texts[0] == "alpha"
    assert texts[-1] == "beta"


def test_dedup_removes_near_duplicates_keeps_stronger():
    packer = ContextPacker(
        max_tokens=100,
        strategy="relevance",
        dedup_threshold=0.85,
        ordering="as-is",
        token_counter=_word_counter(),
    )

    chunks = [
        {"text": "refund will be processed in 3 days", "score": 0.9},
        {"text": "refund will be processed in three days", "score": 0.8},
        {"text": "please contact support for account help", "score": 0.3},
    ]

    packed = packer.pack(chunks)
    texts = [c["text"] for c in packed]

    assert "refund will be processed in 3 days" in texts
    assert "please contact support for account help" in texts
    assert len(texts) == 2


def test_token_limit_enforced():
    packer = ContextPacker(
        max_tokens=6,
        strategy="relevance",
        ordering="as-is",
        token_counter=_word_counter(),
    )

    chunks = [
        {"text": "one two three four", "score": 1.0},
        {"text": "five six seven eight", "score": 0.9},
        {"text": "nine", "score": 0.1},
    ]

    packed = packer.pack(chunks)
    token_total = sum(c["token_count"] for c in packed)

    assert token_total <= 6
    assert [c["text"] for c in packed] == ["one two three four", "nine"]


def test_recency_strategy_prefers_latest_timestamp():
    packer = ContextPacker(
        max_tokens=100,
        strategy="recency",
        ordering="as-is",
        token_counter=_word_counter(),
    )

    chunks = [
        {"text": "older", "metadata": {"timestamp": 100}},
        {"text": "newest", "metadata": {"timestamp": 300}},
        {"text": "middle", "metadata": {"timestamp": 200}},
    ]

    packed = packer.pack(chunks)
    assert [c["text"] for c in packed] == ["newest", "middle", "older"]


def test_invalid_max_tokens_raises():
    import pytest

    with pytest.raises(ValueError, match="max_tokens must be > 0"):
        ContextPacker(max_tokens=0, token_counter=_word_counter())


def test_invalid_strategy_raises():
    import pytest

    with pytest.raises(ValueError, match="strategy must be one of"):
        ContextPacker(strategy="blah", token_counter=_word_counter())


def test_invalid_ordering_raises():
    import pytest

    with pytest.raises(ValueError, match="ordering must be one of"):
        ContextPacker(ordering="random", token_counter=_word_counter())


def test_empty_input_returns_empty():
    packer = ContextPacker(max_tokens=100, token_counter=_word_counter())
    assert packer.pack([]) == []
    assert packer.pack_texts([]) == []


def test_plain_string_inputs_normalized():
    packer = ContextPacker(max_tokens=100, ordering="as-is", token_counter=_word_counter())
    result = packer.pack(["hello world", "foo bar"])
    assert len(result) == 2
    assert all(isinstance(c["text"], str) for c in result)


def test_diversity_strategy_accepted():
    packer = ContextPacker(
        max_tokens=100,
        strategy="diversity",
        ordering="as-is",
        token_counter=_word_counter(),
    )
    result = packer.pack([{"text": "alpha beta", "score": 0.9}])
    assert result[0]["text"] == "alpha beta"
