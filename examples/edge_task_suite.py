"""Deterministic 200-task suite for the edge local-vs-cloud benchmark.

Generates seeded tasks across four families that cover the work a small local
model is actually asked to do in an edge deployment (see issue #736): field
**extraction**, **classification**, **json** structured output, and date
**format** normalization.

Every task is graded programmatically -- exact-ish containment, first-label-wins,
JSON field equality, or regex -- so scoring is free, deterministic, and needs no
API calls of its own.

ponytail: programmatic graders only, so the suite is limited to tasks with one
objectively checkable answer. If open-ended quality (tone, coherence, long-form
summarization) ever needs measuring, add an LLM-judge grader mode alongside
these -- don't loosen the existing ones into fuzzy matching.

Shared by ``benchmarks/edge_local_vs_cloud_bench.py`` (real models) and
``benchmarks/test_edge_local_vs_cloud_bench.py`` (CI harness check).
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass

CATEGORIES = ("extraction", "classification", "json", "format")

_LABELS = ("positive", "negative", "neutral")
_NEGATORS = ("not ", "n't ", "non-")

_FIRST_NAMES = [
    "Alice", "Bianca", "Carlos", "Diana", "Ewan", "Farah", "Gustavo", "Hana",
    "Ivan", "Julia", "Kenji", "Lena", "Mateo", "Nadia", "Omar", "Priya",
]
_LAST_NAMES = [
    "Chen", "Okafor", "Silva", "Novak", "Haddad", "Kim", "Rossi", "Patel",
    "Nguyen", "Larsen", "Costa", "Ibrahim", "Voss", "Reyes", "Singh",
]
_PRODUCTS = [
    "Nimbus Platform", "Orion Gateway", "Falcon Engine", "Vertex Pipeline",
    "Atlas Console", "Cobalt Ledger", "Ember Beacon", "Juniper Router",
]
_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
_POSITIVE = [
    "works exactly as advertised and setup took minutes",
    "has been rock solid since we rolled it out",
    "saved our team hours of manual work every week",
]
_NEGATIVE = [
    "crashed twice during our launch window",
    "corrupted our export and support never replied",
    "is far slower than the documentation claims",
]
_NEUTRAL = [
    "was installed on the staging cluster on Tuesday",
    "is documented in the internal runbook",
    "ships with the standard configuration defaults",
]


@dataclass(frozen=True)
class EdgeTask:
    """One benchmark task with an objectively checkable answer.

    ``grader`` selects the scoring mode in :func:`grade`; ``expected`` is
    interpreted per mode (a literal string, a label, ``k=v;k=v`` JSON fields,
    or a regex).
    """

    name: str
    category: str
    prompt: str
    expected: str
    grader: str


def _first_json_object(text: str) -> dict | None:
    """Return the first parseable JSON object in ``text``, if any.

    Models routinely wrap JSON in prose or fences, so scan for balanced braces
    rather than trying to parse the whole response.
    """
    for start in (m.start() for m in re.finditer(r"\{", text)):
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : i + 1])
                    except ValueError:
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break
    return None


def grade(task: EdgeTask, output: str) -> float:
    """Score a model response as 1.0 (correct) or 0.0 (incorrect).

    Graders tolerate surrounding prose -- small models rarely honour "reply with
    only X" -- but never tolerate a wrong answer.
    """
    text = output.strip()
    lowered = text.lower()

    if task.grader == "contains":
        return 1.0 if task.expected.lower() in lowered else 0.0

    if task.grader == "label":
        # First label mentioned wins, so trailing rambling can't flip a correct
        # answer. A short "not "/"non-" prefix check keeps "not positive" from
        # scoring as positive.
        # ponytail: only catches simple adjacent negation, not "far from positive"
        # or sarcasm -- widen the prefix window or add an LLM judge if that
        # shows up in real model output.
        hits = []
        for lbl in _LABELS:
            for m in re.finditer(rf"\b{lbl}\b", lowered):
                prefix = lowered[max(0, m.start() - 5) : m.start()]
                if any(neg in prefix for neg in _NEGATORS):
                    continue
                hits.append((m.start(), lbl))
        if not hits:
            return 0.0
        return 1.0 if min(hits)[1] == task.expected else 0.0

    if task.grader == "json_field":
        parsed = _first_json_object(text)
        if parsed is None:
            return 0.0
        for pair in task.expected.split(";"):
            key, _, value = pair.partition("=")
            if str(parsed.get(key, "")).strip().lower() != value.lower():
                return 0.0
        return 1.0

    if task.grader == "regex":
        return 1.0 if re.search(task.expected, text) else 0.0

    raise ValueError(f"unknown grader: {task.grader!r}")


def _extraction_task(rng: random.Random, i: int) -> EdgeTask:
    product = rng.choice(_PRODUCTS)
    total = f"${rng.randint(100, 9999):,}.{rng.randint(0, 99):02d}"
    invoice = f"INV-{rng.randint(1000, 9999)}"
    return EdgeTask(
        name=f"extraction-{i:03d}",
        category="extraction",
        prompt=(
            f"Invoice {invoice} from {product}, issued "
            f"{rng.choice(_MONTHS)} {rng.randint(1, 28)}, 2023, total {total}, "
            f"paid by card.\n\n"
            "Extract the invoice total. Reply with only the amount."
        ),
        expected=total,
        grader="contains",
    )


def _classification_task(rng: random.Random, i: int) -> EdgeTask:
    label = rng.choice(_LABELS)
    body = {
        "positive": _POSITIVE,
        "negative": _NEGATIVE,
        "neutral": _NEUTRAL,
    }[label]
    return EdgeTask(
        name=f"classification-{i:03d}",
        category="classification",
        prompt=(
            f'Review: "{rng.choice(_PRODUCTS)} {rng.choice(body)}."\n\n'
            "Classify the sentiment as positive, negative, or neutral. "
            "Reply with one word."
        ),
        expected=label,
        grader="label",
    )


def _json_task(rng: random.Random, i: int) -> EdgeTask:
    name = f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"
    age = rng.randint(21, 65)
    return EdgeTask(
        name=f"json-{i:03d}",
        category="json",
        prompt=(
            f"{name} is {age} years old and works on {rng.choice(_PRODUCTS)}.\n\n"
            'Return a JSON object with exactly the keys "name" and "age". '
            "Reply with only JSON."
        ),
        expected=f"name={name};age={age}",
        grader="json_field",
    )


def _format_task(rng: random.Random, i: int) -> EdgeTask:
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    year = rng.randint(2019, 2024)
    return EdgeTask(
        name=f"format-{i:03d}",
        category="format",
        prompt=(
            f"Convert the date '{_MONTHS[month - 1]} {day}, {year}' to "
            "YYYY-MM-DD format. Reply with only the date."
        ),
        expected=rf"\b{year}-{month:02d}-{day:02d}\b",
        grader="regex",
    )


_BUILDERS = {
    "extraction": _extraction_task,
    "classification": _classification_task,
    "json": _json_task,
    "format": _format_task,
}


def generate_tasks(n: int = 200, seed: int = 42) -> list[EdgeTask]:
    """Build ``n`` tasks split evenly across the four categories.

    Deterministic for a given ``seed``. ``n`` is rounded down to a multiple of
    ``len(CATEGORIES)`` so every category is equally weighted and per-category
    accuracies stay comparable.
    """
    if n < len(CATEGORIES):
        raise ValueError(f"n must be at least {len(CATEGORIES)}, got {n}")

    rng = random.Random(seed)
    per_category = n // len(CATEGORIES)
    tasks = [
        _BUILDERS[category](rng, i)
        for category in CATEGORIES
        for i in range(per_category)
    ]
    rng.shuffle(tasks)
    return tasks


if __name__ == "__main__":
    suite = generate_tasks()
    print(f"{len(suite)} tasks across {len(CATEGORIES)} categories\n")
    for task in suite[:4]:
        print(f"--- {task.name} ({task.grader}) ---")
        print(task.prompt)
        print(f"expected: {task.expected}\n")
