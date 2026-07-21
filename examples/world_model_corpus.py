"""Deterministic synthetic corpus generator for WorldModelRAG demos/benchmarks.

Builds a corpus of causal two-hop "chains" (Person -> Product -> Incident)
plus filler documents. Chain sentences are worded to match
``HeuristicWorldModelExtractor``'s entity/relation/date patterns, so entities,
causal relations, and events are extracted with zero LLM calls. Each chain's
two hops are split across two separate documents so a query naming only the
head (person) cannot trivially vector-match the tail (incident) document by
shared vocabulary -- reaching it requires graph traversal. This is what makes
the WorldModelRAG hybrid-vs-vector-only retrieval comparison in
``benchmarks/world_model_retrieval_bench.py`` a stable, structural gap rather
than an incidental one.

Shared by ``examples/world_model_rag.py`` (10k-doc demo) and
``benchmarks/world_model_retrieval_bench.py`` (retrieval accuracy benchmark)
so corpus-building logic isn't duplicated.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta

_FIRST_NAMES = [
    "Alice", "Bianca", "Carlos", "Diana", "Ewan", "Farah", "Gustavo", "Hana",
    "Ivan", "Julia", "Kenji", "Lena", "Mateo", "Nadia", "Omar", "Priya",
    "Quinn", "Rosa", "Sami", "Tara",
]
_LAST_NAMES = [
    "Chen", "Okafor", "Silva", "Novak", "Haddad", "Kim", "Rossi", "Patel",
    "Nguyen", "Larsen", "Costa", "Ibrahim", "Voss", "Reyes", "Singh",
]
_PRODUCT_WORDS = [
    "Nimbus", "Orion", "Falcon", "Vertex", "Atlas", "Cobalt", "Ember",
    "Halyard", "Juniper", "Kestrel", "Lumen", "Meridian", "Nomad", "Onyx",
    "Pulsar", "Quartz", "Raven", "Solstice", "Terra", "Wraith",
]
_PRODUCT_NOUNS = [
    "Platform", "Gateway", "Engine", "Pipeline", "Console", "Ledger",
    "Beacon", "Router", "Fabric", "Codex",
]
_INCIDENT_WORDS = [
    "Outage", "Regression", "Rollback", "Breach", "Slowdown", "Failover",
    "Corruption", "Timeout", "Overrun", "Disruption",
]
_INCIDENT_NOUNS = [
    "Case", "Report", "Event", "Ticket", "Alert", "Review", "Postmortem",
    "Escalation", "Advisory", "Bulletin",
]
_BASE_DATE = date(2023, 1, 1)
_FILLER_TEMPLATES = [
    "{entity} was mentioned in the quarterly newsletter.",
    "{entity} came up during the weekly sync.",
    "{entity} appeared in the internal changelog.",
    "{entity} was filed for reference in the notes.",
    "{entity} was discussed briefly in the planning retro.",
    "{entity} documentation was updated this sprint.",
]


@dataclass(frozen=True)
class Chain:
    """A causal two-hop chain: person worked on product, product caused incident."""

    person: str
    product: str
    incident: str
    head_doc_id: str
    tail_doc_id: str


def question_for_chain(chain: Chain) -> str:
    """A multi-hop question naming only the chain's head entity (the person).

    Deliberately omits the tail entity's name so answering requires following
    the causal edge(s) rather than matching the tail document by keyword.
    """
    return f"What did {chain.person} work on, and what went wrong afterward?"


def generate_corpus(
    n_docs: int = 10_000, seed: int = 42
) -> tuple[list[dict], list[Chain]]:
    """Generate a deterministic synthetic corpus with causal ground-truth chains.

    Returns ``(docs, chains)`` where each doc is ``{"text": ..., "metadata":
    {"source": doc_id}}`` and each ``Chain`` records the entities and document
    ids that make up one causal path, for use as retrieval ground truth.
    """
    rng = random.Random(seed)
    n_chains = max(1, n_docs // 20)
    docs: list[dict] = []
    chains: list[Chain] = []

    for idx in range(n_chains):
        person = f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}{idx}"
        product = f"{rng.choice(_PRODUCT_WORDS)} {rng.choice(_PRODUCT_NOUNS)}{idx}"
        incident = f"{rng.choice(_INCIDENT_WORDS)} {rng.choice(_INCIDENT_NOUNS)}{idx}"
        # Full ISO dates, unique per chain/hop. HeuristicWorldModelExtractor creates
        # one shared event node per distinct date it sees; reusing a coarse
        # year-only date across many chains would merge them all into one hub
        # node and falsely bridge unrelated chains in the graph.
        date_a = (_BASE_DATE + timedelta(days=2 * idx)).isoformat()
        date_b = (_BASE_DATE + timedelta(days=2 * idx + 1)).isoformat()

        head_doc_id = f"chain{idx:05d}_a"
        tail_doc_id = f"chain{idx:05d}_b"
        docs.append(
            {
                "text": f"{person} worked on {product} in {date_a}.",
                "metadata": {"source": head_doc_id},
            }
        )
        docs.append(
            {
                "text": f"{product} caused {incident} in {date_b}.",
                "metadata": {"source": tail_doc_id},
            }
        )
        chains.append(
            Chain(
                person=person,
                product=product,
                incident=incident,
                head_doc_id=head_doc_id,
                tail_doc_id=tail_doc_id,
            )
        )

    vocabulary = [c.person for c in chains] + [c.product for c in chains] + [
        c.incident for c in chains
    ]
    filler_count = max(0, n_docs - len(docs))
    for fidx in range(filler_count):
        entity = rng.choice(vocabulary)
        template = rng.choice(_FILLER_TEMPLATES)
        docs.append(
            {
                "text": template.format(entity=entity),
                "metadata": {"source": f"filler{fidx:05d}"},
            }
        )

    rng.shuffle(docs)
    return docs, chains
