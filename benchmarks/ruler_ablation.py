"""Reproducible offline wiring ablation; canned outputs are not model results."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from synapsekit.evaluation.ruler import (
    LLMRulerJudge,
    RulerGroup,
    RulerMessage,
    RulerScore,
    RulerScorer,
    RulerTrajectory,
    calibrate,
)

FIXTURE = Path(__file__).resolve().parents[1] / "tests/evaluation/fixtures/ruler_ablation.json"


class CannedLLM:
    def __init__(self, fixture: dict) -> None:
        self.fixture = fixture

    async def generate(self, prompt: str) -> str:
        return json.dumps(
            {
                "scores": [
                    {"id": item["id"], "score": item["canned_judge"]}
                    for item in self.fixture["trajectories"]
                ]
            }
        )


async def evaluate(fixture: dict, llm) -> dict:
    """Run one group using the supplied judge; no result is cached or invented."""
    group = RulerGroup(
        fixture["task"],
        tuple(
            RulerTrajectory(item["id"], (RulerMessage("assistant", item["answer"]),))
            for item in fixture["trajectories"]
        ),
    )
    baseline = tuple(
        RulerScore(item["id"], float(item["answer"].strip() == fixture["baseline_exact_answer"]))
        for item in fixture["trajectories"]
    )
    gold = {item["id"]: item["gold"] for item in fixture["trajectories"]}
    baseline_report = calibrate(baseline, gold)  # Validate gold before a potentially paid call.
    scores = await RulerScorer(LLMRulerJudge(llm)).score(group)
    return {
        "fixture_version": fixture["version"],
        "trajectory_count": len(group.trajectories),
        "scores": [asdict(score) for score in scores],
        "ruler": asdict(calibrate(scores, gold)),
        "baseline": asdict(baseline_report),
    }


async def run() -> dict:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    result = await evaluate(fixture, CannedLLM(fixture))
    result["mode"] = "offline canned judge; not model evaluation"
    return result


async def run_real(args) -> dict:
    from synapsekit.llm.base import LLMConfig
    from synapsekit.llm.openai import OpenAILLM

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY must be set for --real")
    raw = args.fixture.read_bytes()
    fixture = json.loads(raw)
    config = LLMConfig(
        model=args.model,
        provider="openai",
        api_key=key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=0,
    )
    async with OpenAILLM(config) as llm:
        result = await evaluate(fixture, llm)
        result.update(
            {
                "mode": "live judge; one group, not evidence of general performance",
                "provider": "openai",
                "model": args.model,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "fixture_sha256": hashlib.sha256(raw).hexdigest(),
                "tokens_used": llm.tokens_used,
            }
        )
        return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline demo by default; --real explicitly enables a paid OpenAI call."
    )
    parser.add_argument("--real", action="store_true", help="authorize a live OpenAI API call")
    parser.add_argument(
        "--fixture", type=Path, help="gold-labelled JSON fixture (required with --real)"
    )
    parser.add_argument("--model", help="OpenAI model identifier (required with --real)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()
    if args.real:
        if not args.fixture or not args.model:
            parser.error("--real requires --fixture and --model")
        result = asyncio.run(run_real(args))
    else:
        if args.fixture or args.model:
            parser.error("--fixture and --model require --real")
        result = asyncio.run(run())
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
