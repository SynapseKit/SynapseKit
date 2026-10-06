"""
Continuous fine-tuning pipeline — end-to-end demo.

Demonstrates the full closed-loop cycle using in-memory / stub providers
so the example runs without real API keys or network access.

Stages shown
------------
1. Collect production feedback (positive + negative with corrections)
2. Generate a JSONL training dataset
3. Start a fine-tuning job via ContinuousTrainer (stub provider)
4. Simulate the job completing
5. Activate A/B test rollout (with eval gate demo)
6. Record A/B test observations
7. Evaluate and advance the rollout via ContinuousTrainer
8. Run cost-benefit analysis

Usage
-----
    python examples/continuous_finetuning.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from synapsekit.training.ab_testing import ABTestRouter
from synapsekit.training.cost_analysis import CostBenefitAnalyzer, InferenceProfile
from synapsekit.training.dataset import TrainingDataGenerator
from synapsekit.training.feedback import FeedbackCollector, InMemoryFeedbackBackend
from synapsekit.training.finetune import AnthropicFineTuneProvider
from synapsekit.training.orchestrator import ContinuousTrainer
from synapsekit.training.rollout import AutoRolloutManager
from synapsekit.training.types import ABTestResult, RolloutPolicy

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_MODEL = "claude-3-haiku-20240307"
FINETUNED_MODEL = "claude-3-haiku-ft-demo"
DATASET_PATH = os.path.join(tempfile.gettempdir(), "synapsekit_demo_training.jsonl")


def _sep(title: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print("-" * 60)


# ── Stage 1: collect feedback ──────────────────────────────────────────────────


async def collect_feedback(trainer: ContinuousTrainer) -> None:
    _sep("Stage 1 — Collecting production feedback")

    feedback_pairs = [
        # (query, response, feedback_type, corrected_response, latency_ms, cost_usd)
        ("What is the capital of France?", "Paris", "positive", None, 120.0, 0.0002),
        ("Summarize the French Revolution.", "It was a period of major change.", "positive", None, 310.0, 0.0008),
        ("Explain photosynthesis.", "Plants make food using sunlight.", "positive", None, 180.0, 0.0004),
        ("What is 2 + 2?", "Five.", "negative", "4", 95.0, 0.0001),
        ("Who wrote Hamlet?", "Dickens.", "negative", "Shakespeare wrote Hamlet.", 100.0, 0.0001),
        ("What is the boiling point of water?", "100°C at standard pressure.", "positive", None, 140.0, 0.0003),
        ("Translate 'hello' to Spanish.", "Hola.", "positive", None, 90.0, 0.0001),
        ("What is machine learning?", "It's a thing computers do.", "negative",
         "Machine learning is a branch of AI that enables systems to learn from data.", 200.0, 0.0005),
        ("Name the planets in our solar system.",
         "Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune.", "positive", None, 260.0, 0.0006),
        ("What causes thunder?", "It's caused by lightning heating air rapidly.", "positive", None, 130.0, 0.0003),
    ]

    for query, response, fb_type, correction, latency, cost in feedback_pairs:
        trainer.record_feedback(
            query=query,
            response=response,
            feedback=fb_type,  # type: ignore[arg-type]
            corrected_response=correction,
            latency_ms=latency,
            cost_usd=cost,
            metadata={"source": "demo"},
        )
        print(f"  [{fb_type:8s}] {query[:50]!r}")

    # Flush ensures all samples are persisted before the next stage reads them.
    await trainer._collector.flush()
    stored = await trainer._collector.stored_count()
    print(f"\n  Stored {stored} feedback samples.")


# ── Stage 2 + 3: generate dataset and trigger training ────────────────────────


async def trigger_training(trainer: ContinuousTrainer) -> None:
    _sep("Stage 2+3 — Generating dataset and starting fine-tune job")

    job = await trainer.maybe_trigger_training()
    if job is None:
        print("  Not enough new feedback yet — no job started.")
        return
    print(f"  Job started: {job.job_id} (status={job.status})")
    print(f"  Dataset written to: {DATASET_PATH}")


# ── Stage 4: simulate job completion ─────────────────────────────────────────


async def simulate_job_complete(
    trainer: ContinuousTrainer,
    provider: AnthropicFineTuneProvider,
) -> str:
    _sep("Stage 4 — Simulating job completion")

    job = trainer.pending_job
    if job is None:
        print("  No pending job.")
        return FINETUNED_MODEL

    provider._update_status(job.job_id, "succeeded", fine_tuned_model=FINETUNED_MODEL)
    updated = await trainer.check_job_status()
    print(f"  Job status: {updated.status if updated else 'unknown'}")
    print(f"  Fine-tuned model: {updated.fine_tuned_model if updated else 'N/A'}")
    trainer.set_training_cost(6.00)  # demo: $6 training cost
    return FINETUNED_MODEL


# ── Stage 5: A/B test activation ──────────────────────────────────────────────


async def activate_ab_test(
    trainer: ContinuousTrainer,
    router: ABTestRouter,
    rollout_manager: AutoRolloutManager,
) -> None:
    _sep("Stage 5 — Activating A/B test rollout")

    activated = await trainer.activate_ab_test()
    if not activated:
        print("  Rollout blocked by eval gate.")
        return

    print(f"  Rollout active at {rollout_manager.state.current_pct:.1f}% fine-tuned traffic.")
    print(f"  Policy stages: {rollout_manager.policy.stages}")

    print("\n  Sample routing:")
    for uid in ["alice", "bob", "carol", "dave", "eve"]:
        model, variant = router.route(uid)
        print(f"    {uid:8s} -> {variant:10s} ({model})")


# ── Stage 6: record A/B observations ─────────────────────────────────────────


def record_ab_observations(router: ABTestRouter) -> None:
    _sep("Stage 6 — Recording A/B test observations")

    import random
    random.seed(42)

    base_results = [
        ABTestResult(
            user_id=f"base-user-{i}",
            model_variant="base",
            latency_ms=random.gauss(220, 30),
            cost_usd=0.0012,
            feedback="positive" if random.random() < 0.55 else "negative",
            eval_score=random.uniform(0.6, 0.85),
        )
        for i in range(80)
    ]
    ft_results = [
        ABTestResult(
            user_id=f"ft-user-{i}",
            model_variant="finetuned",
            latency_ms=random.gauss(195, 25),
            cost_usd=0.0010,
            feedback="positive" if random.random() < 0.72 else "negative",
            eval_score=random.uniform(0.72, 0.92),
        )
        for i in range(60)
    ]

    for r in base_results + ft_results:
        router.record_result(r)

    print(f"  Recorded {len(base_results)} base results, {len(ft_results)} fine-tuned results.")


# ── Stage 7: evaluate & advance rollout ───────────────────────────────────────


async def evaluate_rollout(trainer: ContinuousTrainer) -> None:
    _sep("Stage 7 — Evaluating A/B metrics and rollout decision")

    metrics, _ = await trainer.run_evaluation_cycle()
    print(f"  Base positive rate:       {metrics.base_positive_rate:.2%}")
    print(f"  Fine-tuned positive rate: {metrics.finetuned_positive_rate:.2%}")
    print(f"  Quality delta:            {metrics.quality_delta_pct:+.1f}%")
    print(f"  Latency delta:            {metrics.latency_delta_pct:+.1f}%")
    print(f"  Cost delta:               {metrics.cost_delta_pct:+.1f}%")
    print(f"\n  Rollout now at: {trainer._rollout_manager.state.current_pct:.1f}%")


# ── Stage 8: cost-benefit analysis ────────────────────────────────────────────


async def cost_benefit(trainer: ContinuousTrainer) -> None:
    _sep("Stage 8 — Cost-benefit analysis")

    profile = InferenceProfile(
        monthly_requests=500_000,
        avg_input_tokens=512,
        avg_output_tokens=256,
        base_model=BASE_MODEL,
        finetuned_model=FINETUNED_MODEL,
    )
    _, report = await trainer.run_evaluation_cycle(profile=profile)

    if report is None:
        print("  No report generated (no inference profile).")
        return

    print(f"  Estimated training cost:   ${report.training_cost_usd:.4f}")
    print(f"  Monthly inference savings: ${report.monthly_inference_savings_usd:.2f}")
    print(f"  Quality improvement:       {report.quality_improvement_pct:+.1f}%")
    if report.estimated_payback_days == float("inf"):
        print("  Estimated payback:         ∞ (no cost savings)")
    else:
        print(f"  Estimated payback:         {report.estimated_payback_days:.1f} days")


# ── Main ───────────────────────────────────────────────────────────────────────


async def main() -> None:
    print("=" * 60)
    print("  SynapseKit Continuous Fine-Tuning Pipeline Demo")
    print("=" * 60)

    # ── Initialise components ─────────────────────────────────────────────────
    backend = InMemoryFeedbackBackend()
    collector = FeedbackCollector(backend=backend, queue_maxsize=1_000)
    collector.start()

    generator = TrainingDataGenerator(
        system_prompt="You are a helpful, accurate, and concise AI assistant."
    )
    provider = AnthropicFineTuneProvider(api_key=None)
    router = ABTestRouter(
        base_model=BASE_MODEL,
        finetuned_model=FINETUNED_MODEL,
        rollout_pct=0.0,
        experiment_id="demo-experiment-v1",
    )
    policy = RolloutPolicy(
        stages=[5.0, 25.0, 50.0, 100.0],
        min_samples_per_stage=50,
        improvement_threshold_pct=2.0,
        latency_regression_pct=20.0,
        cost_regression_pct=15.0,
    )
    rollout_manager = AutoRolloutManager(router=router, policy=policy)
    analyzer = CostBenefitAnalyzer()

    trainer = ContinuousTrainer(
        collector=collector,
        generator=generator,
        provider=provider,
        router=router,
        rollout_manager=rollout_manager,
        analyzer=analyzer,
        base_model=BASE_MODEL,
        min_feedback_before_train=8,  # low threshold for demo
        dataset_path=DATASET_PATH,
    )

    # ── Run pipeline stages ───────────────────────────────────────────────────
    await collect_feedback(trainer)
    await trigger_training(trainer)
    await simulate_job_complete(trainer, provider)
    await activate_ab_test(trainer, router, rollout_manager)
    record_ab_observations(router)
    await evaluate_rollout(trainer)
    await cost_benefit(trainer)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    await collector.stop()
    try:
        os.unlink(DATASET_PATH)
    except FileNotFoundError:
        pass

    print("\n" + "=" * 60)
    print("  Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
