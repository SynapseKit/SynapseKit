# Relative trajectory rewards (RULER-style)

`RulerScorer` compares multiple rollouts of **one task** in a single async judge call. This implementation is inspired by [OpenPipe RULER](https://art.openpipe.ai/fundamentals/ruler/), not an implementation of ART's API. The judge assigns each trajectory a 0–1 goal-achievement score relative to the group; scores are not automatically normalized, calibrated probabilities, or comparable across different groups.

```python
from synapsekit.evaluation.ruler import (
    LLMRulerJudge, RulerGroup, RulerMessage, RulerScorer, RulerTrajectory,
    calibrate,
)

# llm is an existing configured synapsekit.llm.base.BaseLLM instance.
shared = RulerMessage("user", "Explain 2 + 2.")
group = RulerGroup("Answer accurately and explain briefly", (
    RulerTrajectory("one", (shared, RulerMessage("assistant", "4: two pairs."))),
    RulerTrajectory("two", (shared, RulerMessage("assistant", "5"))),
))
scores = await RulerScorer(LLMRulerJudge(llm)).score(group)
report = calibrate(scores, {"one": 1.0, "two": 0.0})
```

`RulerGroup` requires 2–16 trajectories with unique nonempty IDs. Messages are immutable tuples of `RulerMessage` with `system`, `user`, `assistant`, or `tool` roles. A trajectory has 1–100 messages, each content is at most 20,000 characters; task text is nonempty and at most 20,000 characters; IDs are at most 128 characters; the group has at most 200,000 task-plus-message characters, counting repeated prefixes. These are character limits, not model token limits; provider context limits may be smaller. Identical leading messages across *all* trajectories are sent once, with each suffix in input order. Identical full trajectories remain distinct IDs. The caller is responsible for grouping rollouts of the same task.

An injected async callable accepting a `RulerGroup` may replace `LLMRulerJudge`; it must return a list of `{"id": str, "score": number}` entries. The LLM adapter uses the existing `BaseLLM.generate` interface and requests strict JSON `{"scores": [...]}`. Response entries can be reordered, but the result is returned in group order. Malformed JSON, duplicate JSON keys, malformed entries, missing/extra/duplicate/unknown IDs, booleans, non-numeric, nonfinite, or out-of-range scores raise `ValueError`; transport/provider exceptions propagate. No fallback scores are fabricated.

The prompt explicitly marks the task, IDs, message roles, and message contents as **untrusted data**. This is a mitigation, not a security boundary: an LLM can still follow adversarial text. Do not put secrets into rollouts or send sensitive data to a provider without the appropriate authorization. Validate performance against representative gold data, including adversarial cases, before using scores in consequential workflows.

## Calibration

`calibrate(scores, gold)` is a **diagnostic**, not a fitted transform: it requires exactly matching unique IDs and finite gold values in [0, 1]. It returns mean absolute error against the labels and pairwise accuracy over pairs with *different* gold values. A predicted tie for a non-tied gold pair counts as incorrect. Gold-tied pairs are excluded; if there are no comparable pairs, accuracy is `None`. Label provenance, inter-rater reliability, task selection, model/prompt version, group order, and repeated-run variability remain the caller's responsibility. Because scoring is relative, consider both ranking and scale calibration.

To opt into a fitted transformation, provide judge scores and gold labels from a **separate calibration set**; evaluate the transformed scores on held-out tasks, not the fitting set:

```python
from synapsekit.evaluation.ruler import fit_calibration, calibrate

# calibration_scores and calibration_gold come from labelled calibration tasks;
# test_scores and test_gold are independently held out.
model = fit_calibration(calibration_scores, calibration_gold)
calibrated_test_scores = model.apply(test_scores)
held_out_report = calibrate(calibrated_test_scores, test_gold)
print(model.metadata, held_out_report)
```

The fit uses weighted isotonic regression (pool-adjacent-violators): equal input scores share a mean gold label, violating adjacent blocks are pooled, then predictions are linearly interpolated between sorted knots and held constant outside the observed range. One distinct score produces a constant model. Inputs and outputs must be finite and in [0, 1]. This diagnostic fit may overfit, particularly on small samples; held-out improvement is not guaranteed and a monotonic map can create ties. It does not make relative scores probabilities, guarantee cross-group comparability, or enforce that data splits are independent; the caller must ensure separation at the task/group level.

## Pinned offline ablation

From the repository root, with the package importable:

```sh
PYTHONPATH=src python benchmarks/ruler_ablation.py
```

This uses `tests/evaluation/fixtures/ruler_ablation.json`: one toy arithmetic task, four hand-written responses and manually specified illustrative gold scores. The baseline is a deliberately simple exact-string reward, `1` only for the complete answer `4`; the other arm passes **prewritten canned scores**, not model-generated judgments, through the real adapter/scorer/calibration pipeline. The output reports MAE, pairwise accuracy, and comparable-pair count. It checks reproducible plumbing and demonstrates a baseline's limitations; its numbers cannot establish judge quality, superiority, or training benefit.

## Opt-in live judge ablation

After explicit authorization for network/paid calls, an OpenAI key in `OPENAI_API_KEY`, and the existing OpenAI provider dependency, run one independently labelled group:

```sh
PYTHONPATH=src python benchmarks/ruler_ablation.py --real \
  --fixture path/to/labelled-group.json --model YOUR_PINNED_MODEL \
  --temperature 0 --max-tokens 1024
```

`--real` is required to enable a call; omitting it runs only the canned offline demo. The required fixture has the same shape as the pinned example: `version`, `task`, `baseline_exact_answer`, and `trajectories` containing `id`, `answer`, and independently supplied `gold` in [0, 1]; `canned_judge` is ignored in live mode. One fixture is one group and the single call sends its task and answers to OpenAI. The output contains raw judge scores in group order, both arms' diagnostics, model/settings, input-file SHA-256 and provider-reported token counts. It does not calculate monetary cost or uncertainty. Invalid gold is rejected before the call; provider or malformed-response errors propagate rather than fabricating scores. Test code exercises the live path with an injected mock provider; **no real model evaluation has been run here**.

A meaningful empirical ablation still requires representative independently labelled tasks, separate calibration and held-out groups, an appropriate baseline, pinned model and source/prompt revision, group membership/order, repeated runs, cost and uncertainty. Run and retain results for multiple groups rather than generalizing from one. The exact-string baseline is intentionally weak and is not a general reward benchmark. Obtain authorization before sending sensitive material or making paid calls.
