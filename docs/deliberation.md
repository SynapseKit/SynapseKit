# Parallel deliberation

`deliberate` concurrently requests N independent, short JSON responses and uses the model's self-reported confidence as a vote weight. It groups **exact**, whitespace-trimmed answers, selects the greatest sum of confidences (ties use lowest trace index), and returns that group's mean confidence. The confidence is not calibrated probability, and semantically equivalent but differently worded answers do not agree. A response must provide a nonempty string `answer` and finite numeric `confidence` in [0, 1]. Failures and malformed responses are retained as structured traces; no valid result yields `answer=None`.

With `early_exit_confidence`, at least two identical completed answers must have a mean confidence at or above the threshold and pending work must remain. In addition, the winning weighted score must be **strictly greater** than the highest competing score plus the number of pending tasks: each pending task could contribute a maximum weight of 1 to the same competing answer, including an answer not yet observed. Ties do not permit early exit. Every task in an `asyncio.wait` completion batch is scored before this check; pending tasks are then cancelled and drained. Cancellation-suppressing adapters may still return completions, which count in usage and in the returned confidence; the margin preserves the winner under the bounded vote contract. This guarantees only that the winner cannot be overturned by those pending votes, not that the answer is correct or that the full-run confidence would meet the threshold. Cancellation cannot guarantee a remote service stops billing. An adapter that ignores cancellation can delay cleanup indefinitely.

## Bounded adapter contract

`BoundedAdapter.complete(prompt, *, max_output_tokens)` must enforce the cap **at generation**, including hidden reasoning output, not truncate text afterwards. It must not make unaccounted retries, and must return `Completion(text, input_tokens, output_tokens, cost)` with per-request usage/cost where known. Unknown values must be `None`. The orchestrator reserves `n * max_output_tokens` before launching; a smaller `total_output_budget` is rejected. This bounds output only, conditional on the adapter/provider honoring the cap; it is not a monetary or input-token budget. Reported output beyond the cap invalidates the trace. The protocol cannot verify hidden tokens or prevent a dishonest adapter. `BaseLLM.generate` accepts `**kw`, but providers vary in whether they honor it and shared cumulative token counters do not provide reliable per-request usage, especially on cancellation. Do not wrap arbitrary `BaseLLM` as a bounded adapter without verifying its implementation.

`OpenAILLM` (a separate provider, not a bounded adapter) disables OpenAI SDK retries to prevent duplicate retry layers. `LLMConfig.max_retries` controls the framework retry wrapper for `generate`, `generate_with_messages`, and `call_with_tools`: retryable failures cause at most `max_retries + 1` attempts. Direct `stream` and `stream_with_messages` calls bypass that wrapper and make one attempt.

`OpenAIChatAdapter` supports an injected async OpenAI chat client with `max_retries=0` and explicitly sends `max_completion_tokens`; use only models that honor this limit for all generated tokens. Example (network calls may incur charges):

```python
from openai import AsyncOpenAI
from synapsekit.llm.deliberation import OpenAIChatAdapter, deliberate

client = AsyncOpenAI(max_retries=0)
adapter = OpenAIChatAdapter(client, model="YOUR_SUPPORTED_MODEL")
try:
    result = await deliberate(adapter, "What is 2 + 2?", n=3,
                              max_output_tokens=128, total_output_budget=384,
                              early_exit_confidence=0.9)
    print(result.answer, result.confidence, result.traces, result.total_cost)
finally:
    await client.close()
```

Adapter exceptions are represented by a generic error message in traces; provider exception text is not included. Successful and invalid completions may still retain raw model output, which can contain sensitive content.

Optional per-token input and output rates provide a **cost estimate**, not an invoice; cached-token discounts, taxes, other fees or different rate tiers are not included. Leave rates unset if such accounting would mislead. Every launched call, including errors, cancelled requests, and late completions, must be considered: any missing usage makes the corresponding total `None`, not zero. Provider error/cancellation may hide billable work. No separate judge is used, so no judge cost is incurred. Parent cancellation drains child tasks then propagates cancellation; no result/accounting is returned in that case.

## Paired evaluation

`benchmarks/deliberation_eval.py` compares N concise calls with one longer, capped call explicitly prompted to explain its reasoning in a JSON `reasoning` field. It scores both on the same fixture using the same validated `answer` and `confidence` fields and exact-match scoring, without a model judge. It retains the baseline's raw response and status for inspection; the prompt requests reasoning but cannot force a model to provide it. It reports per-case answers, correctness, token use and estimated cost, with unknown totals preserved. To run, provide an importable `module:factory` returning a configured bounded adapter:

```sh
python benchmarks/deliberation_eval.py --adapter my_adapter:create --n 3 --short-cap 128 --long-cap 1024
```

The CLI owns the factory-created adapter and awaits its optional async `aclose()` in the same event loop after success, failure, or cancellation. Adapters without `aclose` remain supported. `OpenAIChatAdapter.aclose()` closes its injected client, so do not share that client with work that outlives the CLI. Calling `evaluate` or `deliberate` directly does not close a caller-supplied adapter; its owner must do so. Cleanup exceptions propagate and can replace an evaluation exception.

The default disables early exit for a complete paired comparison. To evaluate the optional early-exit policy, add `--early-exit-confidence 0.9` (or pass `early_exit_confidence` to `evaluate`). The threshold must be finite and in [0, 1]. Numeric settings and all fixture entries are validated before the CLI loads the adapter factory or makes requests; each fixture entry needs string `prompt` and `answer` fields.

Reports include each case's early-exit flag and launched and cancelled trace counts, plus totals for these counts and the configured threshold. A cancelled trace does not prove that remote generation or billing stopped; a cancellation-suppressing completion is included in usage. If any launched call lacks cost or token usage, that per-case and aggregate total is `null`, not a saving or zero. Compare paired exact-match results and known usage only; this is not a separate no-early-exit run.

The fixture contains small deterministic-answer tasks, not measured model outputs. A real evaluation requires explicit authorization and an adapter whose accounting and caps are verified; repeated runs, task diversity, pricing assumptions, and uncertainty should be reported. No real-model run or improvement in accuracy/cost is claimed here. Fake-adapter tests verify orchestration only, not empirical model quality.
