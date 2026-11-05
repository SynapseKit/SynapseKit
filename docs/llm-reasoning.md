# Reasoning effort routing

`ReasoningLLM` accepts `effort=EffortLevel.LOW`, `MEDIUM`, or `HIGH` to select a provider-native reasoning control. Set `effort="auto"` to run the built-in, deterministic prompt heuristic; supply `effort_classifier(prompt)` to replace it. The classifier is local and makes no model or network calls.

```python
from synapsekit.llm import EffortLevel, ReasoningLLM

fixed = ReasoningLLM("o3", effort=EffortLevel.MEDIUM)
automatic = ReasoningLLM("o3", effort="auto")
custom = ReasoningLLM(
    "o3",
    effort="auto",
    effort_classifier=lambda prompt: EffortLevel.HIGH if "prove" in prompt else EffortLevel.LOW,
)
```

The classifier returns `low` for short/simple prompts, `medium` for explanatory or longer prompts, and `high` for explicit multi-step, comparison, proof, debugging, design, or complexity work. It is a cheap heuristic, not a quality guarantee. A custom classifier must return an `EffortLevel` or one of the strings `low`, `medium`, or `high`.

`effort` is mutually exclusive with `budget_tokens`; specifying both raises `ValueError`. Omitting `effort` keeps the legacy `budget_tokens` request behavior unchanged. OpenAI, Anthropic, and Gemini are supported. DeepSeek and Qwen do not expose a supported effort knob here and fail clearly if effort routing is requested.

## Provider request controls

| Provider | Native control | Mapping / compatibility |
| --- | --- | --- |
| OpenAI | `reasoning_effort` | Sends the selected `low`, `medium`, or `high` value. Verified against the locked OpenAI SDK 2.38.0 Chat Completions signature. |
| Anthropic | `output_config={"effort": level}` | Uses adaptive thinking (`thinking={"type": "adaptive"}`) with the selected output effort. Verified against the locked Anthropic SDK 0.104.1 types. The model must support adaptive thinking and effort; unsupported models return their API error without a silent fallback. |
| Gemini | `GenerateContentConfig(thinking_config=ThinkingConfig(thinking_budget=...))` | Maps low/medium/high to 512/1024/2048 thinking-budget tokens. Uses the locked `google-genai` 1.75.0 SDK for effort requests; if it is unavailable, an actionable `ImportError` is raised. Models may have different allowed thinking budgets. |

The pre-existing Gemini path remains in use when `effort` is omitted, including its `google-generativeai` request configuration. No effort setting is inferred from `budget_tokens` beyond the existing provider behavior.

## Offline budget/cost proxy fixture

`tests/llm/fixtures/effort_eval.json` is a six-prompt, deterministic fixture. It compares each automatic tier's configured budget proxy with an all-high baseline. Run this snippet from the repository root to reproduce the table:

```python
import json
from pathlib import Path

from synapsekit.llm.effort import GEMINI_THINKING_BUDGETS, EffortLevel, classify_effort

fixture = json.loads(Path("tests/llm/fixtures/effort_eval.json").read_text())
baseline_level = EffortLevel(fixture["baseline_effort"])
baseline_tokens = 0
routed_tokens = 0
for case in fixture["cases"]:
    level = classify_effort(case["prompt"])
    routed = GEMINI_THINKING_BUDGETS[level]
    baseline = GEMINI_THINKING_BUDGETS[baseline_level]
    baseline_tokens += baseline
    routed_tokens += routed
    print(f"{case['id']}: {level.value} {routed} vs {baseline}")
rate = fixture["output_cost_usd_per_1000_budget_tokens"] / 1000
print(f"configured budget proxy: {routed_tokens} vs {baseline_tokens}")
print(f"cost proxy: ${routed_tokens * rate:.5f} vs ${baseline_tokens * rate:.5f}")
```

Observed offline result: two low, two medium, and two high prompts; routed budget proxy 7,168 vs 12,288 all-high tokens (5,120 fewer, 41.67%). At the fixture's illustrative $0.01 per 1,000 budget tokens, the proxy is $0.07168 vs $0.12288 (a $0.05120 decrease). This is a configured-budget/cost estimate—not provider-measured generation or billing. OpenAI and Anthropic effort levels are not fixed token caps, provider pricing differs, and the fixture is not a quality benchmark. It intentionally has no numeric CI savings threshold and makes no live API calls.
