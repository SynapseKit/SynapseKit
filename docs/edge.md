# Edge SynapseKit

Edge SynapseKit runs local models first and falls back to a cloud model only
when an explicit policy allows it. This is intended for privacy-sensitive
agents that need to work on laptops, edge servers, and offline environments.

## Install

Install only the backends you need:

```bash
pip install "synapsekit[edge]"     # ONNX Runtime + SQLite-vec
pip install "synapsekit[onnx]"     # ONNX embeddings only
pip install "synapsekit[mlx]"      # Apple Silicon
```

The llama.cpp backend is **not** bundled in any extra and must be installed
directly:

```bash
pip install llama-cpp-python
```

It is excluded because its `diskcache` dependency has no patched release for
CVE-2025-69872. Installing it is your call; SynapseKit will not pull it into
your dependency tree on your behalf.

The core package does not import llama.cpp, ONNX Runtime, or MLX unless you
instantiate those providers.

## Local-first runtime

```python
from synapsekit import EdgeRuntime, FallbackPolicy, LLMConfig
from synapsekit.llm.llamacpp import LlamaCppLLM

local = LlamaCppLLM(
    LLMConfig(model="llama-3.2-3b", api_key="", provider="llamacpp"),
    model_path="/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    n_ctx=8192,
)

runtime = EdgeRuntime(
    local_llm=local,
    fallback=FallbackPolicy(),
)
```

With no cloud model and no fallback gates, all calls stay local. If the local
model cannot handle a request, SynapseKit raises a clear edge fallback error
instead of silently sending data to a remote provider.

## Cloud fallback

Fallback is default-deny. Configure only the routes your application accepts:

```python
from synapsekit import EdgeRuntime, FallbackPolicy
from synapsekit.llm.anthropic import AnthropicLLM

runtime = EdgeRuntime(
    local_llm=local,
    cloud_llm=AnthropicLLM(...),
    fallback=FallbackPolicy(
        if_context_exceeds=8192,
        if_tool_unsupported_locally=True,
        if_user_opts_in=True,
        require_pii_redaction_before_fallback=True,
    ),
)
```

Supported fallback reasons:

- context exceeds the configured local limit
- tool calling is not supported by the local provider
- the caller explicitly passes `allow_cloud_fallback=True`
- local model errors, only when `fallback_on_local_error=True`

The last route is observable:

```python
answer = await runtime.generate(prompt)
print(runtime.last_route, runtime.last_fallback_reason)
```

## PII redaction

When `require_pii_redaction_before_fallback=True`, prompts and chat messages
are passed through `PIIRedactor` before the cloud model receives them. Email
addresses, phone numbers, SSNs, and other configured PII patterns are replaced
with stable placeholders.

```python
answer = await runtime.generate("Email alice@example.com", allow_cloud_fallback=True)
print(runtime.last_redaction.pii_types_found)
```

Use `mode="redact"` on `PIIRedactor` for irreversible redaction, or the default
masking mode when your application needs a reversible mapping.

## Local embeddings and SQLite-vec

For edge RAG, pair `ONNXEmbeddings` with `SQLiteVecStore`:

```python
from synapsekit import ONNXEmbeddings, SQLiteVecStore

embeddings = ONNXEmbeddings("/models/all-MiniLM-L6-v2.onnx")
store = SQLiteVecStore(embeddings, db_path="edge_vectors.db")
await store.add(["PHI policy document"], metadata=[{"source": "policy"}])
```

This keeps vectors and source text in a local SQLite database and avoids a
network vector store in regulated deployments.

## CLI

List known edge models:

```bash
synapsekit edge list
```

Pull a known model into the local cache:

```bash
synapsekit edge pull llama-3.2-3b
```

Quantize a local model with llama.cpp:

```bash
synapsekit edge quantize model-f16.gguf model-q4.gguf --quantization Q4_K_M
```

## HIPAA-safe agent pattern

1. Run the primary LLM through `LlamaCppLLM`, `GPT4AllLLM`, `OllamaLLM`, or
   `MLXLLM`.
2. Use `SQLiteVecStore` for retrieval memory.
3. Set fallback gates narrowly, preferably only for context overflow or
   explicit user opt-in.
4. Keep `require_pii_redaction_before_fallback=True`.
5. Log `runtime.last_route` and `runtime.last_fallback_reason` for audits.

Mobile Swift/Kotlin wrappers are intentionally left for a future PR. The
runtime added here is the Python policy and provider foundation those bindings
can call into later.

## Local vs. cloud tradeoffs

`benchmarks/edge_local_vs_cloud_bench.py` runs a 200-task suite (extraction,
classification, structured JSON, and date formatting; see
`examples/edge_task_suite.py`) through a local GGUF model and a cloud model
directly, and reports accuracy, p50/p95 latency, and USD cost per category.
This is a head-to-head model comparison, not a test of `EdgeRuntime`'s
routing -- it tells you which task families are safe to keep on-device before
you set `FallbackPolicy` gates.

Reproduce it locally (needs a GGUF file and `ANTHROPIC_API_KEY`; CI only runs
the harness-correctness checks in `test_edge_local_vs_cloud_bench.py` against
stub models, since it has neither):

```bash
python benchmarks/edge_local_vs_cloud_bench.py \
    --model-path /models/Llama-3.2-3B-Instruct-Q4_K_M.gguf --n-tasks 200
```

<!--
Fill in with real numbers from a Llama-3.2-3B-Instruct-Q4_K_M run once
available -- this table is the "document tradeoffs" acceptance criterion for
issue #736 and should not stay empty.
-->

| model | accuracy | p50 ms | p95 ms | cost USD |
|---|---|---|---|---|
| local (Llama 3.2 3B Q4_K_M) | _pending_ | _pending_ | _pending_ | 0.0000 |
| cloud (`claude-haiku-4-5-20251001`) | _pending_ | _pending_ | _pending_ | _pending_ |

| category | local | cloud | gap |
|---|---|---|---|
| extraction | _pending_ | _pending_ | _pending_ |
| classification | _pending_ | _pending_ | _pending_ |
| json | _pending_ | _pending_ | _pending_ |
| format | _pending_ | _pending_ | _pending_ |
