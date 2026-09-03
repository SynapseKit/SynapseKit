# Cache-Augmented Generation (CAG)

Cache-Augmented Generation (CAG) is an alternative to Retrieval-Augmented Generation (RAG) for stable, small-to-medium corpora. Instead of retrieving a few relevant chunks from a vector database and injecting them into the prompt at query-time, CAG loads the *entire* corpus into the model's KV cache (Key-Value cache) once, saves the cache state, and reloads it for subsequent queries.

## CAG vs RAG

| Feature | RAG | CAG |
|---|---|---|
| **Approach** | Retrieve chunks + prompt LLM | Cache full corpus + prompt LLM |
| **Corpus Size** | Unlimited | Constrained by context window (`n_ctx`) |
| **Corpus Stability** | Ideal for volatile/dynamic data | Ideal for stable/static data |
| **Query Latency** | Higher (vector search + overhead) | Extremely low (warm cache reuse) |
| **Setup Cost** | Vector database & embedding pipeline | Initial KV cache generation (cold run) |

## Router Behavior

The `CAGRouter` wraps a standard `Retriever` and decides whether to route the request to the CAG or RAG path using the following decision tree:

1. **Backend Support**: Does the LLM support persistent KV caching? (Currently `LlamaCppLLM` only). If not, route to RAG.
2. **Corpus Stability**: Is the corpus stable (`stable=True`)? If not, route to RAG.
3. **Token Budget**: Does the corpus size exceed the maximum allowed tokens (`max_cag_tokens` or a fraction of `n_ctx`)? If so, route to RAG.
4. **Cache Matching**: Does a valid persisted cache exist matching the corpus fingerprint, model ID, and context size?
   - If yes: Route to CAG (loads cache state).
   - If no: Respects the `on_cache_miss` strategy.
     - `rebuild`: Rebuilds the cache now, then routes to CAG.
     - `rag_fallback` (default): Routes to RAG for this query.

## Limitations

Currently, only the `LlamaCppLLM` backend supports saving and loading model KV cache states via `llama_cpp.Llama.save_state()` / `load_state()`.
Other backends, such as cloud APIs (Anthropic, OpenAI) or client-only local libraries (`vllm`, `mlx`), do not provide direct KV cache serialization mechanisms and will automatically route requests to RAG.

**KV-cache reuse depends on an exact prompt prefix.** On the CAG path the router loads the cached state and returns the corpus as context; the loaded state only accelerates a *subsequent* generation whose prompt begins with the exact same `corpus_text` tokens. If you wrap that context in a system prompt or chat template before generating (as a typical pipeline does), the token prefix diverges and llama.cpp re-evaluates from the point of divergence, so the warm-cache speedup is reduced or lost. To get the full benefit, generate directly against the corpus-prefixed prompt (see `LlamaCppCAGBackend.generate_with_cache`, which the benchmark uses). Wiring generation through the router is planned.

## Usage Example

```python
from synapsekit import LLMConfig
from synapsekit.llm.llamacpp import LlamaCppLLM
from synapsekit.rag.cag_router import CAGRouter
from synapsekit.retrieval.retriever import Retriever

# Initialize retriever (RAG fallback path)
retriever = Retriever(vectorstore=my_vector_store)

# Initialize local Llama.cpp model
llm = LlamaCppLLM(
    config=LLMConfig(model="llama-3.1-8b", api_key="", provider="llamacpp"),
    model_path="/models/llama-3.1-8b-instruct.Q4_K_M.gguf"
)

# Instantiate CAGRouter
router = CAGRouter(
    retriever=retriever,
    llm=llm,
    max_cag_tokens=10000,
    stable=True,
    on_cache_miss="rebuild",
)

# Add texts to the corpus
await router.add(["Stable document text..."])

# Query (routes to CAG if size fits, else falls back to RAG)
results = await router.retrieve("User query")
```

## Benchmarks

A standalone script comparing RAG vs CAG (cold cache) vs CAG (warm cache) latencies is provided:
```bash
python benchmarks/cag_vs_rag_bench.py --model-path /path/to/your/model.gguf --runs 5
```
Running without `--model-path` executes using a stubbed LLM, which is useful for CI verification.

