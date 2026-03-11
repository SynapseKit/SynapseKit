lets # SynapseKit

Lightweight, async-first RAG framework. Streaming-native, minimal dependencies.

```python
from synapsekit import RAG

rag = RAG(model="gpt-4o-mini", api_key="sk-...")
rag.add("Your document text here")

# Streaming
async for token in rag.stream("What is the main topic?"):
    print(token, end="", flush=True)

# Non-streaming
answer = await rag.ask("What is the main topic?")

# Sync (notebooks/scripts)
answer = rag.ask_sync("What is the main topic?")
```

## Install

```bash
pip install synapsekit[openai]      # OpenAI
pip install synapsekit[anthropic]   # Anthropic
[ai-letters-05-substack.html](../../engineersofai/engineersofai-ui/blog-drafts/ai-letters-05-substack.html)pip install synapsekit[all]         # Everything
```

## vs LangChain / LlamaIndex

- **No chains, no callbacks, no magic** — just async functions and plain classes
- **Streaming is first-class**, not bolted on
- **Two hard dependencies**: `numpy` + `chunkrank`
- **Python 3.10+**

## License

MIT
