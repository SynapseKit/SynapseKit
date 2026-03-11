# SynapseKit

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
# Core (pick your LLM)
pip install synapsekit[openai]       # OpenAI
pip install synapsekit[anthropic]    # Anthropic / Claude

# Local & other cloud LLMs
pip install synapsekit[ollama]       # Ollama (local)
pip install synapsekit[cohere]       # Cohere
pip install synapsekit[mistral]      # Mistral AI
pip install synapsekit[gemini]       # Google Gemini
pip install synapsekit[bedrock]      # AWS Bedrock

# Document loaders
pip install synapsekit[pdf]          # PDFLoader
pip install synapsekit[html]         # HTMLLoader
pip install synapsekit[web]          # WebLoader (async URL fetch)

# Vector store backends
pip install synapsekit[chroma]       # ChromaVectorStore
pip install synapsekit[faiss]        # FAISSVectorStore
pip install synapsekit[qdrant]       # QdrantVectorStore
pip install synapsekit[pinecone]     # PineconeVectorStore

# Everything
pip install synapsekit[all]
```

## Loaders

All loaders return `List[Document]`. Documents have `.text` and `.metadata`.

```python
from synapsekit import RAG, PDFLoader, CSVLoader, WebLoader, DirectoryLoader

rag = RAG(model="gpt-4o-mini", api_key="sk-...")

# PDF — one Document per page
docs = PDFLoader("report.pdf").load()
rag.add_documents(docs)

# CSV — one Document per row
docs = CSVLoader("data.csv", text_column="content").load()

# Fetch a URL
docs = await WebLoader("https://example.com").load()

# Entire directory (auto-detects .txt, .pdf, .csv, .json, .html)
docs = DirectoryLoader("./my_docs/").load()

rag.add_documents(docs)
answer = rag.ask_sync("What did I just load?")
```

## Output parsers

```python
from synapsekit import JSONParser, ListParser, PydanticParser
from pydantic import BaseModel

# Extract JSON from LLM output
parser = JSONParser()
data = parser.parse('Here is the result: {"name": "Alice", "age": 30}')

# Parse bullet / numbered lists
items = ListParser().parse("- item one\n- item two\n- item three")

# Parse into a Pydantic model
class Person(BaseModel):
    name: str
    age: int

person = PydanticParser(Person).parse('{"name": "Bob", "age": 25}')
```

## Prompt templates

```python
from synapsekit import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate

# f-string style
pt = PromptTemplate("Summarise this in {language}: {text}")
prompt = pt.format(language="French", text="...")

# Chat messages
cpt = ChatPromptTemplate([
    {"role": "system", "content": "You are a {persona}."},
    {"role": "user",   "content": "Tell me about {topic}."},
])
messages = cpt.format_messages(persona="chef", topic="pasta")

# Few-shot
fsp = FewShotPromptTemplate(
    examples=[{"q": "2+2", "a": "4"}],
    example_template="Q: {q}\nA: {a}",
    suffix="Q: {question}\nA:",
)
prompt = fsp.format(question="3+3")
```

## Vector store backends

```python
from synapsekit import SynapsekitEmbeddings, Retriever
from synapsekit.retrieval.chroma import ChromaVectorStore
from synapsekit.retrieval.faiss import FAISSVectorStore

embeddings = SynapsekitEmbeddings()

# Chroma
store = ChromaVectorStore(embeddings, collection_name="my_docs")

# FAISS
store = FAISSVectorStore(embeddings)

# All backends share the same interface
retriever = Retriever(store)
await retriever.add(["chunk one", "chunk two"])
results = await store.search("my query", top_k=5)
```

## LLM providers

```python
from synapsekit import RAG

# OpenAI
rag = RAG(model="gpt-4o-mini", api_key="sk-...")

# Anthropic
rag = RAG(model="claude-sonnet-4-6", api_key="sk-ant-...")

# Ollama (local, no api_key needed)
rag = RAG(model="llama3", api_key="", provider="ollama")

# Cohere
rag = RAG(model="command-r-plus", api_key="...", provider="cohere")

# Mistral
rag = RAG(model="mistral-large-latest", api_key="...", provider="mistral")

# Google Gemini
rag = RAG(model="gemini-1.5-pro", api_key="...", provider="gemini")

# AWS Bedrock
rag = RAG(model="anthropic.claude-3-sonnet-20240229-v1:0", api_key="env", provider="bedrock")
```

## vs LangChain / LlamaIndex

| | LangChain | LlamaIndex | **SynapseKit** |
|---|---|---|---|
| Streaming-native | ✗ | ✗ | **✓** |
| Async-native | Partial | Partial | **✓** |
| Install size | ~500MB | ~400MB | **~50MB** |
| No magic / no callbacks | ✗ | ✗ | **✓** |
| Hard dependencies | Many | Many | **2** (`numpy` + `rank-bm25`) |

## License

MIT
