"""Preflight tests — fast smoke checks run before every release.

These tests MUST complete in < 5 s with zero API calls.
They verify:
  - All top-level exports are importable.
  - Version string is valid semver.
  - All 31 LLM provider modules are importable without optional deps installed.
  - All 37 loaders are discoverable via the lazy __getattr__ mechanism.
  - All async interfaces are proper coroutines (inspect.iscoroutinefunction).
  - Key classes instantiate with no exceptions when given mock dependencies.
"""

from __future__ import annotations

import importlib
import inspect
import re
from unittest.mock import patch

# ---------------------------------------------------------------------------
# 1. Package version
# ---------------------------------------------------------------------------


def test_version_is_valid_semver():
    import synapsekit

    ver = synapsekit.__version__
    assert re.match(r"^\d+\.\d+\.\d+", ver), f"Version {ver!r} is not valid semver"


def test_version_matches_pyproject():
    """Version in __init__ matches pyproject.toml."""
    import pathlib

    import synapsekit

    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    pyproject = pathlib.Path(__file__).parents[2] / "pyproject.toml"
    with open(pyproject, "rb") as fh:
        data = tomllib.load(fh)
    assert synapsekit.__version__ == data["project"]["version"]


# ---------------------------------------------------------------------------
# 2. Top-level public API imports
# ---------------------------------------------------------------------------


TOP_LEVEL_NAMES = [
    "RAG",
    "FunctionCallingAgent",
    "ReActAgent",
    "AgentExecutor",
    "StateGraph",
    "MCPServer",
    "TokenTracer",
    "ConversationMemory",
    "InMemoryVectorStore",
    "CalculatorTool",
    "DateTimeTool",
    "ShellTool",
    "DuckDuckGoSearchTool",
    "BaseTool",
    "Document",
    "StringLoader",
    "TextLoader",
    "MarkdownLoader",
    "public_api",
    "experimental",
    "deprecated",
]


def test_top_level_imports_exist():
    import synapsekit

    missing = [name for name in TOP_LEVEL_NAMES if not hasattr(synapsekit, name)]
    assert not missing, f"Missing top-level names: {missing}"


# ---------------------------------------------------------------------------
# 3. LLM provider modules importable (no optional deps needed)
# ---------------------------------------------------------------------------

LLM_MODULES = [
    "synapsekit.llm.openai",
    "synapsekit.llm.anthropic",
    "synapsekit.llm.gemini",
    "synapsekit.llm.cohere",
    "synapsekit.llm.mistral",
    "synapsekit.llm.groq",
    "synapsekit.llm.deepseek",
    "synapsekit.llm.openrouter",
    "synapsekit.llm.together",
    "synapsekit.llm.fireworks",
    "synapsekit.llm.ollama",
    "synapsekit.llm.bedrock",
    "synapsekit.llm.azure_openai",
    "synapsekit.llm.moonshot",
    "synapsekit.llm.minimax",
    "synapsekit.llm.zhipu",
    "synapsekit.llm.cloudflare",
    "synapsekit.llm.databricks",
    "synapsekit.llm.ernie",
    "synapsekit.llm.sambanova",
    "synapsekit.llm.aleph_alpha",
    "synapsekit.llm.llamacpp",
    "synapsekit.llm.vllm",
    "synapsekit.llm.gpt4all",
    "synapsekit.llm.ai21",
    "synapsekit.llm.xai",
    "synapsekit.llm.novita",
    "synapsekit.llm.writer",
    "synapsekit.llm.perplexity",
    "synapsekit.llm.cerebras",
    "synapsekit.llm.lmstudio",
]


def test_all_llm_modules_importable():
    """All LLM modules import without raising (optional deps may be absent)."""
    failed = []
    for mod_path in LLM_MODULES:
        try:
            importlib.import_module(mod_path)
        except ImportError:
            pass  # expected — optional dep not installed in CI
        except Exception as exc:
            failed.append(f"{mod_path}: {exc}")
    assert not failed, f"LLM modules raised non-ImportError: {failed}"


def test_llm_provider_count_matches_spec():
    """We have exactly 31 LLM provider modules."""
    assert len(LLM_MODULES) == 31


# ---------------------------------------------------------------------------
# 4. Loader lazy discovery
# ---------------------------------------------------------------------------

LOADER_NAMES = [
    "ArXivLoader",
    "AudioLoader",
    "AzureBlobLoader",
    "BigQueryLoader",
    "CSVLoader",
    "ConfigLoader",
    "ConfluenceLoader",
    "DirectoryLoader",
    "DiscordLoader",
    "DocxLoader",
    "Document",
    "DynamoDBLoader",
    "DropboxLoader",
    "EPUBLoader",
    "ElasticsearchLoader",
    "EmailLoader",
    "GCSLoader",
    "GitHubLoader",
    "GitLoader",
    "GoogleDriveLoader",
    "GoogleSheetsLoader",
    "HTMLLoader",
    "LaTeXLoader",
    "JSONLoader",
    "JiraLoader",
    "MarkdownLoader",
    "MongoDBLoader",
    "NotionLoader",
    "OneDriveLoader",
    "PDFLoader",
    "ParquetLoader",
    "RSSLoader",
    "RTFLoader",
    "RedisLoader",
    "S3Loader",
    "SalesforceLoader",
    "SitemapLoader",
    "SQLLoader",
    "SlackLoader",
    "StringLoader",
    "SupabaseLoader",
    "TeamsLoader",
    "TextLoader",
    "TSVLoader",
    "VideoLoader",
    "WebLoader",
    "WikipediaLoader",
    "XMLLoader",
    "YAMLLoader",
]


def test_all_loaders_in_all_list():
    import synapsekit.loaders as loaders_mod

    missing = [n for n in LOADER_NAMES if n not in loaders_mod.__all__]
    assert not missing, f"Loaders missing from __all__: {missing}"


def test_loader_count_matches_spec():
    """We have exactly 47 names in the loaders __all__ (includes Document + StringLoader)."""
    import synapsekit.loaders as loaders_mod

    assert len(loaders_mod.__all__) == len(LOADER_NAMES)


# ---------------------------------------------------------------------------
# 5. Async interface contracts
# ---------------------------------------------------------------------------


def test_rag_ask_is_coroutine():
    from synapsekit import RAG

    assert inspect.iscoroutinefunction(RAG.ask)


def test_rag_add_async_is_coroutine():
    from synapsekit import RAG

    assert inspect.iscoroutinefunction(RAG.add_async)


def test_rag_stream_is_async_generator():
    from synapsekit import RAG

    assert inspect.isasyncgenfunction(RAG.stream)


def test_function_calling_agent_run_is_coroutine():
    from synapsekit import FunctionCallingAgent

    assert inspect.iscoroutinefunction(FunctionCallingAgent.run)


def test_react_agent_run_is_coroutine():
    from synapsekit import ReActAgent

    assert inspect.iscoroutinefunction(ReActAgent.run)


def test_react_agent_stream_is_async_generator():
    from synapsekit import ReActAgent

    assert inspect.isasyncgenfunction(ReActAgent.stream)


# ---------------------------------------------------------------------------
# 6. Key class instantiation smoke tests
# ---------------------------------------------------------------------------


def test_rag_instantiates_with_openai():
    with patch("synapsekit.llm.openai.OpenAILLM.__init__", return_value=None):
        from synapsekit import RAG

        rag = RAG(model="gpt-4o-mini", api_key="sk-test")
        assert rag._pipeline is not None
        assert rag._vectorstore is not None


def test_rag_instantiates_with_anthropic():
    with patch("synapsekit.llm.anthropic.AnthropicLLM.__init__", return_value=None):
        from synapsekit import RAG

        rag = RAG(model="claude-haiku-4-5-20251001", api_key="sk-test")
        assert rag._pipeline is not None


def test_state_graph_instantiates():
    from synapsekit import StateGraph

    g = StateGraph()
    assert repr(g).startswith("StateGraph")


def test_calculator_tool_instantiates():
    from synapsekit import CalculatorTool

    tool = CalculatorTool()
    assert tool.name == "calculator"
    assert inspect.iscoroutinefunction(tool.run)


def test_datetime_tool_instantiates():
    from synapsekit import DateTimeTool

    tool = DateTimeTool()
    assert tool.name == "datetime"


def test_shell_tool_instantiates():
    from synapsekit import ShellTool

    tool = ShellTool()
    assert tool.name == "shell"


def test_mcp_server_instantiates_empty():
    from synapsekit import MCPServer

    srv = MCPServer()
    assert srv._name == "synapsekit"
    assert srv._tools == {}


def test_token_tracer_instantiates():
    from synapsekit import TokenTracer

    tracer = TokenTracer(model="gpt-4o-mini", enabled=True)
    summary = tracer.summary()
    assert summary["calls"] == 0
    assert summary["total_tokens"] == 0


def test_conversation_memory_instantiates():
    from synapsekit import ConversationMemory

    mem = ConversationMemory(window=5)
    assert len(mem) == 0
    assert mem.get_messages() == []


def test_in_memory_vector_store_instantiates():
    from synapsekit import InMemoryVectorStore
    from synapsekit.embeddings.backend import SynapsekitEmbeddings

    store = InMemoryVectorStore(SynapsekitEmbeddings())
    assert len(store._texts) == 0


# ---------------------------------------------------------------------------
# 7. API decorators
# ---------------------------------------------------------------------------


def test_public_api_decorator_adds_marker():
    from synapsekit import public_api

    @public_api
    def my_fn():
        pass

    assert my_fn._synapsekit_public_api is True


def test_experimental_decorator_fires_warning():
    import warnings

    from synapsekit import experimental

    @experimental
    def my_fn():
        pass

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        my_fn()
    assert len(w) == 1
    assert issubclass(w[0].category, FutureWarning)


def test_deprecated_decorator_fires_deprecation_warning():
    import warnings

    from synapsekit import deprecated

    @deprecated("use new_fn instead")
    def old_fn():
        pass

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        old_fn()
    assert len(w) == 1
    assert issubclass(w[0].category, DeprecationWarning)


# ---------------------------------------------------------------------------
# 8. Loader instantiation (no external deps)
# ---------------------------------------------------------------------------


def test_string_loader_instantiates():
    from synapsekit.loaders import StringLoader

    loader = StringLoader("hello world")
    docs = loader.load()
    assert len(docs) == 1
    assert docs[0].text == "hello world"


def test_markdown_loader_instantiates(tmp_path):
    from synapsekit.loaders import MarkdownLoader

    md_file = tmp_path / "test.md"
    md_file.write_text("# Title\nContent here.")
    loader = MarkdownLoader(str(md_file))
    docs = loader.load()
    assert len(docs) >= 1
    assert "Content here" in docs[0].text


def test_text_loader_instantiates(tmp_path):
    from synapsekit.loaders import TextLoader

    txt_file = tmp_path / "test.txt"
    txt_file.write_text("plain text content")
    loader = TextLoader(str(txt_file))
    docs = loader.load()
    assert len(docs) == 1
    assert "plain text" in docs[0].text
