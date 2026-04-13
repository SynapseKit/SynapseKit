# Changelog

All notable changes to SynapseKit are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
SynapseKit uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.5.5] — 2026-04-13

### Added

- **`S3Loader`** — load files from Amazon S3 buckets into Documents; supports text, binary fallback, and rich file extraction (PDF, DOCX, XLSX, PPTX, CSV, JSON, HTML) via existing loaders; prefix filtering, extension filtering, `max_files` limit; credential chain (explicit keys, session tokens, or ambient IAM role); async `aload()` via executor; `pip install synapsekit[s3]`; closes #522
- **`AzureBlobLoader`** — load blobs from Azure Blob Storage containers; supports both connection-string and account URL + credential auth; same extraction chain as S3Loader; prefix filtering, `max_files`; `pip install synapsekit[azure]`; closes #520
- **`MongoDBLoader`** — load documents from a MongoDB collection as Documents; configurable `text_fields` and `metadata_fields`; optional `query_filter`; projection builder fetches only requested fields; defensive copy of filter dict; sync + async; `pip install synapsekit[mongodb]`; closes #519
- **`DropboxLoader`** — load files from a Dropbox folder; supports 20+ text/code extensions; pagination via cursor; `limit` stops fetching early; download-error skipping; `pip install synapsekit[dropbox]`; closes #517
- **`EvalDataset` / `EvalRecord`** — filterable, exportable collection of eval result records; `filter_score(min_score, max_score)` narrows to weak/strong cases; `export()` writes fine-tuning datasets in OpenAI, Anthropic, Together, JSONL, and DPO pair formats; `from_snapshot()` loads from existing EvalCI snapshots
- **`FineTuner`** — orchestrates fine-tuning jobs against OpenAI and Together AI; injectable adapter pattern for extensibility; `submit()`, `status()`, `wait()` (polls until terminal state with configurable timeout/interval); `FineTuneJob` dataclass tracks id, provider, status, model_id, error
- **`@eval_case(capture_io=True)`** — opt-in capture of `input`, `output`, and `ideal` fields in eval case results; required for `EvalDataset.export()`
- **`synapsekit eval` CLI** — `report <snapshot>` summarises scores and weak cases; `export <snapshot> --format openai --output data.jsonl` writes fine-tune dataset; `compare <baseline> <current>` runs regression comparison
- **`synapsekit finetune` CLI** — `submit <dataset> --provider openai --base-model gpt-4o-mini`; `status <job_id>`; `wait <job_id>` blocks until completion
- **Recursive Subgraph Support** — allow a `StateGraph` to be passed to `subgraph_node()`, enabling self-referential / recursive workflows; implements a `max_recursion_depth` guard (default 10) to prevent infinite loops; tracks depth via internal `__recursion_depth__` state key; adds `RecursionDepthError` to handle limit breaches; lazy compilation supports definition-time self-referencing.
- **Discord community link** — added Discord server link to README community section.
- **`LMStudioLLM`** — local model provider via LM Studio's OpenAI-compatible API; connects to a running LM Studio server (default `http://localhost:1234/v1`); supports streaming, tool calling, and custom `base_url` via constructor kwarg; no API key required; `pip install synapsekit[lmstudio]`; closes #176
- **`MCPServer` SSE transport + package refactor** — `MCPServer` now lives in `synapsekit.mcp.server` package; adds `run_sse(host, port, api_key)` for HTTP/SSE MCP serving with optional Bearer auth; backwards-compatible with existing `MCPServer(tools=[...])`, `MCPServer(rag)`, and `MCPServer(agent)` usage
- **`LaTeXLoader`** — load `.tex` files as plain text; strips commands, environments, inline/display math, and comments via regex; captures section/subsection titles into metadata; no external deps required
- **`TSVLoader`** — load tab-separated files one Document per row; configurable `text_column` to extract a specific column as text; remaining columns become metadata; skips empty rows; async `aload()`
- **`RTFLoader`** — load RTF files as plain text via `striprtf`; handles malformed RTF gracefully; `pip install synapsekit[rtf]`
- **`EPUBLoader`** — load EPUB files chapter-by-chapter; extracts title, author, and chapter name into metadata; strips HTML tags safely; `pip install synapsekit[epub]`
- **`ConfigLoader`** — load `.env`, `.ini`, `.cfg`, `.toml`, and environment-specific dotfiles (`.env.local`, `.env.staging`, `.env.production`) into Documents; redacts sensitive keys (password, secret, token, api_key, auth) automatically; one Document per INI section; Python 3.11+ uses stdlib `tomllib`, falls back to `tomli` on older versions
- **`OneDriveLoader`** — load files from OneDrive and SharePoint via Microsoft Graph API; folder traversal with optional recursion; extension filtering; `max_files` cap; extracts PDF, DOCX, XLSX, PPTX, CSV, JSON, HTML via existing loaders; async `aload()`; uses stdlib HTTP (no external SDK required)

### Fixed

- **`DropboxLoader` SDK compatibility** — original implementation called `.get()` on Dropbox SDK entry objects (`FileMetadata`, `FolderMetadata`), which are Stone-generated Python classes, not dicts; fixed with `_normalise_entry()` static method that converts SDK objects to canonical dicts via attribute access while passing test-mock dicts through unchanged
- **`LMStudioLLM` `base_url`** — `LLMConfig` has no `base_url` field; passing it via `LLMConfig(base_url=...)` would raise `TypeError` before `LMStudioLLM.__init__` ran. Fixed by adding `base_url: str | None = None` as a keyword argument to `LMStudioLLM.__init__` directly (mirrors the `XaiLLM` / `NovitaLLM` pattern). Custom server usage: `LMStudioLLM(config, base_url="http://192.168.1.10:1234/v1")`
- **`LMStudioLLM` stream stability** — removed `stream_options={"include_usage": True}` which caused API errors on older LM Studio builds; usage tracking now reads `chunk.usage` defensively via `getattr` so it still captures tokens when the server returns them
- **`ConfigLoader` rejects `.env.local` / `.env.staging`** — `os.path.splitext(".env.local")` returns `('.env', '.local')` making `ext = '.local'` which fell through to `ValueError: Unsupported config file type`. Fixed by detecting any file whose basename starts with `.env` and treating it as the env format regardless of secondary extension
- **`RTFLoader` default encoding** — changed default from `"utf-8"` to `"latin-1"` (Windows-1252 superset) since real-world RTF files from Office/WordPad are almost universally Windows-encoded, not UTF-8

---

## [1.5.3] — 2026-04-11

### Added

- **`TeamsLoader`** — load messages from Microsoft Teams channels via the Microsoft Graph API; automatic pagination via `@odata.nextLink`; HTML-to-plain-text conversion (strips tags, decodes entities); exponential backoff retry for 429 and 5xx responses; graceful handling of missing author/timestamp/body fields; sync `load()` and async `aload()`; `pip install synapsekit[teams]`; closes #51
- **`CodeInterpreterTool`** — execute Python code in an isolated subprocess and capture stdout, stderr, generated files, matplotlib plot artifacts, and pandas dataframe reprs; configurable timeout (default 5s) and memory limit (default 256 MB, enforced via `RLIMIT_AS` on Linux); workspace isolation via `tempfile.TemporaryDirectory`; structured JSON output; closes #216

### Fixed

- **`ShellTool` Windows compatibility** — use `asyncio.create_subprocess_shell()` on Windows so shell builtins (`echo`, `dir`, etc.) work correctly; keep `create_subprocess_exec()` on Unix; closes #502
- **pytest Windows warnings** — suppress harmless `PytestUnraisableExceptionWarning` from asyncio proactor transport GC cleanup on Windows

---

## [1.5.2] — 2026-04-10

### Added

- **`JSONSplitter`** — JSON-aware chunking; splits arrays by element and objects by top-level key; item-level overlap preserves valid JSON structure (character-level overlap would produce invalid JSON fragments); closes #501
- **EvalCI GitHub Action** — live on [GitHub Marketplace](https://github.com/marketplace/actions/evalci-by-synapsekit); LLM quality gates on every PR, zero infrastructure, 2-minute setup

### Fixed

- **Async `@eval_case` not awaited** — decorated async functions now correctly preserve `inspect.iscoroutinefunction()` identity; the CLI runner was skipping `asyncio.run()` for async cases, passing a raw coroutine to `float()` and raising `TypeError`; regression tests added

---

## [1.5.1] — 2026-04-09

### Added

- **WeatherTool tests** — full test suite covering current weather, forecast, missing API key, empty results, network errors, schema validation; closes #383
- **`GitLoader`** — load files from any Git repository (local path or remote URL) at a specific revision; glob pattern filtering; metadata includes path, commit hash, author, date; `pip install synapsekit[git]`
- **`GoogleSheetsLoader`** — load rows from a Google Sheets spreadsheet as Documents; service account auth via credentials file; auto-detects first sheet if none specified; header-based row-to-text formatting; `pip install synapsekit[gsheets]`
- **`JiraLoader`** — load Jira issues via JQL queries; full Atlassian Document Format (ADF) parsing; pagination; rate-limit retry; async `aload()` via httpx; optional `limit`; `pip install synapsekit[jira]`
- **`SupabaseLoader`** — load rows from a Supabase table as Documents; configurable text/metadata columns; env var auth (`SUPABASE_URL`, `SUPABASE_KEY`); `pip install synapsekit[supabase]`

### Security

- **SQL injection** — `SQLSchemaInspectionTool` now validates table names against `^[A-Za-z0-9_]+$` before interpolating into `PRAGMA table_info()`; closes #494
- **Shell injection** — `ShellTool` switched from `create_subprocess_shell` to `create_subprocess_exec` with `shlex.split()`; allowlist now checked against the actual binary (`argv[0]`) instead of a whitespace split; closes #495
- **Path traversal** — `FileReadTool` and `FileWriteTool` now accept an optional `base_dir`; all paths are resolved with `Path.resolve()` and checked to be within the sandbox before I/O; closes #496
- **TOCTOU in VideoLoader** — replaced `tempfile.mktemp()` with `tempfile.NamedTemporaryFile(delete=False)` to eliminate the race window; closes #497
- **SSRF** — `WebLoader` and `WebScraperTool` now validate URL scheme (must be `http`/`https`) and block requests to private/internal IP ranges (RFC 1918, loopback, link-local, IPv6 ULA); closes #498
- **ReDoS** — `WebScraperTool` limits CSS selector length to 200 characters; closes #498

---

## [1.5.0] — 2026-04-07

### Added

- **RSSLoader** — load articles from RSS/Atom feeds as Documents; content/summary fallback; metadata includes title, published, link, author; async `aload()`; `pip install synapsekit[rss]`
- **SentenceTextSplitter** — split text into chunks by grouping complete sentences; `chunk_size` and `chunk_overlap` in sentences; regex-based sentence boundary detection
- **CodeSplitter** — split source code using language-aware separators; supports Python, JavaScript, TypeScript, Go, Rust, Java, C++; preserves logical structures (classes, functions); falls back to recursive character splitting
- **ConfluenceLoader** — load pages from Atlassian Confluence as Documents; supports single page by `page_id` or full space by `space_key`; automatic pagination; converts Confluence storage format to plain text via BeautifulSoup; rich metadata (title, author, version, URL); retry with exponential back-off for rate limits; sync `load()` and async `aload()`; `pip install synapsekit[confluence]`
- **SentenceWindowSplitter** — splits text into one chunk per sentence, each padded with up to `window_size` surrounding sentences for context; custom `split_with_metadata()` adds `target_sentence` to chunk metadata; useful for retrieval systems that embed with context but index by target sentence
- **XaiLLM** — xAI Grok LLM provider; OpenAI-compatible API; supports grok-beta, grok-2, grok-2-mini; streaming and tool calling; auto-detected from model name; `pip install synapsekit[openai]`
- **WeatherTool** — get current weather and short-term forecasts via OpenWeatherMap; actions: current, forecast (1-5 day); async-safe with run_in_executor; auth via OPENWEATHERMAP_API_KEY
- **WriterLLM** — Writer (Palmyra) LLM provider; OpenAI-compatible API; supports palmyra-x-004, palmyra-x-003-instruct, palmyra-med, palmyra-fin; streaming and tool calling; `pip install synapsekit[openai]`
- **NovitaLLM** — NovitaAI LLM provider; OpenAI-compatible API; supports Llama, Mistral, Qwen, and other open models; streaming and tool calling; `pip install synapsekit[openai]`
- **StripeTool** — read-only Stripe data lookup: get_customer, list_invoices, get_charge, list_products; stdlib urllib only; auth via STRIPE_API_KEY; async-safe with run_in_executor
- **HTMLTextSplitter** — split HTML documents on block-level tags (h1-h6, p, div, section, article, li, blockquote, pre); strips tags to plain text; falls back to RecursiveCharacterTextSplitter for long sections; stdlib html.parser only
- **TwilioTool** — send SMS and WhatsApp messages via the Twilio REST API; stdlib `urllib` only, no extra deps; auth via constructor args or `TWILIO_ACCOUNT_SID` / `TWILIO_AUTH_TOKEN` / `TWILIO_FROM_NUMBER` env vars; automatic `whatsapp:` prefix handling for both sender and recipient; security warning logged on instantiation; closes #386
- **LinearTool** — manage Linear issues via the Linear GraphQL API; actions: list_issues, get_issue, create_issue, update_issue; stdlib urllib only, no extra deps; auth via constructor arg or `LINEAR_API_KEY` env var; closes #387
- **GCSLoader** — load files from Google Cloud Storage buckets as Documents; supports service account auth (file path or dict) or default credentials; prefix filtering, max_files limit, binary file handling; sync `load()` and async `aload()`; `pip install synapsekit[gcs]`; closes #56
- **SQLLoader** — load rows from any SQLAlchemy-supported database (PostgreSQL, MySQL, SQLite, etc.) as Documents; configurable text/metadata columns; full SQL query support; sync `load()` and async `aload()`; `pip install synapsekit[sql]`; closes #58
- **GitHubLoader** — load README, issues, pull requests, or repository files from GitHub via the REST API; retry with exponential back-off for rate limits and 5xx; optional token auth for higher rate limits; path filtering and limit for files; uses existing `httpx` dep; sync `load_sync()` and async `load()`; closes #48
- **NewsTool** — fetch top headlines and search articles via NewsAPI; actions: get_headlines, search; stdlib urllib only; auth via constructor arg or NEWS_API_KEY env var; closes #384

---

## [1.4.8] — 2026-04-03

### Added

- **WikipediaLoader** — load Wikipedia articles as Documents via `wikipedia-api`; pipe-delimited multi-title support; async `aload()`; `pip install synapsekit[wikipedia]`
- **ArXivLoader** — search arXiv and load papers as Documents (downloads PDFs); uses arxiv v2 `Client` API; async `aload()`; `pip install synapsekit[arxiv,pdf]`
- **EmailLoader** — load emails from IMAP mailboxes (Gmail, Outlook, etc.) as Documents; stdlib-only, no extra deps; configurable folder and IMAP search; async `aload()`
- **ColBERTRetriever** — late-interaction ColBERT retrieval via RAGatouille; `add()`, `retrieve()`, `retrieve_with_scores()`; lazy-loads ragatouille on first use; `pip install synapsekit[colbert]`
- 50 new tests (1500 total)

---

## [1.4.7] — 2026-04-02

### Added

- **SlackLoader** — load messages from Slack channels via Bot API; sync `load()` and async `aload()`; configurable `limit`; per-message metadata; `pip install synapsekit[slack]`
- **NotionLoader** — load pages or full databases from Notion via the Notion API; sync `load()` and async `aload()`; configurable retry/timeout; `pip install synapsekit[notion]`
- **NotionTool** — agent tool for Notion: `search`, `get_page`, `create_page`, `append_block`; built-in retry with exponential back-off; `pip install synapsekit[notion]`
- **Subgraph error handling** — `subgraph_node()` gains `on_error` (`"raise"` / `"retry"` / `"fallback"` / `"skip"`), `max_retries`, and `fallback` parameters
- 17 new tests (1467 total)

---

## [1.4.6] — 2026-04-01

### Added

- **Subgraph error handling** — `subgraph_node()` gains four keyword-only parameters: `on_error` (`"raise"` / `"retry"` / `"fallback"` / `"skip"`), `max_retries` (default 3), and `fallback` (a secondary `CompiledGraph`). On any handled failure the parent state receives `"__subgraph_error__"` with `"type"`, `"message"`, and `"attempts"` keys. Backward-compatible: default behaviour (`on_error="raise"`) is unchanged. (#378)
- 17 new tests (1450 total)

---

## [1.4.5] — 2026-03-31

### Added

- **WeaviateVectorStore** — Weaviate v4 client; lazy collection creation; cosine vector search via `query.near_vector`; metadata filtering; `pip install synapsekit[weaviate]`
- **PGVectorStore** — PostgreSQL + pgvector; async psycopg3; cosine / L2 / inner-product distance; SQL-injection-safe via `psycopg.sql.Identifier`; metadata JSONB; `pip install synapsekit[pgvector]`
- **MilvusVectorStore** — IVF_FLAT and HNSW index types; `MilvusIndexType` enum; metadata filtering via Milvus expressions; Zilliz Cloud support; `pip install synapsekit[milvus]`
- **LanceDBVectorStore** — embedded, serverless; local and cloud (S3/GCS) storage; automatic FTS index; metadata filtering; `pip install synapsekit[lancedb]`

All four backends follow the existing lazy-import `_BACKENDS` pattern and are included in `synapsekit[all]`.

- 0 new tests (1433 total)

---

## [1.4.4] — 2026-03-30

### Added

- **SambaNova provider** — fast inference on Meta Llama, Qwen, and other open models via SambaNova Cloud's OpenAI-compatible API; requires `pip install synapsekit[openai]`; always set `provider="sambanova"`
- **GoogleDriveLoader** — load files and folders from Google Drive via service-account credentials; supports Google Docs (text export), Sheets (CSV export), PDFs, and text files; `pip install synapsekit[gdrive]`
- **`split_with_metadata()`** — new method on `BaseSplitter`; returns `list[dict]` with `text` and `metadata` keys; automatically injects `chunk_index`; all splitters inherit it

### Fixed

- `asyncio.get_event_loop()` → `asyncio.get_running_loop()` in `GoogleDriveLoader` (deprecated in Python 3.10+)
- `build()` in `GoogleDriveLoader.aload()` wrapped in executor (was blocking the event loop)
- Failed file downloads in `GoogleDriveLoader._load_folder` now log a warning instead of silently skipping

- 49 new tests (1452 total)

---

## [1.4.3] — 2026-03-29

### Added

- **XMLLoader** — load XML files via stdlib `xml.etree.ElementTree`; optional `tags` filter; no new dependencies
- **DiscordLoader** — load messages from Discord channels via bot token; `before_message_id` / `after_message_id` pagination; rich metadata (author, timestamp, attachments); `pip install synapsekit[discord]`
- **PythonREPLTool timeout** — `timeout: float = 5.0` parameter; Unix uses `signal.SIGALRM`, Windows uses `multiprocessing.Process`; security warning logged on instantiation

### Improved

- **Mermaid conditional edges** — render as dashed arrows (`-.->`) to distinguish from deterministic edges; branch labels prefixed with condition function name (e.g. `route:approve`)
- **SQLiteCheckpointer** — supports `async with` for automatic connection cleanup
- **Windows compatibility** — `audio/x-wav` MIME normalised to `audio/wav`; shell timeout test uses portable Python sleep; graph tracer uses `time.perf_counter()` for sub-millisecond resolution

- 35 new tests (1403 total)

---

## [1.4.2] — 2026-03-28

### Added

- **HuggingFaceLLM** — Hugging Face Inference API via `AsyncInferenceClient`; supports serverless API (model ID) and Dedicated Inference Endpoints (URL); streaming and generate
- **DynamoDBCacheBackend** — serverless LLM response caching on AWS DynamoDB; configurable partition key, TTL via `ttl_seconds`, custom region and endpoint
- **MemcachedCacheBackend** — high-throughput distributed LLM caching via `aiomcache`; TTL via `exptime`; async-native
- **GoogleSearchTool** — Google web search via SerpAPI (`google-search-results`); graceful handling of missing keys, empty results, and API errors
- **Graph versioning and checkpoint migration** — `StateGraph(version="1", migrations={...})`; `CompiledGraph.resume()` now applies migration chains when loading older checkpoint versions; supports direct and chained `(next_version, state_dict)` migrations; missing paths raise `GraphRuntimeError`
- 11 new tests (1368 total)

### Improved

- **SQLQueryTool** — added `params` dict for parameterized queries (eliminates string interpolation); `max_rows` cap; fixed variable name shadowing

---

## [1.4.1] — 2026-03-27

### Added

- **MinimaxLLM provider** — `MinimaxLLM` for Minimax API with SSE streaming; requires `group_id`; auto-detected from `minimax-*` model prefix
- **AlephAlphaLLM provider** — `AlephAlphaLLM` for Aleph Alpha Luminous and Pharia models; httpx-based streaming; auto-detected from `luminous-*`/`pharia-*` prefixes
- **YAMLLoader** — load YAML files (list-of-objects or single-object) into `Document` list; configurable `text_key` and `metadata_keys`; uses `yaml.safe_load()`
- **BingSearchTool** — web search via Bing Web Search API v7; auth via `Ocp-Apim-Subscription-Key`; supports `query`, `count` (capped at 50), `market`
- **WolframAlphaTool** — computational queries via Wolfram Alpha API; short-answer endpoint; thread-executor async wrapper
- **Usage examples** — `examples/` directory with 5 runnable scripts: RAG quickstart, agent with tools, graph workflow, multi-provider, caching & retries
- 30 new tests (1357 total)

### Fixed

- Added missing return type annotations to `_loader_for()` in `loaders/directory.py` and `__getattr__()` in `loaders/__init__.py`

---

## [1.4.0] — 2026-03-25

### Added

- **AI21 Labs provider** — `AI21LLM` for Jamba models (`jamba-1.5-mini`, `jamba-1.5-large`) with 256K context; streaming and native function calling; auto-detected from `jamba-*` model prefix
- **Databricks provider** — `DatabricksLLM` for Databricks Foundation Model APIs (DBRX, Llama 3.1, Mixtral); OpenAI-compatible endpoint; resolves workspace URL from `DATABRICKS_HOST`; auto-detected from `dbrx-*`/`databricks-*`
- **Baidu ERNIE provider** — `ErnieLLM` for ERNIE Bot (`ernie-4.0`, `ernie-3.5`, `ernie-speed`, `ernie-lite`, `ernie-tiny-8k`); native function calling; auto-detected from `ernie-*`
- **llama.cpp provider** — `LlamaCppLLM` for local GGUF models with no API key; queue+thread true streaming; GPU offload via `n_gpu_layers`; auto-detected from `llamacpp` provider string
- **ImageAnalysisTool** — analyze images with any multimodal LLM; accepts local file paths or public URLs; supports OpenAI and Anthropic message formats
- **TextToSpeechTool** — convert text to speech audio via OpenAI TTS; supports `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer` voices and mp3/wav/flac/aac formats
- **SpeechToTextTool** — transcribe audio files via OpenAI Whisper API or local Whisper model; delegates to `AudioLoader`
- **APIBuilderTool** — build and execute API calls from OpenAPI specs or natural-language intent; LLM-assisted operation selection; supports inline specs, spec URLs, and explicit path/method
- **GoogleCalendarTool** — create, list, and delete Google Calendar events via Google Calendar API v3 with Application Default Credentials
- **AWSLambdaTool** — invoke AWS Lambda functions with RequestResponse/Event/DryRun invocation types; standard boto3 credential resolution
- 222 new tests (1327 total)

### Changed

- `RAG` facade auto-detection extended with `moonshot-*`, `glm-*`, `jamba-*`, `@cf/*`, `@hf/*`, `dbrx-*`/`databricks-*`, `ernie-*` model prefixes

---

## [1.0.0] — 2026-03-18

### Added

- **Multimodal support** — `ImageContent` (from_file, from_url, from_base64), `AudioContent`, `MultimodalMessage` with OpenAI/Anthropic format conversion
- **Image loader** — `ImageLoader` with sync/async loading and optional vision LLM description
- **API stability markers** — `@public_api`, `@experimental` (FutureWarning), `@deprecated(reason, alternative)` (DeprecationWarning)
- 42 new tests (1011 total)

---

## [0.9.0] — 2026-03-18

### Added

- **A2A protocol** — `A2AClient`, `A2AServer`, `AgentCard` for Google Agent-to-Agent protocol; `A2ATask`, `A2AMessage`, `TaskState` types
- **Agent guardrails** — `ContentFilter` (blocked patterns/words/max length), `PIIDetector` (email, phone, SSN, credit card, IP), `TopicRestrictor`, `Guardrails` (composite checker)
- **Distributed tracing** — `DistributedTracer` and `TraceSpan` with parent-child relationships, events, and timing
- 64 new tests (1008 total)

---

## [0.8.0] — 2026-03-18

### Added

- **Evaluation metrics** — `FaithfulnessMetric` (claim verification), `RelevancyMetric` (document relevance), `GroundednessMetric` (answer grounding score 0-1)
- **Evaluation pipeline** — `EvaluationPipeline` runs multiple metrics, `EvaluationResult` with mean_score aggregation
- **OpenTelemetry tracing** — `OTelExporter` with optional OTLP export, `Span` spans, `TracingMiddleware` auto-traces LLM calls
- **Tracing UI** — `TracingUI` renders traces as HTML dashboard, saves to file, or serves via local HTTP server
- 50 new tests (944 total)

---

## [0.7.0] — 2026-03-18

### Added

- **MCP client** — `MCPClient` connects to MCP servers via stdio or SSE transport; `MCPToolAdapter` wraps MCP tools as `BaseTool` instances
- **MCP server** — `MCPServer` exposes SynapseKit tools as MCP-compatible tools via stdio or SSE
- **Supervisor agent** — `SupervisorAgent` delegates tasks to `WorkerAgent` instances via DELEGATE/FINAL protocol
- **Handoff chain** — `HandoffChain` with condition-based `Handoff` transfers between agents, returns `HandoffResult`
- **Crew** — `Crew` for role-based multi-agent teams with `CrewAgent`, `Task`, sequential and parallel execution
- 60 new tests (844 total)

---

## [0.6.9] — 2026-03-18

### Added

- **Slack tool** — `SlackTool` sends messages via Slack webhook URL or Web API bot token (`SLACK_WEBHOOK_URL` / `SLACK_BOT_TOKEN` env vars, stdlib only)
- **Jira tool** — `JiraTool` interacts with Jira REST API v2: search issues (JQL), get issue, create issue, add comment (`JIRA_URL`, `JIRA_EMAIL`, `JIRA_API_TOKEN`, stdlib only)
- **Brave Search tool** — `BraveSearchTool` web search via Brave Search API (`BRAVE_API_KEY`, stdlib only)
- **Approval node** — `approval_node()` factory returns a graph node that gates on human approval, raising `GraphInterrupt` when `state[key]` is falsy; supports dynamic messages via callable
- **Dynamic route node** — `dynamic_route_node()` factory returns a graph node that routes to different compiled subgraphs based on a routing function; supports sync/async routing and input/output mapping
- 52 new tests (795 total)

---

## [0.6.8] — 2026-03-18

### Added

- **Email tool** — `EmailTool` sends emails via SMTP with configurable settings or environment variables (`SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`)
- **GitHub API tool** — `GitHubAPITool` searches repos/issues and fetches details via GitHub REST API (stdlib only, no deps)
- **PubMed search tool** — `PubMedSearchTool` searches biomedical literature on PubMed via NCBI E-utilities (stdlib only)
- **Vector search tool** — `VectorSearchTool` wraps any `Retriever` as an agent tool for knowledge base queries
- **YouTube search tool** — `YouTubeSearchTool` searches YouTube videos with titles, channels, durations, view counts (`pip install synapsekit[youtube]`)
- **Execution trace** — `ExecutionTrace` and `TraceEntry` collect and analyze graph execution events with timing, durations, and human-readable summaries
- **WebSocket streaming** — `ws_stream()` streams graph execution over WebSocket connections (works with Starlette, FastAPI, plain websockets)
- `GraphEvent.to_ws()` — JSON serialization for WebSocket transmission
- 45 new tests (743 total)

---

## [0.6.7] — 2026-03-17

### Changed

- **Python version requirement** — raised minimum from `>=3.9` to `>=3.10`
- Added Python 3.14 classifier

---

## [0.6.6] — 2026-03-16

### Added

- **Perplexity LLM** — `PerplexityLLM` for Perplexity AI with Sonar models, OpenAI-compatible
- **Cerebras LLM** — `CerebrasLLM` for Cerebras ultra-fast inference, OpenAI-compatible
- **Hybrid search retrieval** — `HybridSearchRetriever` combines BM25 + vector similarity via Reciprocal Rank Fusion
- **Self-RAG retrieval** — `SelfRAGRetriever` with self-reflective retrieve-grade-generate-check loop
- **Adaptive RAG retrieval** — `AdaptiveRAGRetriever` classifies query complexity and routes to different retrieval strategies
- **Multi-step retrieval** — `MultiStepRetriever` iterative retrieval-generation with gap identification
- **arXiv search tool** — `ArxivSearchTool` searches arXiv for academic papers (stdlib only)
- **Tavily search tool** — `TavilySearchTool` AI-optimized web search via Tavily API
- **Buffer memory** — `BufferMemory` simplest unbounded message buffer
- **Entity memory** — `EntityMemory` LLM-based entity extraction with running descriptions and eviction
- 56 new tests (698 total)

---

## [0.6.5] — 2026-03-15

### Added

- **Cohere reranker** — `CohereReranker` reranks retrieval results using the Cohere Rerank API
- **Step-back retrieval** — `StepBackRetriever` generates step-back questions for broader context + parallel retrieval
- **FLARE retrieval** — `FLARERetriever` Forward-Looking Active REtrieval with iterative `[SEARCH: ...]` markers
- **DuckDuckGo search tool** — `DuckDuckGoSearchTool` extended search with text and news types
- **PDF reader tool** — `PDFReaderTool` reads and extracts text from PDF files with optional page selection
- **GraphQL tool** — `GraphQLTool` executes GraphQL queries against any endpoint
- **Token buffer memory** — `TokenBufferMemory` token-budget-aware memory that drops oldest messages (no LLM)
- **Redis LLM cache** — `RedisLLMCache` distributed Redis cache backend (`pip install synapsekit[redis]`)
- 55 new tests (642 total)

---

## [0.6.4] — 2026-03-15

### Added

- **Docx loader** — `DocxLoader` for Word documents via `python-docx`
- **Markdown loader** — `MarkdownLoader` with optional YAML frontmatter stripping
- **HyDE retrieval** — `HyDERetriever` Hypothetical Document Embeddings retrieval strategy
- **Shell tool** — `ShellTool` shell command execution with timeout and allowed-commands filter
- **SQL schema inspection tool** — `SQLSchemaInspectionTool` lists tables and describes columns
- **Filesystem LLM cache** — `FilesystemLLMCache` persistent JSON file-based cache backend
- **JSON file checkpointer** — `JSONFileCheckpointer` JSON file-based graph checkpoint persistence
- **TokenTracer COST_TABLE** — added GPT-4.1, o3, o4-mini, Gemini 2.5, DeepSeek-V3/R1, Groq models
- 45 new tests (587 total)

---

## [0.6.3] — 2026-03-14

### Added

- **Typed state with reducers** — `TypedState` and `StateField` for safe parallel state merging in graph workflows; per-field reducers control how concurrent node outputs are combined (closes #253)
- **Parallel subgraph execution** — `fan_out_node()` runs multiple subgraphs concurrently with `asyncio.gather()`, supports per-subgraph input mappings and custom merge functions (closes #248)
- **SSE streaming** — `sse_stream()` streams graph execution as Server-Sent Events for HTTP responses (closes #238)
- **Event callbacks** — `EventHooks` and `GraphEvent` for registering callbacks on node_start, node_complete, wave_start, wave_complete events during graph execution (closes #240)
- **Semantic LLM cache** — `SemanticCache` uses embeddings for similarity-based cache lookup instead of exact match; configurable threshold and maxsize (closes #196)
- **Summarization tool** — `SummarizationTool` summarizes text using an LLM with concise, bullet_points, or detailed styles (closes #223)
- **Sentiment analysis tool** — `SentimentAnalysisTool` analyzes text sentiment (positive/negative/neutral) with confidence and explanation (closes #225)
- **Translation tool** — `TranslationTool` translates text between languages with optional source language specification (closes #224)
- 28 new tests (540 total)

---

## [0.6.2] — 2026-03-13

### Added

- **CRAG (Corrective RAG)** — `CRAGRetriever` grades retrieved documents for relevance using an LLM, rewrites the query and retries when too few are relevant (closes #152)
- **Query Decomposition** — `QueryDecompositionRetriever` breaks complex queries into sub-queries, retrieves for each, and deduplicates results (closes #156)
- **Contextual Compression** — `ContextualCompressionRetriever` compresses retrieved documents to only the relevant excerpts using an LLM (closes #146)
- **Ensemble Retrieval** — `EnsembleRetriever` fuses results from multiple retrievers using weighted Reciprocal Rank Fusion (closes #147)
- **SQLite Conversation Memory** — `SQLiteConversationMemory` persists chat history to SQLite with multi-conversation support and optional sliding window (closes #138)
- **Summary Buffer Memory** — `SummaryBufferMemory` tracks token budget and progressively summarizes older messages when the buffer exceeds the limit (closes #135)
- **Human Input Tool** — `HumanInputTool` pauses agent execution to ask the user a question, supports custom sync/async input functions (closes #228)
- **Wikipedia Tool** — `WikipediaTool` searches and fetches Wikipedia article summaries using the REST API, no extra dependencies (closes #202)
- 30 new tests (512 total)

---

## [0.6.1] — 2026-03-13

### Added

- **Human-in-the-Loop** — `GraphInterrupt` exception pauses graph execution for human review; `InterruptState` holds interrupt details; `resume(updates=...)` applies human edits and continues from checkpoint
- **Subgraphs** — `subgraph_node(compiled_graph, input_mapping, output_mapping)` nests a `CompiledGraph` as a node in a parent graph with key mapping
- **Token-level streaming** — `llm_node(llm, stream=True)` wraps any `BaseLLM` as a graph node; `stream_tokens()` yields `{"type": "token", ...}` events for real-time output
- **Self-Query retrieval** — `SelfQueryRetriever` uses an LLM to decompose natural-language queries into semantic search + metadata filters automatically
- **Parent Document retrieval** — `ParentDocumentRetriever` embeds small chunks for precision search, returns full parent documents for richer context
- **Cross-Encoder reranking** — `CrossEncoderReranker` reranks retrieval results with cross-encoder models for higher accuracy (requires `synapsekit[semantic]`)
- **Hybrid memory** — `HybridMemory` keeps a sliding window of recent messages in full, plus an LLM-generated summary of older messages for token-efficient long conversations
- 30 new tests (482 total)

---

## [0.6.0] — 2026-03-13

### Added

- **Built-in tools** (6 new):
  - `HTTPRequestTool` — GET/POST/PUT/DELETE/PATCH with aiohttp, configurable timeout and max response length
  - `FileWriteTool` — write/append files with auto-mkdir
  - `FileListTool` — list directories with glob patterns, recursive mode
  - `DateTimeTool` — current time, parse, format with timezone support
  - `RegexTool` — findall, match, search, replace, split with flag support
  - `JSONQueryTool` — dot-notation path queries on JSON data
- **LLM providers** (3 new, all OpenAI-compatible):
  - `OpenRouterLLM` — unified API for 200+ models (auto-detected from `/` in model name)
  - `TogetherLLM` — Together AI fast inference
  - `FireworksLLM` — Fireworks AI optimized serving
- **Advanced retrieval** (2 new):
  - `ContextualRetriever` — Anthropic-style contextual retrieval (LLM prepends context before embedding)
  - `SentenceWindowRetriever` — sentence-level embedding with configurable window expansion at retrieval time
- RAG facade auto-detects `openrouter` (model names with `/`), `together`, and `fireworks` providers
- 37 new tests (452 total)

### Changed

- Lazy imports extended for new providers (`OpenRouterLLM`, `TogetherLLM`, `FireworksLLM`)
- `agents/tools/__init__.py` exports 11 built-in tools (was 5)

---

## [0.5.3] — 2026-03-12

### Added

- **Azure OpenAI LLM provider** — `AzureOpenAILLM` for enterprise Azure OpenAI deployments with streaming and function calling (closes #183)
- **Groq LLM provider** — `GroqLLM` for ultra-fast inference with Llama, Mixtral, Gemma models (closes #166)
- **DeepSeek LLM provider** — `DeepSeekLLM` with OpenAI-compatible API, supports function calling (closes #170)
- **SQLite LLM cache** — persistent `cache_backend="sqlite"` option via `LLMConfig(cache=True, cache_backend="sqlite")`, survives process restarts (closes #191)
- **RAG Fusion retrieval** — `RAGFusionRetriever` generates query variations with an LLM and fuses results via Reciprocal Rank Fusion for better recall (closes #158)
- **Excel loader** — `ExcelLoader` for `.xlsx` files, one Document per sheet (closes #63)
- **PowerPoint loader** — `PowerPointLoader` for `.pptx` files, one Document per slide (closes #62)
- RAG facade auto-detects `deepseek` and `groq` providers from model names
- 26 new tests (415 total)

### Changed

- `LLMConfig` gains `cache_backend` (`"memory"` or `"sqlite"`) and `cache_db_path` fields
- Lazy imports extended for new providers (`AzureOpenAILLM`, `GroqLLM`, `DeepSeekLLM`) and loaders (`ExcelLoader`, `PowerPointLoader`)

---

## [0.5.2] — 2026-03-12

### Added

- **`__repr__` methods** — human-readable repr on `StateGraph`, `CompiledGraph`, `RAGPipeline`, `ReActAgent`, `FunctionCallingAgent` (closes #3)
- **Empty document handling** — `RAGPipeline.add()` silently skips empty/whitespace-only text instead of producing empty chunks (closes #20)
- **Retry for `call_with_tools()`** — `LLMConfig(max_retries=N)` now applies to native function-calling, not just `generate()` (closes #22)
- **Cache hit/miss statistics** — `BaseLLM.cache_stats` property returns `{"hits", "misses", "size"}` when caching is enabled (closes #23)
- **MMR retrieval** — `InMemoryVectorStore.search_mmr()` and `Retriever.retrieve_mmr()` for diversity-aware retrieval (closes #30)
- **Rate limiting** — `LLMConfig(requests_per_minute=N)` adds token-bucket rate limiting to all LLM calls (closes #35)
- **Structured output with retry** — `generate_structured(llm, prompt, schema=MyModel)` parses LLM output into Pydantic models, retrying on parse failure (closes #43)
- 29 new tests (389 total)

### Changed

- LLM providers now override `_call_with_tools_impl()` instead of `call_with_tools()` (base class handles retry + rate limiting)
- `LLMConfig` gains `requests_per_minute` field (default `None` — off)

---

## [0.5.1] — 2026-03-12

### Added

- **`@tool` decorator** — create agent tools from plain functions with `@tool(name="...", description="...")`; auto-generates JSON Schema from type hints, supports sync and async functions
- **Metadata filtering** — `VectorStore.search(metadata_filter={"key": "value"})` filters results by metadata before ranking; implemented in `InMemoryVectorStore`, signature updated in all backends
- **Vector store lazy exports** — `ChromaVectorStore`, `FAISSVectorStore`, `QdrantVectorStore`, `PineconeVectorStore` now importable from `synapsekit` and `synapsekit.retrieval` via lazy imports
- **File existence checks** — `PDFLoader`, `HTMLLoader`, `CSVLoader`, `JSONLoader` now raise `FileNotFoundError` with a clear message before attempting to read
- **Parameter validation** — `FunctionCallingAgent` and `ReActAgent` reject `max_iterations < 1`; `ConversationMemory` rejects `window < 1`

### Fixed

- Loader import-error tests now use temp files to work correctly with file existence checks

### Stats

- 357 tests passing (was 332)

---

## [0.5.0] — 2026-03-12

### Added

- **Text Splitters** — `BaseSplitter` ABC, `CharacterTextSplitter`, `RecursiveCharacterTextSplitter`, `TokenAwareSplitter`, `SemanticSplitter` (cosine similarity boundaries)
- **Function calling** — `call_with_tools()` added to `GeminiLLM` and `MistralLLM` (now 4 providers support native tool use)
- **LLM Caching** — `AsyncLRUCache` with SHA-256 cache keys, opt-in via `LLMConfig(cache=True)`
- **LLM Retries** — exponential backoff via `retry_async()`, skips auth errors, opt-in via `LLMConfig(max_retries=N)`
- **Graph Cycles** — `compile(allow_cycles=True)` skips static cycle detection for intentional loops
- **Configurable max_steps** — `compile(max_steps=N)` overrides the default `_MAX_STEPS=100` guard
- **Graph Checkpointing** — `BaseCheckpointer` ABC, `InMemoryCheckpointer`, `SQLiteCheckpointer`
- `CompiledGraph.resume(graph_id, checkpointer)` — re-execute from saved state
- Adjacency index optimization for `CompiledGraph._next_wave()`
- `RAGConfig.splitter` — plug any `BaseSplitter` into the RAG pipeline
- `TextSplitter` alias preserved for backward compatibility
- 65 new tests (332 total)

### Changed

- `LLMConfig` gains `cache`, `cache_maxsize`, `max_retries`, `retry_delay` fields (all off by default)
- `pyproject.toml` description updated
- Version bumped to `0.5.0`

---

## [0.4.0] — 2026-03-11

### Added

- **Graph Workflows** — `StateGraph` fluent builder with compile-time validation and DFS cycle detection
- **`CompiledGraph`** — wave-based async executor with `run()`, `stream()`, and `run_sync()`
- **`Node`**, **`Edge`**, **`ConditionalEdge`** — sync and async node functions, static and conditional routing
- **`agent_node()`**, **`rag_node()`** — wrap `AgentExecutor` or `RAGPipeline` as graph nodes
- **Parallel execution** — nodes in the same wave run concurrently via `asyncio.gather()`
- **Mermaid export** — `CompiledGraph.get_mermaid()` returns a flowchart string
- **`_MAX_STEPS = 100`** guard against infinite conditional loops
- **`GraphConfigError`**, **`GraphRuntimeError`** — distinct error types for build vs runtime failures
- 44 new tests (267 total)

### Changed

- **Build tooling** migrated from Poetry to [uv](https://github.com/astral-sh/uv)
- `pyproject.toml` updated to PEP 621 `[project]` format with hatchling build backend
- Version bumped to `0.4.0`

---

## [0.3.0] — 2026-03-10

### Added

- **`BaseTool` ABC** — `run()`, `schema()`, `anthropic_schema()`, `ToolResult`
- **`ToolRegistry`** — tool lookup by name, OpenAI and Anthropic schema generation
- **`AgentMemory`** — step scratchpad with `format_scratchpad()` and `max_steps` limit
- **`ReActAgent`** — Thought → Action → Observation loop, works with any `BaseLLM`
- **`FunctionCallingAgent`** — native OpenAI `tool_calls` and Anthropic `tool_use`, multi-tool per step
- **`AgentExecutor`** — unified runner with `run()`, `stream()`, `run_sync()`, auto-selects agent type
- **`call_with_tools()`** — added to `OpenAILLM` and `AnthropicLLM`
- **Built-in tools**: `CalculatorTool`, `PythonREPLTool`, `FileReadTool`, `WebSearchTool`, `SQLQueryTool`
- 82 new tests (223 total)

---

## [0.2.0] — 2026-03-08

### Added

- **Loaders**: `PDFLoader`, `HTMLLoader`, `CSVLoader`, `JSONLoader`, `DirectoryLoader`, `WebLoader`
- **Output parsers**: `JSONParser`, `PydanticParser`, `ListParser`
- **Vector store backends**: `ChromaVectorStore`, `FAISSVectorStore`, `QdrantVectorStore`, `PineconeVectorStore`
- **LLM providers**: `OllamaLLM`, `CohereLLM`, `MistralLLM`, `GeminiLLM`, `BedrockLLM`
- **Prompt templates**: `PromptTemplate`, `ChatPromptTemplate`, `FewShotPromptTemplate`
- **`VectorStore` ABC** — unified interface for all backends
- `Retriever.add()` — cleaner public API
- `RAGPipeline.add_documents(docs)` — ingest `List[Document]` directly
- `RAG.add_documents()` and `RAG.add_documents_async()`
- 89 new tests (141 total)

---

## [0.1.0] — 2026-03-05

### Added

- **`BaseLLM` ABC** and `LLMConfig`
- **`OpenAILLM`** — async streaming
- **`AnthropicLLM`** — async streaming
- **`SynapsekitEmbeddings`** — sentence-transformers backend
- **`InMemoryVectorStore`** — numpy cosine similarity with `.npz` persistence
- **`Retriever`** — vector search with optional BM25 reranking
- **`TextSplitter`** — pure Python, zero dependencies
- **`ConversationMemory`** — sliding window
- **`TokenTracer`** — tokens, latency, and cost per call
- **`TextLoader`**, **`StringLoader`**
- **`RAGPipeline`** — full retrieval-augmented generation orchestrator
- **`RAG`** facade — 3-line happy path
- **`run_sync()`** — works inside and outside running event loops
- 52 tests
