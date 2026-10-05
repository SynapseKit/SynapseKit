# Memory-as-MCP-Server

Expose personal memory to any Model Context Protocol (MCP) client (such as Claude Desktop, Cursor, or custom tools) via `KnowledgeMesh` and `MCPServer`.

## Overview

SynapseKit exposes three dedicated personal memory MCP tools alongside standard mesh retrieval tools:

- `memory_search`: Performs ranked search over indexed personal memory, returning PII-redacted hits audited as `MEMORY_READ`.
- `memory_recall`: Returns the synthesized natural-language answer plus ranked hits, redacted and audited as `MEMORY_READ`.
- `memory_store`: Policy-gates inbound content, writes Universal Memory Protocol (UMP) markdown files into a managed `memory_root`, triggers incremental reindexing, and audits the write as `MEMORY_WRITE`.

Because `MCPServer` dynamically invokes `target.as_mcp_tools()`, passing a `KnowledgeMesh` target exposes all 7 tools to any MCP client over standard MCP stdio or SSE transports with no server code changes.

## Tool Specifications

### `memory_search`
- **Parameters**:
  - `query` (string, required): Search query.
  - `top_k` (integer, optional): Maximum ranked hits to return.
  - `actor` (string, optional): Identifier of the client or user making the read (default `"mcp-client"`).
- **Behavior**: Retrieves ranked hits from `KnowledgeMesh`, redacts PII in hit text, records a `MEMORY_READ` audit event (recording query and hit count, excluding raw memory content), and returns JSON hits.

### `memory_recall`
- **Parameters**:
  - `query` (string, required): Search query.
  - `top_k` (integer, optional): Maximum ranked hits to return.
  - `actor` (string, optional): Identifier of the client or user making the read (default `"mcp-client"`).
- **Behavior**: Retrieves synthesized answer and hits, redacts PII in answer and hits, records a `MEMORY_READ` audit event, and returns JSON `{"answer": ..., "hits": [...]}`.

### `memory_store`
- **Parameters**:
  - `content` (string, required): Memory content to store.
  - `name` (string, optional): Human-readable title for this memory item.
  - `memory_type` (string, optional): Memory type (one of: `user`, `feedback`, `project`, `reference`, `general`; default `"general"`).
  - `actor` (string, optional): Identifier of the writer (default `"mcp-client"`).
- **Behavior**:
  1. Runs `GuardrailPolicy.check_input(content)`. If policy rejects the write, stops immediately without touching disk and returns an error. The policy records a `GUARDRAIL_VIOLATION` event, and `memory_store` records an actor-attributed `MEMORY_WRITE` event (`blocked=True`, `reasons=[...]`, `actor=actor`).
  2. Constructs a UMP document with metadata and provenance (`authors=[actor]`).
  3. Writes file to `memory_root / "<slug>-<id>.md"`.
  4. Emits a `MEMORY_WRITE` audit event.
  5. Runs `KnowledgeMesh.reindex(force=False)`.

## Policy Gating and Audit Receipts

Policy gating and audit trail tracing are managed via a single shared `GuardrailPolicy` instance passed in `MemoryMCPConfig`:

```python
from pathlib import Path
from synapsekit.guardrails import GuardrailPolicy
from synapsekit.guardrails.input_guards import TopicGuard
from synapsekit.mesh import KnowledgeMesh, MeshConfig, MemoryMCPConfig

# 1. Configure mesh root and memory root
memory_dir = Path.home() / ".synapsekit" / "memory"
mesh = KnowledgeMesh(MeshConfig(roots=[memory_dir]))

# 2. Configure GuardrailPolicy with topic gating
policy = GuardrailPolicy(
    guards=[TopicGuard(blocked_topics=["secrets", "confidential"])]
)

# 3. Create MemoryMCPConfig and build tools
config = MemoryMCPConfig(memory_root=memory_dir, policy=policy)
tools = mesh.as_mcp_tools(config)
```

- **Policy Gating**: Opt-in via guard list. An empty guard list allows all writes by default while still recording audit receipts.
- **Audit Trail**: Always on. Every read and write emits signed audit records. Calling `policy.export_audit_bundle(path)` generates a signed, cryptographic receipt bundle verifiable via `synapsekit.audit.verifier.verify(path)`.

## PII Redaction & Scope

Memory reads (`memory_search` and `memory_recall`) pass text through `PIIRedactor`:

- **Covered Text**: `hit.text` and `result.answer` have detected PII (emails, SSNs, credit cards, IP addresses, phone numbers) replaced with `[REDACTED:<LABEL>]`.
- **Structural Metadata**: Structural citation fields (`path`, `line_start`, `line_end`, `score`, `headings`) are preserved without redaction to maintain citation utility.
- **Known Limitation**: Local file paths are not redacted. If file paths contain sensitive user directories or names, ensure workspace paths are sanitized appropriately.

## Memory Root Requirements

`memory_config.memory_root` **must** be present in `mesh.config.expanded_roots()`. Attempting to construct tools with a `memory_root` outside the mesh's configured roots raises a `ValueError` at tool build time.

## Complete Usage Example

```python
import asyncio
from pathlib import Path
from synapsekit import MCPServer
from synapsekit.guardrails import GuardrailPolicy
from synapsekit.guardrails.input_guards import PromptInjectionGuard
from synapsekit.mesh import KnowledgeMesh, MeshConfig, MemoryMCPConfig

async def main():
    root = Path.home() / ".synapsekit" / "memory"
    mesh = KnowledgeMesh(MeshConfig(roots=[root]))

    policy = GuardrailPolicy(guards=[PromptInjectionGuard()])
    config = MemoryMCPConfig(memory_root=root, policy=policy)

    # Expose mesh over MCP server
    server = MCPServer(mesh)
    
    # Export signed audit receipts on shutdown
    # policy.export_audit_bundle("receipts.audit.zip")

if __name__ == "__main__":
    asyncio.run(main())
```
