# MCP tool-definition scanning

`scan_tools` applies local, deterministic heuristics to MCP `ToolDefinition`
snapshots. It reports possible instruction overrides, role spoofing, secret
exfiltration, concealment, remote script execution, and hidden or bidirectional
characters. It does not contact a service or require the MCP SDK.
It scans the effective description and every string key and value in the input
schema, including nested property names, descriptions, examples and defaults.
Each string is scanned separately; signals split across strings can be missed.
Findings are deduplicated by rule per tool and do not identify the matching field.
When a description is missing or empty, it scans the adapter's effective
fallback description (`MCP tool: {name}`). Findings retain the fingerprint of
the original snapshot, including its original description, so pinning semantics
do not change.

```python
from synapsekit.mcp.pinning import make_tool_definition
from synapsekit.mcp.scanner import scan_tools

snapshot = make_tool_definition(
    "lookup", "Search public records", {"type": "object", "properties": {}}
)
report = await scan_tools([snapshot])
for finding in report.findings:
    print(finding.tool_name, finding.fingerprint, finding.severity, finding.rule)
```

Each immutable finding includes the tool name, definition fingerprint, severity,
rule identifier and a short explanation. Matched text itself is not copied
into the report. Tool names originate from the server: escape or sanitize them
before writing to logs or terminals. An empty report means no rule matched,
**not** that a tool is safe.

## Opt-in registration gate and pinning

```python
from synapsekit.mcp.client import MCPClient
from synapsekit.mcp.pinning import ToolApprovalRequiredError, ToolDriftDetectedError
from synapsekit.mcp.scanner import ToolScanBlockedError

client = MCPClient(
    server_id="catalog-prod",             # stable identity, not a server-supplied name
    pin_store_path="tool-pins.json",     # optional, but recommended
    block_on_high_severity=True,          # disabled by default
)
try:
    tools = await client.connect_stdio("example-mcp-server")
except ToolScanBlockedError as exc:
    report = exc.report  # inspect findings; no adapters were registered
except (ToolApprovalRequiredError, ToolDriftDetectedError) as exc:
    # Review exc.pending_tools and the scan report before explicitly approving.
    report = client.last_scan_report
    # After human review: tools = await client.approve_current_tools()
```

The gate scans the exact snapshots used for registration, including without a
pin store. It also scans the freshly fetched snapshots during approval, before
any pin write or registration. High findings block the entire registration;
medium findings are reported but do not block. Scanner failures propagate and
prevent registration when the gate is enabled. The most recent report is
available as `client.last_scan_report`; it is `None` before a scan or after a
scanner failure. A scan block does not overwrite saved pins. Pinning separately
requires review of new or changed fingerprints and detects drift; prior approval
never overrides a high-severity finding. Reconnecting clears exposed tools
before tearing down the old connection; with pinning or the gate enabled, it
also revokes retained adapters before teardown. Calls already in flight are not
cancelled. With both protections disabled, retained adapters are not revoked
and no automatic report is produced.

When an initial connection is rejected or registration fails, the newly opened
session and transport are closed; teardown errors do not replace the original
failure. Approval-required and drift exceptions retain the connection so that
the user can review and approve definitions.

## Limits and operational precautions

Scanning accepts at most 256 tools and 16,384 Unicode characters per effective
description or individual schema string. Descriptions and schema strings share
a 262,144-character total budget across all tools, measured before normalization.
Schema traversal also allows at most 16,384 nodes in total (including container,
key and value nodes) and depth 64, with each schema root at depth zero. Tool
names have a separate limit of 16,384 characters each and 262,144 in total.
Schema integers are limited to 4,096 bits; invalid JSON-like values, non-string
keys and non-finite floats are rejected. Cycles eventually exceed the depth limit.
The gate checks these limits before creating tool snapshots, including on
approval. `scan_tools` raises `ToolScanLimitError` (a `ValueError`) instead of
returning a partial report if a limit is exceeded; the gate fails closed.
Normalization and pattern matching run in a worker thread. This avoids running
the scan directly on the event loop, but threads do not provide a hard CPU-time
limit or isolation from Python's GIL. These limits do not bound the transport
response and do not prevent resource exhaustion before the SDK receives and
decodes it. The pre-snapshot validation itself runs on the event loop.

These rules are deliberately narrow heuristics and may produce false positives
or miss paraphrased, encoded, multilingual, context-dependent, or novel attacks.
They inspect effective descriptions and input-schema strings, not tool names when
a description is present, server manifests, tool
outputs, transport behavior, or the server's code. A malicious server can change
its behavior without changing its definition, or change a tool after a scan; a
snapshot and a fingerprint cannot enforce server-side behavior. Do not treat a
clean scan or a matching pin as proof of safety. Review the server, limit its
permissions and network access, isolate credentials, and monitor tool use.
