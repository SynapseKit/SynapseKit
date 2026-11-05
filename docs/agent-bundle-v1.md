# SynapseKit Agent Bundle Format 1.0

Status: open, versioned reference specification for issue #751.

An agent bundle is a ZIP archive with the `.agent` extension. Consumers must
treat every bundle as untrusted input. Verification and installation must not
import or execute any bundled file.

## Required layout

```text
example.agent
|-- manifest.json
|-- signature.ed25519
|-- README.md
`-- evals/
    `-- ... at least one eval-suite file
```

`prompts/`, `tools/`, `memory/`, `router/`, and `style.md` are portable payload
conventions. Memory content declares `ump/1.0` in the manifest when present.
An optional entrypoint uses `relative/file.py:object_name` syntax, but an
installer never loads it directly.

## Manifest

`manifest.json` uses UTF-8 JSON and `schema_version: "1.0"`. It records the
agent identity, version, author, tags, optional eval score, publisher identity,
sandbox policy, and a sorted inventory of payload files. Every inventory item
contains a POSIX relative path, byte size, and lowercase SHA-256 digest.

The signature is calculated over canonical JSON: keys sorted, no insignificant
whitespace, ASCII JSON escapes, and no NaN values. `signature.ed25519` contains
the 64 raw Ed25519 signature bytes. The manifest embeds the signer's key id and
raw public key in base64.

An embedded public key proves that the bundle was not modified after signing;
it does not prove who signed it. Authenticity requires pinning the public key
through an independent channel.

## Verification and extraction

Readers must reject duplicate ZIP entries, absolute or parent-relative paths,
backslashes, drive-qualified paths, symbolic links, untracked payloads, missing
payloads, oversized archives, invalid hashes, and invalid signatures. Readers
must verify the complete archive before extracting it.

Untrusted installs are inert and require a sandbox runner before use. Trusting
a publisher does not make extraction execute code; it only records that the
signature matched a caller-pinned key.

## Runtime audit receipts (separate artifact)

This specification defines installable `.agent` packages, not runtime audit
receipts. `synapsekit.audit.export_audit_bundle(...)` writes a separate signed
`.audit.zip` archive. New audit bundles include an optional `provenance.json`
document in W3C PROV-JSON format, with audit records as activities, actors as
agents, and explicitly supplied source/input/output identifiers as entities.

The graph omits raw prompts, retrieved text, tool arguments, credentials, and
model output values. SHA-256 values in the graph are deterministic, unsalted
pseudonymous digests, not encryption or anonymization. They do not prevent
guessing low-entropy identifiers or content by hashing likely values. Do not rely
on these digests to keep sensitive values confidential. Activity and record
attributes expose only `sk:eventIdHash` and `sk:runIdHash`; agents expose only
`sk:actorHash`. Referenced-entity IDs include the hashed run ID, so identical
references in separate runs remain distinct. Source entities are created only
from `RETRIEVAL` or `TOOL_RESULT` records that provide source-ID metadata;
`citation_id` is an explicit source alias, but `citation_ids` is not. RAG
lineage requires both an earlier source-producing record and trusted source-ID
metadata on the exact LLM call.
The signed `manifest.json` contains `provenance_hash`, the canonical JSON
digest of `provenance.json`, before the manifest is signed.
`synapsekit.audit.verify(...)` checks the digest and graph references; bundles
without this optional entry remain readable for legacy compatibility.

### Opt-in runtime instrumentation

Runtime capture is opt-in; ordinary RAG and agent calls remain unchanged. Wrap
the live objects with `VerifiableAgent` and name the nested components that
their implementation calls internally:

```python
from synapsekit.audit import (
    AuditTracer,
    PIIRedactor,
    SigningPolicy,
    VerifiableAgent,
    export_audit_bundle,
)

tracer = AuditTracer(redactor=PIIRedactor())
audited_rag = VerifiableAgent(
    rag_pipeline,
    tracer=tracer,
    instrument_attrs=["config.retriever", "config.llm"],
)
answer = await audited_rag.ask("question")

audited_agent = VerifiableAgent(
    function_calling_agent,
    tracer=tracer,
    instrument_attrs=["_llm", "_registry._tools"],
)
answer = await audited_agent.run("question")

policy = SigningPolicy.ed25519()
bundle_path = export_audit_bundle(tracer.records, policy, "run.audit.zip")
```

`instrument_attrs` accepts dotted attribute paths; mapping and sequence leaves
instrument their contained objects. This captures the actual retriever, LLM,
and tool methods without changing provider clients. The RAG example invokes
the wrapper directly; when nested components are instrumented, unpaired
single-record calls do not become forward parent references because their
audit event is appended only after completion. For an explicit source ID, the
audit record retains it subject to the configured redactor; the PROV graph
stores only its deterministic SHA-256-derived digest. LLM message contents
become separate prompt-fragment entities identified by content hashes, which
are also susceptible to guessing for low-entropy content. Retrieved text, prompts,
tool arguments/results, credentials, and model output are not copied into
`provenance.json`. Source entities are admitted only from returned metadata in
`RETRIEVAL`/`TOOL_RESULT` output, never from input arguments. A citation in an LLM
response is linked only to a matching source from the same run that was recorded
before its parent LLM call and explicitly attached to that exact call as trusted
retrieved-source metadata. Prompt/source edges use structured source IDs
propagated internally by SynapseKit's RAG message builder. This audit-only
metadata travels out of band on a list-compatible message object; it adds no
message fields or provider arguments. Visual citations keep the user-facing
`[page N, bbox X]` format; a unique locator maps to its retrieved source ID only
in audit metadata. Duplicate page/bbox locators remain unresolved. Multimodal
prompt fragments hash text blocks in order and omit image/binary blocks.
Free-form prompt text, including user-supplied `[SOURCE]` blocks, is never
parsed to create lineage. Unmatched citations set
`sk:lineageStatus: "unresolved"` and expose only hashed citation IDs, without
creating source metadata.

RAG prompt privacy: `RAGPipeline._format_context_result` includes explicit
`source_id`, `document_id`, `doc_id`, and `citation_id` values in prompt text for
model citations, so those values are sent to the configured LLM provider.
Audit redaction changes audit records only; it does not redact provider
requests. Use opaque aliases or omit sensitive identifiers from metadata when
they should not be sent to the provider.

Tool-result lineage is stricter than same-run ordering: the graph links a
tool result to a later LLM call only when an actual LLM tool-call record
matches the instrumented tool name and exact keyword-argument map recorded in
`kwargs`, and the subsequent message contains the same tool-call ID and result
content. Positional-only invocations are not correlated because positional
values are not normalized against the provider's named tool arguments.
Ambiguous or incomplete correlations are left unlinked.

### Relationship to `GroundedSignal`

`GroundedSignal` labels the trust source of a numeric signal such as a reward
or quality score (`external_override` versus `self_reported`). It does not
represent document/tool/prompt data flow and is not embedded into these
lineage records. Conversely, a signed PROV graph records and authenticates
relationships among runtime inputs and outputs; it does not establish that a
numeric claim or the content of a source is true. The two mechanisms are
complementary and orthogonal, so their semantics should not be conflated.

## Registry

The reference registry is a static directory containing `index.json`, immutable
bundles under `packages/NAME/VERSION/`, and signed reviews under
`reviews/NAME/VERSION/`. This layout can be served by any static HTTP server.
Publishing requires a pinned publisher key unless the registry operator
explicitly enables open, self-signed publication.

Ranking uses the publisher's eval score as 70% of the score and the mean signed
review quality as 30%. Review quality is the mean of its eval score and its
one-to-five rating normalized to zero-to-one. Without publisher evals, signed
review quality is the score; without either, the score is zero.
