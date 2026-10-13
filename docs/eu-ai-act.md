# EU AI Act record-keeping mode

`VerifiableAgent(..., compliance="eu_ai_act")` turns on a record-keeping mode built on the existing
`synapsekit.audit` trail (hash-chained `AuditTracer`, Ed25519-signed bundles, standalone `verify`).
There is no second log: the mode adds a lifecycle record, purpose tags and a retention policy to the
records the tracer already writes.

All references are to Regulation (EU) 2024/1689 (the AI Act) as amended by Regulation (EU) 2026/1744 of 8 July 2026 (OJ L, 24.7.2026), which did not
change Articles 12, 19 or 26.

This is an engineering aid, not legal advice and not a conformity assessment. Whether a system is
high-risk, and which of its events are relevant, is for its provider and deployer to decide.

## When the obligations apply

Article 113, third paragraph, point (c), as replaced by Regulation (EU) 2026/1744, applies
Chapter III, Sections 1 to 3 (which hold Articles 12, 19 and 26), except Article 6(5):

- from **2 December 2027** to AI systems classified as high-risk under Article 6(2) and Annex III;
- from **2 August 2028** to AI systems classified as high-risk under Article 6(1) and Annex I.

## Quickstart

```python
from synapsekit.audit import (
    RetentionPolicy,
    SigningPolicy,
    VerifiableAgent,
    end_agent_lifetime,
    export_audit_bundle,
    verify,
)

agent = VerifiableAgent(
    my_agent,
    compliance="eu_ai_act",
    retention=RetentionPolicy(role="deployer"),  # Article 26(6), six months by default
)

await agent.run("...")              # recorded and tagged, as any proxied call
await end_agent_lifetime(agent)     # records the end of the agent's lifetime

policy = SigningPolicy.ed25519()
export_audit_bundle(agent.tracer.drain(), policy, "run.audit.zip")
result = verify("run.audit.zip", trusted_keys={policy.provider.key_id: policy.provider.public_key_bytes()})
assert result.verdict == "MATCH"
```

## Retention policy: the role is required, the period defaults to six months

Article 12 asks for a logging capability; it says nothing about how long logs are kept. The period is
in two other articles, one per role. Article 19(1) has providers keep the logs "for a period
appropriate to the intended purpose of the high-risk AI system, of at least six months, unless provided
otherwise in the applicable Union or national law, in particular in Union law on the protection of
personal data"; Article 26(6) sets the same period, with the same carve-out, for deployers.

| Role | Article | `RetentionPolicy` |
|---|---|---|
| provider | 19(1) | `RetentionPolicy(role="provider")` |
| deployer | 26(6), first subparagraph | `RetentionPolicy(role="deployer")` |

- `role` has no default, because it decides which article applies. `compliance="eu_ai_act"` without a
  `retention` raises `ValueError` saying exactly that.
- `months` defaults to `6`. Longer periods are accepted as given (the article asks for a period
  "appropriate to the intended purpose", at least six months).
- A period shorter than six months is refused, unless `overriding_law` cites the Union or national law
  that provides otherwise — the carve-out both articles make:

  ```python
  RetentionPolicy(role="deployer", months=3, overriding_law="<the law, article and paragraph>")
  ```

  The citation is recorded in the log; SynapseKit does not check that the cited law says so.
- `retain_until(recorded_at)` adds the period to the record's timestamp in calendar months (31 August
  + 6 months is 28 or 29 February), and `may_delete(record, now=...)` is `True` from that moment on.
  Both require timezone-aware datetimes. This is SynapseKit's arithmetic, not a legal determination of
  when the period ends: counting rules such as those of Regulation (EEC, Euratom) No 1182/71, where
  they are applied, can end a period later (at the end of the last day, or on the next working day).

`RetentionPolicy` declares and records the policy. It does not store, archive or delete anything:
keeping the logs for the period is the job of the sink or storage you use.

## Feature → article map

| Feature | Article | What it does |
|---|---|---|
| Every call made through the `VerifiableAgent` proxy (and through `instrument_attrs`) is recorded | 12(1) | Automatic recording of events while the agent is in use |
| First record `agent_lifetime_started`, `end_agent_lifetime()` records `agent_lifetime_ended` | 12(1) "over the lifetime" | Brackets the agent's lifetime inside the hash chain; recording does not stop after the end record, so nothing is dropped silently |
| `purposes` tag in every record's payload (`"12(2)(a)"`, `"12(2)(b)"`, `"12(2)(c)"`) | 12(2)(a) risk situations within Article 79(1) or substantial modification; 12(2)(b) post-market monitoring (Article 72); 12(2)(c) monitoring of operation (Article 26(5)) | Records which purpose each event is logged for. Default: all three for every event kind; `purposes={EventKind.ERROR: ["12(2)(a)"], ...}` narrows them per kind. The full map is written in the lifetime-start record |
| `RetentionPolicy` in the lifetime-start record | 19(1) provider, 26(6) deployer | Records the role, period, legal basis and any overriding law |
| `RemoteBiometricIdentificationProfile` + `record_biometric_use()` | 12(3)(a)–(d), for Annex III point 1(a) systems | One record per use with: start and end of the use (a), the reference database (b), the input data that led to a match (c), the natural persons who verified the result (d, see Article 14(5)) |
| Hash chain, Merkle roots, signed manifest, `verify()` | — (engineering) | Makes the log tamper-evident. This is a design choice, not a requirement of Article 12 |

The tags live inside the hashed payload, so they are covered by the hash chain like any other field;
the bundle schema (`1.2`) and the standalone verifier are unchanged.

## What this mode does not cover

- **Classification.** It does not decide whether a system is high-risk (Article 6, Annexes I and III).
- **Keeping the logs.** Retention for six months or more is up to your sink/storage; see above.
- **Events that do not pass through the proxy.** Article 12(1) is a property of the whole system. Only
  calls made through the `VerifiableAgent` proxy, or through attributes listed in `instrument_attrs`,
  are recorded. Calls made with the `@audited` decorator or directly on an `AuditTracer` are recorded
  but not tagged.
- **Choosing purposes.** The default tags every event with all three Article 12(2) purposes; it does
  not establish that an event is relevant to any of them. Narrow the map to what your system needs.
- **Article 14(5).** The two-person verification for Annex III point 1(a) systems is not enforced;
  `record_biometric_use` records who verified, it does not count them.
- **Financial institutions.** Article 19(2) and the second subparagraph of Article 26(6) (logs kept
  as part of financial-services documentation) are not handled.
- **Personal data.** Data-protection duties are out of scope. A `PIIRedactor` passed to the
  `VerifiableAgent` applies to call data and to `record_biometric_use`; the lifetime records hold only
  this mode's own configuration and skip it, so a custom detector cannot rewrite the recorded legal
  basis. A redactor set on the `AuditTracer` itself still applies to every record.
- **Truncation of the tail.** Removing the last records is detected only on an exported bundle, through
  the signed manifest's `record_count`. Only pinned `trusted_keys` also rule out a truncated bundle
  re-signed with another key: without them `verify` never returns `MATCH`. A bare list of records
  (`AuditTracer.verify_chain`) or a `FileAuditSink` file has no such anchor.
- **Trusted time.** Timestamps come from the host clock; there is no RFC 3161 timestamping (#958).
