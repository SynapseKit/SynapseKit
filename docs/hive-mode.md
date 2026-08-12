# Hive Mode protocol

Hive Mode is an opt-in mechanism for sharing aggregate project conventions.
It is not a shared memory store and it does not upload Markdown files.

## Data flow

1. `HiveClient` selects local Markdown through `MeshPrivacyFilter` and the
   caller's include/exclude rules.
2. `PatternMiner` converts the selected documents into a finite vocabulary of
   framework, tooling, and practice signals. Paths, headings, URLs, names,
   and body text are discarded.
3. The client pseudonymizes the contributor within the selected team or
   community scope.
4. A bounded Laplace mechanism adds local noise and reserves epsilon from the
   client's persisted budget ledger.
5. The resulting envelope is Ed25519 signed. An AES-GCM payload can be used in
   addition to HTTPS when the client and self-hosted aggregator share a key.
6. The aggregator verifies the signature, rejects stale/replayed input, stores
   only the processed envelope, and emits suggestions only after the minimum
   cohort threshold is met.

## Privacy guarantees and limits

The reference implementation uses a bounded vocabulary, local redaction,
scope-specific HMAC pseudonyms, per-client epsilon accounting, contribution
limits, and minimum cohort suppression. Pseudonyms are not identities and must
not be treated as anonymous credentials. Operators should rotate the
pseudonymization secret when a contributor leaves a scope.

Differential privacy protects an individual observation under the configured
mechanism and budget; it does not protect a user who intentionally submits
highly identifying pattern choices. Teams should review the vocabulary before
enabling community sharing and should use a trusted key registry plus HTTPS
for remote deployments.

## Self-hosting

Install the optional service dependencies:

```text
pip install synapsekit[hive]
```

Create an application with `create_hive_app(HiveAggregator(...))` and run it
with Uvicorn. Use API keys or a custom request authorizer for non-loopback
deployments. SQLite is the reference store; production deployments may supply
another `HiveStore` implementation.

The service exposes `/healthz`, `/v1/contributions`, `/v1/suggestions`,
`/v1/withdraw`, `/v1/transparency`, and `/dashboard`. Suggestions are always
aggregate-only and the dashboard reports what the contributor's local pipeline
selected after privacy processing.

## Governance

The registry and aggregator are self-hostable and transport-neutral. A hosted
community index may be deployed separately, but the SDK does not require a
SynapseKit-operated service. Monetization, payment processing, and publisher
marketplace policy are outside this protocol.
