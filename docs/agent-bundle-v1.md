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
