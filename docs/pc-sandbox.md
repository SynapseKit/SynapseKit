# Sandboxed PC Twin

`PCSandbox` provides a reviewable working copy for destructive agent tasks. The
agent operates on the copy, while the original host tree remains unchanged
until a diff has been evaluated and explicitly applied.

## Basic flow

```python
from synapsekit.sandbox import CallableEvalGate, PCSandbox


async with PCSandbox(base=".", backend="docker", network="none") as sandbox:
    result = await sandbox.environment.exec(("python", "-c", "open('note.txt', 'w').write('ok')"))
    diff = await sandbox.diff()
    receipt = await sandbox.evaluate(
        CallableEvalGate(lambda bundle, _environment: len(bundle.changes) == 1),
        diff,
    )
    if receipt.passed:
        await sandbox.apply(diff, receipt)
```

`PCSandbox.snapshot()` yields the environment directly. A process that needs
to keep a session for CLI review can use `PCSandbox.start()` and later
`PCSandbox.attach(session_id=...)`.

## Safety properties

- The default network policy is `none`; network access must be selected
  explicitly.
- Docker containers drop Linux capabilities, use a read-only root filesystem,
  mount only the sandbox work tree read-write, apply a PID limit, and disable
  privilege escalation.
- Paths in a diff are normalized host-relative paths. Absolute paths,
  traversal, duplicate operations, escaping symlinks, and unsupported special
  files are rejected.
- Applying a diff performs a preflight conflict check against the host, writes
  through temporary files, creates a sibling journal/backups, verifies hashes,
  and rolls back on an apply error.
- `EvalReceipt` is bound to the exact diff digest. A receipt for another bundle
  cannot authorize an apply.
- Host credential locations are excluded from the copied tree by default
  (`.ssh`, cloud credential directories, package-manager credential files,
  `.env`, and `.synapsekit`). Use explicit include/exclude rules for a more
  restricted task scope.

## Marketplace agents and audit

`PCSandbox` implements the marketplace `AgentSandbox` contract. Pass it to an
installed agent and its verified source tree is copied into the sandbox's
private `.synapsekit/agents` runtime area; it is never mounted directly from
the host and is excluded from the host diff. The manifest entrypoint is run as
`python -I ENTRYPOINT --prompt PROMPT` without a shell. It must therefore use
that argv contract.

Sandbox lifecycle calls, command results, and computer-use observation/action
metadata are added to the environment's hash-chained audit trace. Typed text
and screenshot bytes are not copied into those events. To export it:

```python
from synapsekit.audit import SigningPolicy

path = sandbox.environment.export_audit_bundle(
    "sandbox.audit.zip",
    SigningPolicy.ed25519(),
)
```

## Backends

`docker` is the production backend when Docker is available. `orbstack` uses
the Docker-compatible runtime on macOS. `fake` is intentionally a local
process backend for deterministic tests and development only; it is not a
security boundary.

`lima` and `firecracker` probe for their host binaries but fail closed until a
caller supplies the VM/kernel/rootfs/device policy required for a safe mount
and network configuration. They must not silently fall back to an unisolated
local process.

The current filesystem overlay is a deterministic materialized copy. Backends
report `native_cow=False` unless they provide a native copy-on-write layer, so
the 50 GB / 30 second native-CoW acceptance target must be validated by the
runtime-specific integration work rather than inferred from the generic copy.

## CLI

```text
synapse sandbox spawn --base . --backend docker --network none
synapse sandbox diff SESSION_ID --output changes.zip
synapse sandbox apply changes.zip --receipt receipt.json --yes
synapse sandbox discard SESSION_ID
```

`apply` requires both a receipt and `--yes`. Review the generated diff before
authorizing it. The CLI stores session metadata under
`~/.synapsekit/sandboxes` unless `--state-dir` is supplied.
