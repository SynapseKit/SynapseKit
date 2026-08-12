# Agent OS Shell

`synapsekit shell` is a local-first hybrid shell wrapper. Keep ordinary shell
syntax as-is and quote a natural-language request when you want SynapseKit to
plan it with project context.

```console
synapsekit shell run 'git status && "find me the relevant test and rerun it"'
synshell '"why is this directory so large?"'
```

The default planner is deliberately deterministic and offline. Applications
can inject an `LLMShellPlanner` backed by an `EdgeRuntime` or another
`BaseLLM`; planning output is strict JSON and always passes through the same
safety policy before execution.

## Safety model

Commands are parsed into argv vectors and launched with direct subprocess
execution; no user or model text is passed through `shell=True`. `&&`, `||`,
`;`, and `|` are parsed by SynapseKit and executed with bounded output and a
timeout.

Destructive operations—such as `rm`, `Remove-Item`, `git reset`, `git clean`,
force pushes, Docker prune, Kubernetes deletes, and Terraform destroy—are
previewed and require confirmation. They also require an Ed25519 signing key:
the shell exports a signed pre-execution audit receipt before launching the
command and a final bundle with the captured result and post-command Git diff.

```console
synapsekit shell keygen ~/.synapsekit/shell/key
synapsekit shell run --signing-key ~/.synapsekit/shell/key 'git clean -fd'
```

`--yes` only skips the interactive answer after a signing key is supplied; it
does not permit unsigned destructive execution. Use `--dry-run` to inspect a
plan without launching any command.

## Context and privacy

The session collects the current directory, shell dialect, bounded Git state,
a small allowlist of non-secret environment metadata, optional KnowledgeMesh
hits, optional Ambient Agent status JSON, and relevant local history.

History lives at `~/.synapsekit/shell/history.sqlite3`. Common secret-shaped
`api_key`, `token`, `secret`, and `password` values are redacted before that
history is written. Translation caching is local and skips destructive-looking
plans.

Enable or disable context explicitly:

```console
synapsekit shell run --mesh-root . '"run the test related to the recent CLI change"'
synapsekit shell run --no-mesh --ambient-status ~/.synapsekit/ambient/status.json 'git status'
synapsekit shell history search cli test
```

## Shell integrations

Source the generated function for your shell:

```console
synapsekit shell init bash
synapsekit shell init zsh
synapsekit shell init fish
synapsekit shell init powershell
```

Each integration defines `synshell` (and `synapse`) as a small wrapper around
`synapsekit shell run`. The standalone `synshell` console script provides the
same one-shot mode and opens a REPL with no command arguments.

## Embedding in an application

```python
import asyncio

from synapsekit.shell import ShellSession


async def main() -> None:
    session = ShellSession(cwd=".")
    result = await session.run('git status && "run tests"')
    print(result.ok, result.plan.summary)


asyncio.run(main())
```

For a production LLM planner, instantiate `LLMShellPlanner` with a configured
local-first `BaseLLM`/`EdgeRuntime` and pass it to `ShellSession(planner=...)`.
The model proposes commands only; it cannot bypass confirmation, signing, or
the direct-execution boundary.
