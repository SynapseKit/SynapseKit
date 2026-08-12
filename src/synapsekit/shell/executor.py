"""Direct subprocess execution for shell plans."""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, cast

from .lexer import split_shell_commands
from .types import CommandResult, ShellCommand, ShellContext


class ShellExecutionError(RuntimeError):
    """Raised when a command cannot be parsed or started."""


def parse_commands(text: str, *, shell: str = "bash") -> list[ShellCommand]:
    """Parse shell operators into direct argv commands."""

    result: list[ShellCommand] = []
    for raw, connector in split_shell_commands(text, shell=shell):
        try:
            argv = shlex.split(raw, posix=shell != "powershell")
        except ValueError as exc:
            raise ShellExecutionError(f"invalid shell command: {exc}") from exc
        if shell == "powershell":
            argv = [_strip_power_shell_quotes(value) for value in argv]
        if not argv:
            continue
        result.append(
            ShellCommand(tuple(argv), raw, cast(Literal["", "&&", "||", ";", "|"], connector or ""))
        )
    return result


def _strip_power_shell_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


class DirectShellExecutor:
    """Run argv directly and keep subprocesses bounded and collectable."""

    def __init__(self, *, timeout: float = 30.0, max_output_bytes: int = 1_000_000) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")
        self.timeout = timeout
        self.max_output_bytes = max_output_bytes

    async def run_text(self, text: str, context: ShellContext) -> list[CommandResult]:
        commands = parse_commands(text, shell=context.shell.value)
        return await self.run_commands(commands, context)

    async def run_commands(
        self,
        commands: Iterable[ShellCommand],
        context: ShellContext,
    ) -> list[CommandResult]:
        command_list = list(commands)
        results: list[CommandResult] = []
        index = 0
        previous: CommandResult | None = None
        while index < len(command_list):
            command = command_list[index]
            if command.connector == "&&" and previous is not None and not previous.ok:
                result = self._skipped(command)
                results.append(result)
                previous = result
                index += 1
                continue
            if command.connector == "||" and previous is not None and previous.ok:
                result = self._skipped(command)
                results.append(result)
                previous = result
                index += 1
                continue
            if command.connector == "|":
                # A pipe group starts at the previous command. The first
                # result was already emitted, so execute the group only when
                # the parser supplied a leading command with no result.
                index += 1
                continue

            group = [command]
            next_index = index + 1
            while next_index < len(command_list) and command_list[next_index].connector == "|":
                group.append(command_list[next_index])
                next_index += 1
            if len(group) > 1:
                group_results = await self._run_pipeline(group, context)
                results.extend(group_results)
                previous = group_results[-1]
            else:
                previous = await self._run_one(command, context)
                results.append(previous)
            index = next_index
        return results

    async def _run_one(self, command: ShellCommand, context: ShellContext) -> CommandResult:
        started = time.monotonic()
        argv = list(command.argv)
        try:
            process = await self._start(argv, context)
            stdout, stderr, _ = await asyncio.wait_for(self._drain(process), timeout=self.timeout)
        except asyncio.TimeoutError:
            if "process" in locals() and process.returncode is None:
                process.kill()
                await process.wait()
            return CommandResult(
                command=command.raw,
                stdout="",
                stderr=f"command timed out after {self.timeout:g}s",
                exit_code=None,
                duration_seconds=time.monotonic() - started,
                timed_out=True,
            )
        except OSError as exc:
            return CommandResult(
                command=command.raw,
                stdout="",
                stderr=str(exc),
                exit_code=None,
                duration_seconds=time.monotonic() - started,
            )
        return CommandResult(
            command=command.raw,
            stdout=_decode(stdout, self.max_output_bytes),
            stderr=_decode(stderr, self.max_output_bytes),
            exit_code=process.returncode,
            duration_seconds=time.monotonic() - started,
        )

    async def _run_pipeline(
        self,
        commands: list[ShellCommand],
        context: ShellContext,
    ) -> list[CommandResult]:
        """Execute a finite pipeline without giving a shell an input string.

        Stages are intentionally bounded and run sequentially. This is a
        conservative implementation that preserves the observable pipeline
        result while avoiding a shell interpreter and unbounded pipe pumps:
        each stage's output is streamed under a hard byte cap so an unbounded
        producer (``yes | head``) is terminated at the cap rather than hanging.
        """

        input_data = b""
        results: list[CommandResult] = []
        for position, command in enumerate(commands):
            started = time.monotonic()
            try:
                process = await self._start(list(command.argv), context, pipe_stdin=True)
                stdout, stderr, truncated = await asyncio.wait_for(
                    self._drain(process, input_data), timeout=self.timeout
                )
            except asyncio.TimeoutError:
                if "process" in locals() and process.returncode is None:
                    process.kill()
                    await process.wait()
                results.append(
                    CommandResult(
                        command=command.raw,
                        stdout="",
                        stderr=f"pipeline stage timed out after {self.timeout:g}s",
                        exit_code=None,
                        duration_seconds=time.monotonic() - started,
                        timed_out=True,
                    )
                )
                break
            except OSError as exc:
                results.append(
                    CommandResult(
                        command=command.raw,
                        stdout="",
                        stderr=str(exc),
                        exit_code=None,
                        duration_seconds=time.monotonic() - started,
                    )
                )
                break
            output = _decode(stdout, self.max_output_bytes)
            error = _decode(stderr, self.max_output_bytes)
            results.append(
                CommandResult(
                    command=command.raw,
                    stdout=output if position == len(commands) - 1 else "",
                    stderr=error,
                    exit_code=process.returncode,
                    duration_seconds=time.monotonic() - started,
                )
            )
            input_data = stdout[: self.max_output_bytes]
            # A stage killed for exceeding the output cap is expected, not a
            # genuine failure: feed its (truncated) output to the next stage
            # instead of aborting the pipeline.
            if process.returncode != 0 and not truncated:
                break
        return results

    async def _drain(
        self,
        process: asyncio.subprocess.Process,
        input_data: bytes = b"",
    ) -> tuple[bytes, bytes, bool]:
        """Feed stdin and read stdout/stderr under a hard byte cap.

        Unlike ``Process.communicate`` (which buffers the child's entire
        output into memory before any limit is applied), this reads
        incrementally and kills the child as soon as either stream exceeds
        ``max_output_bytes``. That keeps memory bounded and prevents an
        unbounded producer from hanging a pipeline. Returns
        ``(stdout, stderr, overflowed)`` where ``overflowed`` is ``True`` when
        the cap terminated the process.
        """

        limit = self.max_output_bytes
        overflowed = False

        async def _read(stream: asyncio.StreamReader | None) -> bytes:
            nonlocal overflowed
            if stream is None:
                return b""
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = await stream.read(65536)
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > limit:
                    overflowed = True
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
                    break
            return b"".join(chunks)

        async def _feed() -> None:
            stdin = process.stdin
            if stdin is None:
                return
            with contextlib.suppress(BrokenPipeError, ConnectionResetError, RuntimeError, OSError):
                if input_data:
                    stdin.write(input_data)
                    await stdin.drain()
                stdin.close()

        stdout, stderr, _ = await asyncio.gather(
            _read(process.stdout), _read(process.stderr), _feed()
        )
        await process.wait()
        return stdout, stderr, overflowed

    async def _start(
        self,
        argv: list[str],
        context: ShellContext,
        *,
        pipe_stdin: bool = False,
    ) -> asyncio.subprocess.Process:
        stdin = asyncio.subprocess.PIPE if pipe_stdin else asyncio.subprocess.DEVNULL
        try:
            return await asyncio.create_subprocess_exec(
                *argv,
                cwd=context.cwd,
                env=dict(os.environ),
                stdin=stdin,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            # Common built-ins have no standalone executable on Windows. They
            # are still invoked with a fixed argv boundary and only after the
            # command has passed the same analyzer as every other command.
            if os.name == "nt" and argv and argv[0].casefold() in {"echo", "dir"}:
                safe_text = " ".join(shlex.quote(value) for value in argv[1:])
                return await asyncio.create_subprocess_exec(
                    "cmd.exe",
                    "/d",
                    "/c",
                    argv[0],
                    safe_text,
                    cwd=context.cwd,
                    env=dict(os.environ),
                    stdin=stdin,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            raise

    def _skipped(self, command: ShellCommand) -> CommandResult:
        return CommandResult(
            command=command.raw,
            stdout="",
            stderr="skipped by conditional connector",
            exit_code=None,
            duration_seconds=0.0,
            skipped=True,
        )


def _decode(value: bytes | None, limit: int) -> str:
    if not value:
        return ""
    clipped = value[:limit]
    suffix = "\n[output truncated]" if len(value) > limit else ""
    return clipped.decode(errors="replace") + suffix


async def git_diff(cwd: str | Path) -> str:
    """Capture a bounded post-execution working-tree diff."""

    def _read() -> str:
        try:
            completed = subprocess.run(
                ["git", "diff", "--stat", "--no-ext-diff"],
                cwd=Path(cwd),
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        return completed.stdout[:16_384]

    return await asyncio.to_thread(_read)
