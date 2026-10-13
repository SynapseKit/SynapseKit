from __future__ import annotations

import argparse

from synapsekit.cli.memory import run_memory
from synapsekit.memory import MemoryPatch, PatchStore


def test_memory_cli_review_empty(tmp_path, capsys) -> None:
    store_path = tmp_path / "patches.jsonl"
    args = argparse.Namespace(
        memory_command="review",
        patch_id=None,
        store_path=str(store_path),
    )
    run_memory(args)
    captured = capsys.readouterr()
    assert "No pending memory patches to review." in captured.out


def test_memory_cli_review_list(tmp_path, capsys) -> None:
    store_path = tmp_path / "patches.jsonl"
    store = PatchStore(store_path)
    patch = MemoryPatch(
        file_path="CLAUDE.md",
        before_content="A",
        after_content="B",
        unified_diff="diff",
        rationale="Update test",
    )
    store.save(patch)

    args = argparse.Namespace(
        memory_command="review",
        patch_id=None,
        store_path=str(store_path),
    )
    run_memory(args)
    captured = capsys.readouterr()
    assert patch.patch_id in captured.out
    assert "Update test" in captured.out


def test_memory_cli_review_detail(tmp_path, capsys) -> None:
    store_path = tmp_path / "patches.jsonl"
    store = PatchStore(store_path)
    patch = MemoryPatch(
        file_path="CLAUDE.md",
        before_content="A",
        after_content="B",
        unified_diff="diff content",
        rationale="Specific patch description",
    )
    store.save(patch)

    args = argparse.Namespace(
        memory_command="review",
        patch_id=patch.patch_id,
        store_path=str(store_path),
    )
    run_memory(args)
    captured = capsys.readouterr()
    assert f"Patch ID:     {patch.patch_id}" in captured.out
    assert "Specific patch description" in captured.out
    assert "diff content" in captured.out


def test_memory_cli_apply_success(tmp_path, capsys) -> None:
    store_path = tmp_path / "patches.jsonl"
    target_file = tmp_path / "CLAUDE.md"
    target_file.write_text("Hello\n", encoding="utf-8")

    store = PatchStore(store_path)
    patch = MemoryPatch(
        file_path=str(target_file),
        before_content="Hello\n",
        after_content="Hello World\n",
        unified_diff="...",
        rationale="test apply",
    )
    store.save(patch)

    args = argparse.Namespace(
        memory_command="apply",
        patch_id=patch.patch_id,
        store_path=str(store_path),
    )
    run_memory(args)
    captured = capsys.readouterr()
    assert f"Applied patch {patch.patch_id}" in captured.out
    assert target_file.read_text(encoding="utf-8") == "Hello World\n"


def test_memory_cli_revert_success(tmp_path, capsys) -> None:
    store_path = tmp_path / "patches.jsonl"
    target_file = tmp_path / "CLAUDE.md"
    target_file.write_text("Hello World\n", encoding="utf-8")

    store = PatchStore(store_path)
    patch = MemoryPatch(
        file_path=str(target_file),
        before_content="Hello\n",
        after_content="Hello World\n",
        unified_diff="...",
        rationale="test revert",
        status="applied",
    )
    store.save(patch)

    args = argparse.Namespace(
        memory_command="revert",
        patch_id=patch.patch_id,
        store_path=str(store_path),
    )
    run_memory(args)
    captured = capsys.readouterr()
    assert f"Reverted patch {patch.patch_id}" in captured.out
    assert target_file.read_text(encoding="utf-8") == "Hello\n"


def test_memory_cli_log_table(tmp_path, capsys) -> None:
    store_path = tmp_path / "patches.jsonl"
    store = PatchStore(store_path)
    patch = MemoryPatch(
        file_path="CLAUDE.md",
        before_content="A",
        after_content="B",
        unified_diff="diff",
        rationale="History rationale",
        status="reverted",
    )
    store.save(patch)

    args = argparse.Namespace(
        memory_command="log",
        status=None,
        limit=20,
        output_format="table",
        store_path=str(store_path),
    )
    run_memory(args)
    captured = capsys.readouterr()
    assert "History rationale" in captured.out
    assert "reverted" in captured.out


def test_memory_cli_clear_empty(tmp_path, capsys) -> None:
    store_path = tmp_path / "patches.jsonl"
    args = argparse.Namespace(
        memory_command="clear",
        store_path=str(store_path),
        force=False,
    )
    run_memory(args)
    captured = capsys.readouterr()
    assert "No memory patches to clear." in captured.out


def test_memory_cli_clear_force(tmp_path, capsys) -> None:
    store_path = tmp_path / "patches.jsonl"
    store = PatchStore(store_path)
    store.save(
        MemoryPatch(
            file_path="CLAUDE.md",
            before_content="A",
            after_content="B",
            unified_diff="diff",
            rationale="to be cleared",
        )
    )

    args = argparse.Namespace(
        memory_command="clear",
        store_path=str(store_path),
        force=True,
    )
    run_memory(args)
    captured = capsys.readouterr()
    assert "Cleared 1 memory patches." in captured.out
    assert not store_path.exists()
    assert PatchStore(store_path).count() == 0


def test_memory_cli_clear_declined(tmp_path, capsys, monkeypatch) -> None:
    store_path = tmp_path / "patches.jsonl"
    store = PatchStore(store_path)
    store.save(
        MemoryPatch(
            file_path="CLAUDE.md",
            before_content="A",
            after_content="B",
            unified_diff="diff",
            rationale="kept",
        )
    )

    monkeypatch.setattr("builtins.input", lambda _: "n")
    args = argparse.Namespace(
        memory_command="clear",
        store_path=str(store_path),
        force=False,
    )
    run_memory(args)
    captured = capsys.readouterr()
    assert "Aborted." in captured.out
    assert PatchStore(store_path).count() == 1


def test_memory_cli_clear_non_interactive_eof(tmp_path, capsys, monkeypatch) -> None:
    store_path = tmp_path / "patches.jsonl"
    store = PatchStore(store_path)
    store.save(
        MemoryPatch(
            file_path="CLAUDE.md",
            before_content="A",
            after_content="B",
            unified_diff="diff",
            rationale="kept",
        )
    )

    def _raise_eof(_: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", _raise_eof)
    args = argparse.Namespace(
        memory_command="clear",
        store_path=str(store_path),
        force=False,
    )
    run_memory(args)
    captured = capsys.readouterr()
    assert "Aborted." in captured.out
    assert PatchStore(store_path).count() == 1


def test_memory_cli_prune_alias(tmp_path, capsys) -> None:
    store_path = tmp_path / "patches.jsonl"
    store = PatchStore(store_path)
    store.save(
        MemoryPatch(
            file_path="CLAUDE.md",
            before_content="A",
            after_content="B",
            unified_diff="diff",
            rationale="pruned",
        )
    )

    args = argparse.Namespace(
        memory_command="prune",
        store_path=str(store_path),
        force=True,
    )
    run_memory(args)
    captured = capsys.readouterr()
    assert "Cleared 1 memory patches." in captured.out
