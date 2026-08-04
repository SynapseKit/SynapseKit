"""Tests for the ambient source opt-out ignore file."""

from __future__ import annotations

from pathlib import Path

from synapsekit.ambient.privacy import load_disabled_sources


def test_missing_file_returns_empty_set(tmp_path: Path) -> None:
    assert load_disabled_sources(tmp_path / "nonexistent.ignore") == set()


def test_none_path_returns_empty_set() -> None:
    assert load_disabled_sources(None) == set()


def test_parses_source_names_and_skips_comments_and_blanks(tmp_path: Path) -> None:
    ignore_file = tmp_path / "ambient.ignore"
    ignore_file.write_text("# comment\n\nterminal\n\n# another comment\n", encoding="utf-8")

    assert load_disabled_sources(ignore_file) == {"terminal"}


def test_multiple_sources(tmp_path: Path) -> None:
    ignore_file = tmp_path / "ambient.ignore"
    ignore_file.write_text("terminal\ngit\n", encoding="utf-8")

    assert load_disabled_sources(ignore_file) == {"terminal", "git"}
