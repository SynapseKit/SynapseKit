from __future__ import annotations

from synapsekit.shell import SegmentKind, ShellKind, lex_input, parse_commands


def test_lexer_preserves_shell_and_extracts_natural_language() -> None:
    parsed = lex_input('git push && "open the PR I just pushed"')

    assert parsed.has_natural_language
    assert parsed.segments[0].kind is SegmentKind.SHELL
    assert parsed.segments[0].text.strip() == "git push &&"
    assert parsed.segments[1].kind is SegmentKind.NATURAL_LANGUAGE
    assert parsed.segments[1].text == "open the PR I just pushed"


def test_ordinary_shell_quotes_are_not_reclassified() -> None:
    parsed = lex_input('echo "hello world"')

    assert not parsed.has_natural_language
    assert parsed.segments[0].kind is SegmentKind.SHELL


def test_operators_become_direct_argv_commands() -> None:
    commands = parse_commands("echo hello && echo world | sort", shell=ShellKind.BASH.value)

    assert [command.argv for command in commands] == [
        ("echo", "hello"),
        ("echo", "world"),
        ("sort",),
    ]
    assert [command.connector for command in commands] == ["", "&&", "|"]
