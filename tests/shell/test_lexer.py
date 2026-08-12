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


def test_quoted_arg_starting_with_cue_word_stays_shell() -> None:
    # Regression for #930: a quoted argument mid-command that happens to begin
    # with a cue word (find/run/search/...) must remain a shell token, not be
    # reclassified as natural language and stripped out of the command.
    parsed = lex_input('git commit -m "find and fix the bug"')

    assert not parsed.has_natural_language
    assert len(parsed.segments) == 1
    assert parsed.segments[0].kind is SegmentKind.SHELL
    assert parsed.segments[0].text.strip() == 'git commit -m "find and fix the bug"'

    commands = parse_commands('git commit -m "find and fix the bug"', shell=ShellKind.BASH.value)
    assert commands[0].argv == ("git", "commit", "-m", "find and fix the bug")


def test_grep_pattern_with_cue_word_is_preserved() -> None:
    parsed = lex_input('grep "search term" file.txt')
    assert not parsed.has_natural_language

    commands = parse_commands('grep "search term" file.txt', shell=ShellKind.BASH.value)
    assert commands[0].argv == ("grep", "search term", "file.txt")


def test_leading_cue_phrase_is_still_natural_language() -> None:
    # The bare-phrase / cue-word shorthand still applies when the quote opens a
    # command (line start or right after a connector).
    assert lex_input('"find the flaky test"').has_natural_language
    assert lex_input('git push && "run the tests"').has_natural_language


def test_operators_become_direct_argv_commands() -> None:
    commands = parse_commands("echo hello && echo world | sort", shell=ShellKind.BASH.value)

    assert [command.argv for command in commands] == [
        ("echo", "hello"),
        ("echo", "world"),
        ("sort",),
    ]
    assert [command.connector for command in commands] == ["", "&&", "|"]
