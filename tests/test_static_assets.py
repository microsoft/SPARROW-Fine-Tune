"""The browser assets must at least parse.

A syntax error in one of these files is silent: Flask serves it happily, the
network tab shows 200, and the page simply does nothing. That is exactly what
happened — an escaped newline in a confirm() message was written to disk as a
real newline, the string literal spanned three lines, and the whole review
editor stopped running while the server kept returning perfectly good data.

There is no JavaScript parser here, so this checks the one class of damage that
actually occurred and is invisible on inspection: a quoted string that is never
closed before the end of its line.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Located through the package, not by walking up from this file: the suite is
# also run from a copy of tests/ against an installed sparrow_web, where a
# relative guess finds an empty directory and every check below silently skips.
import sparrow_web

_PACKAGE = Path(sparrow_web.__file__).resolve().parent
STATIC = _PACKAGE / "static"
TEMPLATES = _PACKAGE / "templates"


def unterminated_string_lines(source: str) -> list[tuple[int, str]]:
    """Line numbers holding a quote that is never closed on that line.

    Walks the text tracking whether it is inside a string, a line comment or a
    block comment, so an apostrophe in prose and a quote in a comment are both
    ignored. Template literals are skipped deliberately: they are allowed to
    span lines, and nothing here uses them.
    """
    bad: list[tuple[int, str]] = []
    for number, line in enumerate(source.splitlines(), 1):
        quote = None
        previous = ""            # last significant character, for the regex test
        i = 0
        while i < len(line):
            ch = line[i]
            if quote:
                if ch == "\\":
                    i += 2
                    continue
                if ch == quote:
                    quote = None
                i += 1
                continue

            if ch in "\"'":
                quote = ch
            elif ch == "/" and i + 1 < len(line) and line[i + 1] in "/*":
                break                       # comment: the rest of the line is prose
            elif ch == "/" and (previous == "" or previous in "=(,:[!&|?{};+-*%~^<>"):
                # A regex literal, not division — and it may legitimately contain
                # a quote, as escapeHtml's /[&<>'"]/g does. Consume it whole.
                i += 1
                in_class = False
                while i < len(line):
                    c = line[i]
                    if c == "\\":
                        i += 2
                        continue
                    if c == "[":
                        in_class = True
                    elif c == "]":
                        in_class = False
                    elif c == "/" and not in_class:
                        break
                    i += 1

            if not ch.isspace():
                previous = ch
            i += 1
        if quote:
            bad.append((number, line.strip()))
    return bad


def js_files() -> list[Path]:
    return sorted(STATIC.glob("*.js"))


def test_there_are_scripts_to_check():
    """Guard against this suite silently passing because the glob broke."""
    assert {p.name for p in js_files()} >= {"annotate.js", "picker.js"}


@pytest.mark.parametrize("path", js_files(), ids=lambda p: p.name)
def test_no_string_literal_runs_off_the_end_of_its_line(path):
    bad = unterminated_string_lines(path.read_text(encoding="utf-8"))
    assert not bad, "\n".join(f"{path.name}:{n}: {text}" for n, text in bad)


@pytest.mark.parametrize("path", sorted(TEMPLATES.glob("*.html")),
                         ids=lambda p: p.name)
def test_inline_scripts_are_intact_too(path):
    """Templates carry inline JS, and it breaks the same way."""
    text = path.read_text(encoding="utf-8")
    inside, chunks = False, []
    for line in text.splitlines(keepends=True):
        low = line.lower()
        if "<script" in low:
            inside = True
            continue
        if "</script" in low:
            inside = False
        chunks.append(line if inside else "\n")   # keep line numbers aligned
    bad = unterminated_string_lines("".join(chunks))
    assert not bad, "\n".join(f"{path.name}:{n}: {text}" for n, text in bad)


class TestTheCheckItself:
    """A detector that cannot fail is not a detector."""

    def test_it_catches_the_bug_that_prompted_it(self):
        broken = "if (!confirm('Remove it?\n\nAre you sure?')) return;"
        assert unterminated_string_lines(broken)

    def test_it_accepts_a_correctly_escaped_newline(self):
        fine = r"if (!confirm('Remove it?\n\nAre you sure?')) return;"
        assert not unterminated_string_lines(fine)

    @pytest.mark.parametrize("line", [
        """const s = "he said 'hi'";""",
        """const s = 'it\\'s fine';""",
        "// an apostrophe in a comment: don't flag it",
        "const s = 'a' + \"b\";",
        "/* a block comment's apostrophe */",
        # The case that produced the first false positive: a quote inside a
        # regex literal, which escapeHtml uses.
        """return String(s).replace(/[&<>'\"]/g, function (c) {""",
        "const halved = total / 2, name = 'ok';",
    ])
    def test_it_does_not_cry_wolf(self, line):
        assert not unterminated_string_lines(line)
