from __future__ import annotations

import pytest

from synapsekit.loaders.base import Document
from synapsekit.loaders.latex import LaTeXLoader


class TestLaTeXLoader:
    # ------------------------------------------------------------------
    # File-level behaviour
    # ------------------------------------------------------------------

    def test_file_not_found(self):
        loader = LaTeXLoader("/nonexistent/file.tex")
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_returns_list_of_documents(self, tmp_path):
        p = tmp_path / "doc.tex"
        p.write_text(r"\documentclass{article}\begin{document}Hello\end{document}")
        docs = LaTeXLoader(str(p)).load()
        assert isinstance(docs, list)
        assert len(docs) == 1
        assert isinstance(docs[0], Document)

    def test_metadata_contains_source(self, tmp_path):
        p = tmp_path / "doc.tex"
        p.write_text("Hello world")
        docs = LaTeXLoader(str(p)).load()
        assert docs[0].metadata["source"] == str(p)

    def test_empty_file_returns_document(self, tmp_path):
        p = tmp_path / "empty.tex"
        p.write_text("")
        docs = LaTeXLoader(str(p)).load()
        assert len(docs) == 1
        assert docs[0].text == ""

    async def test_aload(self, tmp_path):
        p = tmp_path / "doc.tex"
        p.write_text("Hello async world")
        docs = await LaTeXLoader(str(p)).aload()
        assert len(docs) == 1
        assert "async world" in docs[0].text
        assert docs[0].metadata["source"] == str(p)

    # ------------------------------------------------------------------
    # Command stripping
    # ------------------------------------------------------------------

    def test_strips_simple_command(self, tmp_path):
        p = tmp_path / "cmd.tex"
        # \textbf{Hello} — the command AND its argument are consumed by the regex.
        # Plain text outside the command (" world") must survive.
        p.write_text(r"\textbf{Hello} world")
        docs = LaTeXLoader(str(p)).load()
        assert "textbf" not in docs[0].text
        assert "world" in docs[0].text

    def test_strips_command_with_optional_arg(self, tmp_path):
        p = tmp_path / "opt.tex"
        p.write_text(r"\includegraphics[width=0.5\textwidth]{figure.png} caption")
        docs = LaTeXLoader(str(p)).load()
        assert "includegraphics" not in docs[0].text
        assert "caption" in docs[0].text

    def test_strips_standalone_command(self, tmp_path):
        p = tmp_path / "standalone.tex"
        p.write_text(r"Line one \newline Line two")
        docs = LaTeXLoader(str(p)).load()
        assert r"\newline" not in docs[0].text
        assert "Line one" in docs[0].text
        assert "Line two" in docs[0].text

    # ------------------------------------------------------------------
    # Environment stripping
    # ------------------------------------------------------------------

    def test_strips_document_environment(self, tmp_path):
        p = tmp_path / "env.tex"
        p.write_text("\\begin{document}\nSome text\n\\end{document}")
        docs = LaTeXLoader(str(p)).load()
        assert "begin" not in docs[0].text
        assert "end" not in docs[0].text

    def test_strips_equation_environment(self, tmp_path):
        p = tmp_path / "eq.tex"
        p.write_text("Before\n\\begin{equation}\nE = mc^2\n\\end{equation}\nAfter")
        docs = LaTeXLoader(str(p)).load()
        assert "equation" not in docs[0].text
        assert "E = mc^2" not in docs[0].text
        assert "Before" in docs[0].text
        assert "After" in docs[0].text

    def test_strips_enumerate_environment(self, tmp_path):
        p = tmp_path / "list.tex"
        p.write_text(
            "Intro\n\\begin{enumerate}\n\\item First\n\\item Second\n\\end{enumerate}\nEnd"
        )
        docs = LaTeXLoader(str(p)).load()
        assert "enumerate" not in docs[0].text

    # ------------------------------------------------------------------
    # Math stripping
    # ------------------------------------------------------------------

    def test_strips_inline_math(self, tmp_path):
        p = tmp_path / "math.tex"
        p.write_text(r"The value is $x = 5$ in our formula.")
        docs = LaTeXLoader(str(p)).load()
        assert "$" not in docs[0].text
        assert "x = 5" not in docs[0].text
        assert "The value is" in docs[0].text
        assert "in our formula." in docs[0].text

    def test_strips_display_math(self, tmp_path):
        p = tmp_path / "dmath.tex"
        p.write_text("Intro $$E = mc^2$$ conclusion")
        docs = LaTeXLoader(str(p)).load()
        assert "$$" not in docs[0].text
        assert "E = mc^2" not in docs[0].text
        assert "Intro" in docs[0].text
        assert "conclusion" in docs[0].text

    # ------------------------------------------------------------------
    # Section extraction
    # ------------------------------------------------------------------

    def test_extracts_section_titles(self, tmp_path):
        p = tmp_path / "sections.tex"
        p.write_text(
            "\\section{Introduction}\nSome intro text.\n\\section{Methods}\nSome method text.\n"
        )
        docs = LaTeXLoader(str(p)).load()
        assert "sections" in docs[0].metadata
        assert "Introduction" in docs[0].metadata["sections"]
        assert "Methods" in docs[0].metadata["sections"]

    def test_extracts_subsection_titles(self, tmp_path):
        p = tmp_path / "subsec.tex"
        p.write_text("\\subsection{Background}\nContent here.\n")
        docs = LaTeXLoader(str(p)).load()
        assert "Background" in docs[0].metadata["sections"]

    def test_first_section_becomes_title(self, tmp_path):
        p = tmp_path / "title.tex"
        p.write_text("\\section{Overview}\nText.\n\\section{Details}\nMore text.\n")
        docs = LaTeXLoader(str(p)).load()
        assert docs[0].metadata.get("title") == "Overview"

    def test_no_sections_omits_metadata_keys(self, tmp_path):
        p = tmp_path / "nosec.tex"
        p.write_text("Just plain text without any sections.")
        docs = LaTeXLoader(str(p)).load()
        assert "sections" not in docs[0].metadata
        assert "title" not in docs[0].metadata

    # ------------------------------------------------------------------
    # Whitespace normalisation
    # ------------------------------------------------------------------

    def test_collapses_extra_blank_lines(self, tmp_path):
        p = tmp_path / "ws.tex"
        p.write_text("Line one\n\n\n\n\nLine two")
        docs = LaTeXLoader(str(p)).load()
        # Should not have more than two consecutive newlines
        assert "\n\n\n" not in docs[0].text

    def test_strips_leading_trailing_whitespace(self, tmp_path):
        p = tmp_path / "trim.tex"
        p.write_text("   \n\nHello world\n\n   ")
        docs = LaTeXLoader(str(p)).load()
        assert docs[0].text == docs[0].text.strip()

    # ------------------------------------------------------------------
    # Malformed / edge-case LaTeX
    # ------------------------------------------------------------------

    def test_unmatched_brace_does_not_raise(self, tmp_path):
        p = tmp_path / "brace.tex"
        p.write_text(r"Some text with stray { brace")
        docs = LaTeXLoader(str(p)).load()  # should not raise
        assert "Some text" in docs[0].text

    def test_plain_text_passthrough(self, tmp_path):
        p = tmp_path / "plain.tex"
        p.write_text("No LaTeX here at all.")
        docs = LaTeXLoader(str(p)).load()
        assert "No LaTeX here at all." in docs[0].text
