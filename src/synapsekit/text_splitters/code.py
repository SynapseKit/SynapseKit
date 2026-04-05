from __future__ import annotations

from .base import BaseSplitter
from .recursive import RecursiveCharacterTextSplitter

# Language-specific separator lists, ordered from largest to smallest unit
_LANGUAGE_SEPARATORS: dict[str, list[str]] = {
    "python": [
        "\nclass ",
        "\ndef ",
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ],
    "javascript": [
        "\nclass ",
        "\nfunction ",
        "\nconst ",
        "\nlet ",
        "\nvar ",
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ],
    "typescript": [
        "\nclass ",
        "\ninterface ",
        "\nfunction ",
        "\nconst ",
        "\nlet ",
        "\nvar ",
        "\ntype ",
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ],
    "go": [
        "\nfunc ",
        "\ntype ",
        "\nvar ",
        "\nconst ",
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ],
    "rust": [
        "\nfn ",
        "\nimpl ",
        "\nstruct ",
        "\nenum ",
        "\ntrait ",
        "\nmod ",
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ],
    "java": [
        "\nclass ",
        "\ninterface ",
        "\nenum ",
        "\npublic ",
        "\nprivate ",
        "\nprotected ",
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ],
    "cpp": [
        "\nclass ",
        "\nstruct ",
        "\nenum ",
        "\nnamespace ",
        "\nvoid ",
        "\nint ",
        "\nfloat ",
        "\ndouble ",
        "\nchar ",
        "\nauto ",
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ],
}

_VALID_LANGUAGES = list(_LANGUAGE_SEPARATORS.keys())


class CodeSplitter(BaseSplitter):
    """Split source code using language-aware separators.

    Splits code into chunks preserving logical structures like classes,
    functions, and methods. Falls back to ``RecursiveCharacterTextSplitter``
    logic with language-specific separators.
    """

    def __init__(
        self,
        language: str = "python",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> None:
        """
        Initialize the CodeSplitter.

        Args:
            language: Programming language for separator selection.
                One of: python, javascript, typescript, go, rust, java, cpp.
            chunk_size: Maximum chunk size in characters.
            chunk_overlap: Number of characters to overlap between chunks.
        """
        if language not in _VALID_LANGUAGES:
            raise ValueError(
                f"Unsupported language: {language!r}. Valid options: {_VALID_LANGUAGES}"
            )
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.language = language
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=_LANGUAGE_SEPARATORS[language],
        )

    def split(self, text: str) -> list[str]:
        """
        Split code into chunks preserving logical structures.

        Args:
            text: Source code to split.

        Returns:
            List of code chunks, each at most chunk_size characters.
        """
        return self._splitter.split(text)
