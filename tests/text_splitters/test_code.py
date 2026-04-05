"""Tests for CodeSplitter."""

import pytest

from synapsekit.text_splitters import CodeSplitter

# ------------------------------------------------------------------ #
# Initialization tests
# ------------------------------------------------------------------ #


def test_code_splitter_default_params():
    """Test default parameter values."""
    s = CodeSplitter()
    assert s.language == "python"
    assert s.chunk_size == 512
    assert s.chunk_overlap == 50


def test_code_splitter_custom_params():
    """Test custom parameter values."""
    s = CodeSplitter(language="javascript", chunk_size=256, chunk_overlap=20)
    assert s.language == "javascript"
    assert s.chunk_size == 256
    assert s.chunk_overlap == 20


def test_code_splitter_invalid_language():
    """Test that invalid language raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported language"):
        CodeSplitter(language="haskell")


def test_code_splitter_invalid_chunk_size():
    """Test that chunk_size must be positive."""
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        CodeSplitter(chunk_size=0)

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        CodeSplitter(chunk_size=-1)


def test_code_splitter_invalid_chunk_overlap_negative():
    """Test that chunk_overlap cannot be negative."""
    with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
        CodeSplitter(chunk_overlap=-1)


def test_code_splitter_invalid_chunk_overlap_too_large():
    """Test that chunk_overlap must be less than chunk_size."""
    with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
        CodeSplitter(chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
        CodeSplitter(chunk_size=100, chunk_overlap=150)


# ------------------------------------------------------------------ #
# Supported languages
# ------------------------------------------------------------------ #


@pytest.mark.parametrize(
    "language",
    ["python", "javascript", "typescript", "go", "rust", "java", "cpp"],
)
def test_code_splitter_supported_languages(language):
    """Test all supported languages can be instantiated."""
    s = CodeSplitter(language=language)
    assert s.language == language


# ------------------------------------------------------------------ #
# Basic splitting tests
# ------------------------------------------------------------------ #


def test_code_splitter_empty():
    """Test handling of empty code."""
    s = CodeSplitter()
    assert s.split("") == []
    assert s.split("   ") == []
    assert s.split("\n\n") == []


def test_code_splitter_small_code():
    """Test code smaller than chunk_size."""
    s = CodeSplitter(chunk_size=100)
    code = "x = 1"
    assert s.split(code) == ["x = 1"]


def test_code_splitter_exact_chunk_size():
    """Test code exactly matching chunk_size."""
    code = "x = " + "a" * 100
    s = CodeSplitter(chunk_size=104, chunk_overlap=0)
    result = s.split(code)
    assert len(result) == 1
    assert result[0] == code


# ------------------------------------------------------------------ #
# Python-specific tests
# ------------------------------------------------------------------ #


def test_code_splitter_python_functions():
    """Test splitting Python code on function boundaries."""
    code = "def foo():\n    return 1\n\ndef bar():\n    return 2\n\ndef baz():\n    return 3"
    s = CodeSplitter(language="python", chunk_size=50, chunk_overlap=0)
    result = s.split(code)
    assert len(result) >= 2
    # Each chunk should contain at most chunk_size characters
    assert all(len(c) <= 50 for c in result)


def test_code_splitter_python_classes():
    """Test splitting Python code on class boundaries."""
    code = "class Foo:\n    pass\n\nclass Bar:\n    pass\n\nclass Baz:\n    pass"
    s = CodeSplitter(language="python", chunk_size=40, chunk_overlap=0)
    result = s.split(code)
    assert len(result) >= 2
    assert all(len(c) <= 40 for c in result)


def test_code_splitter_python_mixed():
    """Test Python code with classes and functions."""
    code = "class MyClass:\n    def method(self):\n        pass\n\ndef standalone():\n    pass"
    s = CodeSplitter(language="python", chunk_size=60, chunk_overlap=0)
    result = s.split(code)
    assert len(result) >= 1
    assert all(len(c) <= 60 for c in result)


# ------------------------------------------------------------------ #
# JavaScript-specific tests
# ------------------------------------------------------------------ #


def test_code_splitter_javascript_functions():
    """Test splitting JavaScript code on function boundaries."""
    code = (
        "function foo() {\n    return 1;\n}\nfunction bar() {\n    return 2;\n}\nconst x = () => 3;"
    )
    s = CodeSplitter(language="javascript", chunk_size=50, chunk_overlap=0)
    result = s.split(code)
    assert len(result) >= 2
    assert all(len(c) <= 50 for c in result)


def test_code_splitter_javascript_classes():
    """Test splitting JavaScript code on class boundaries."""
    code = "class Foo {\n    constructor() {}\n}\nclass Bar {\n    constructor() {}\n}"
    s = CodeSplitter(language="javascript", chunk_size=50, chunk_overlap=0)
    result = s.split(code)
    assert len(result) >= 2


# ------------------------------------------------------------------ #
# Other language tests
# ------------------------------------------------------------------ #


def test_code_splitter_go():
    """Test splitting Go code."""
    code = "func foo() int {\n    return 1\n}\nfunc bar() int {\n    return 2\n}"
    s = CodeSplitter(language="go", chunk_size=40, chunk_overlap=0)
    result = s.split(code)
    assert len(result) >= 2


def test_code_splitter_rust():
    """Test splitting Rust code."""
    code = "fn foo() -> i32 {\n    1\n}\nfn bar() -> i32 {\n    2\n}"
    s = CodeSplitter(language="rust", chunk_size=40, chunk_overlap=0)
    result = s.split(code)
    assert len(result) >= 2


def test_code_splitter_java():
    """Test splitting Java code."""
    code = "public class Foo {\n    void bar() {}\n}\npublic class Baz {\n    void qux() {}\n}"
    s = CodeSplitter(language="java", chunk_size=40, chunk_overlap=0)
    result = s.split(code)
    assert len(result) >= 2


def test_code_splitter_cpp():
    """Test splitting C++ code."""
    code = (
        "class Foo {\npublic:\n    void bar() {}\n};\nclass Baz {\npublic:\n    void qux() {}\n};"
    )
    s = CodeSplitter(language="cpp", chunk_size=50, chunk_overlap=0)
    result = s.split(code)
    assert len(result) >= 2


# ------------------------------------------------------------------ #
# Overlap tests
# ------------------------------------------------------------------ #


def test_code_splitter_overlap():
    """Test overlapping chunks."""
    code = "def foo():\n    pass\n\ndef bar():\n    pass\n\ndef baz():\n    pass"
    s = CodeSplitter(language="python", chunk_size=40, chunk_overlap=10)
    result = s.split(code)

    assert len(result) >= 2
    if len(result) >= 2:
        # Second chunk should start with tail of first
        assert result[1].startswith(result[0][-10:])


def test_code_splitter_no_overlap():
    """Test with zero overlap."""
    code = "def foo():\n    pass\n\ndef bar():\n    pass"
    s = CodeSplitter(language="python", chunk_size=30, chunk_overlap=0)
    result = s.split(code)

    assert len(result) >= 1
    assert all(len(c) <= 30 for c in result)


# ------------------------------------------------------------------ #
# Edge cases
# ------------------------------------------------------------------ #


def test_code_splitter_long_single_function():
    """Test function longer than chunk_size."""
    code = "def foo():\n    " + "x = 1\n" * 100
    s = CodeSplitter(language="python", chunk_size=50, chunk_overlap=0)
    result = s.split(code)

    assert len(result) >= 2
    assert all(len(c) <= 50 for c in result)


def test_code_splitter_comments():
    """Test code with comments."""
    code = "# Comment 1\ndef foo():\n    pass\n\n# Comment 2\ndef bar():\n    pass"
    s = CodeSplitter(language="python", chunk_size=50, chunk_overlap=0)
    result = s.split(code)
    assert len(result) >= 1


def test_code_splitter_no_separators():
    """Test code without language-specific separators."""
    code = "x = 1; y = 2; z = 3"
    s = CodeSplitter(language="python", chunk_size=5, chunk_overlap=0)
    result = s.split(code)

    assert len(result) >= 2
    assert all(len(c) <= 5 for c in result)


def test_code_splitter_whitespace_handling():
    """Test code with excessive whitespace."""
    code = "\n\n\ndef foo():\n\n\n    pass\n\n\n"
    s = CodeSplitter(language="python", chunk_size=50, chunk_overlap=0)
    result = s.split(code)

    assert len(result) == 1
    assert result[0].startswith("def foo():")


def test_code_splitter_unicode():
    """Test code with unicode characters."""
    code = "def greet():\n    return 'こんにちは'"
    s = CodeSplitter(language="python", chunk_size=100)
    result = s.split(code)
    assert len(result) == 1
    assert "こんにちは" in result[0]


# ------------------------------------------------------------------ #
# Metadata support
# ------------------------------------------------------------------ #


def test_code_splitter_metadata():
    """Test split_with_metadata."""
    code = "def foo():\n    pass\n\ndef bar():\n    pass"
    s = CodeSplitter(language="python", chunk_size=30, chunk_overlap=0)

    result = s.split_with_metadata(code, {"file": "test.py"})

    assert len(result) >= 1
    assert all(chunk["metadata"]["file"] == "test.py" for chunk in result)
    for idx, chunk in enumerate(result):
        assert chunk["metadata"]["chunk_index"] == idx


def test_code_splitter_metadata_empty():
    """Test metadata with empty code."""
    s = CodeSplitter()
    result = s.split_with_metadata("", {"file": "test.py"})
    assert result == []


# ------------------------------------------------------------------ #
# Integration tests
# ------------------------------------------------------------------ #


def test_code_splitter_realistic_python_file():
    """Test splitting a realistic Python file."""
    code = (
        "import os\n"
        "import sys\n"
        "\n"
        "class MyClass:\n"
        "    def __init__(self):\n"
        "        self.value = 0\n"
        "\n"
        "    def increment(self):\n"
        "        self.value += 1\n"
        "\n"
        "def helper_function():\n"
        "    return MyClass()\n"
        "\n"
        "def main():\n"
        "    obj = helper_function()\n"
        "    obj.increment()\n"
        "    print(obj.value)"
    )
    s = CodeSplitter(language="python", chunk_size=100, chunk_overlap=10)
    result = s.split(code)

    assert len(result) >= 2
    assert all(len(c) <= 100 for c in result)


# ------------------------------------------------------------------ #
# Top-level imports
# ------------------------------------------------------------------ #


def test_top_level_export():
    """Test that CodeSplitter is exported at top level."""
    import synapsekit

    assert hasattr(synapsekit, "CodeSplitter")
    from synapsekit import CodeSplitter as TopLevelImport

    assert TopLevelImport is CodeSplitter
