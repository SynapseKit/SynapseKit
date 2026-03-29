"""Tests for XMLLoader."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from synapsekit.loaders.xml_loader import XMLLoader


class TestXMLLoader:
    def test_load_simple_xml(self, tmp_path):
        """Test loading a simple XML file with basic text content."""
        f = tmp_path / "simple.xml"
        f.write_text(
            """<?xml version="1.0"?>
<root>
    <title>Hello World</title>
    <content>This is some content</content>
</root>"""
        )
        docs = XMLLoader(str(f)).load()
        assert len(docs) == 1
        assert "Hello World" in docs[0].text
        assert "This is some content" in docs[0].text
        assert docs[0].metadata["source"] == str(f)

    def test_load_with_specific_tags(self, tmp_path):
        """Test extracting text only from specified tags."""
        f = tmp_path / "data.xml"
        f.write_text(
            """<?xml version="1.0"?>
<root>
    <title>Title Text</title>
    <description>Description Text</description>
    <ignore>Should not appear</ignore>
</root>"""
        )
        docs = XMLLoader(str(f), tags=["title", "description"]).load()
        assert len(docs) == 1
        assert "Title Text" in docs[0].text
        assert "Description Text" in docs[0].text
        assert "Should not appear" not in docs[0].text

    def test_load_nested_elements(self, tmp_path):
        """Test extracting text from nested XML elements."""
        f = tmp_path / "nested.xml"
        f.write_text(
            """<?xml version="1.0"?>
<root>
    <article>
        <header>
            <title>Article Title</title>
            <author>John Doe</author>
        </header>
        <body>
            <paragraph>First paragraph</paragraph>
            <paragraph>Second paragraph</paragraph>
        </body>
    </article>
</root>"""
        )
        docs = XMLLoader(str(f)).load()
        assert len(docs) == 1
        assert "Article Title" in docs[0].text
        assert "John Doe" in docs[0].text
        assert "First paragraph" in docs[0].text
        assert "Second paragraph" in docs[0].text

    def test_load_with_nested_tag_filter(self, tmp_path):
        """Test extracting only specific tags from nested structure."""
        f = tmp_path / "nested.xml"
        f.write_text(
            """<?xml version="1.0"?>
<root>
    <article>
        <title>Article Title</title>
        <paragraph>First paragraph</paragraph>
        <paragraph>Second paragraph</paragraph>
        <footer>Footer content</footer>
    </article>
</root>"""
        )
        docs = XMLLoader(str(f), tags=["paragraph"]).load()
        assert len(docs) == 1
        assert "First paragraph" in docs[0].text
        assert "Second paragraph" in docs[0].text
        assert "Article Title" not in docs[0].text
        assert "Footer content" not in docs[0].text

    def test_load_with_attributes(self, tmp_path):
        """Test that element attributes are ignored, only text is extracted."""
        f = tmp_path / "attrs.xml"
        f.write_text(
            """<?xml version="1.0"?>
<root>
    <item id="1" type="important">Item text</item>
    <item id="2">Another item</item>
</root>"""
        )
        docs = XMLLoader(str(f)).load()
        assert len(docs) == 1
        assert "Item text" in docs[0].text
        assert "Another item" in docs[0].text
        # Attributes should not be in text
        assert 'id="1"' not in docs[0].text
        assert 'type="important"' not in docs[0].text

    def test_load_mixed_content(self, tmp_path):
        """Test XML with mixed text and element content."""
        f = tmp_path / "mixed.xml"
        f.write_text(
            """<?xml version="1.0"?>
<root>
    <p>Text before <strong>bold text</strong> and after</p>
</root>"""
        )
        docs = XMLLoader(str(f)).load()
        assert len(docs) == 1
        assert "Text before" in docs[0].text
        assert "bold text" in docs[0].text
        assert "and after" in docs[0].text

    def test_missing_file_raises(self):
        """Test that loading a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="XML file not found"):
            XMLLoader("/nonexistent/path.xml").load()

    def test_malformed_xml_raises(self, tmp_path):
        """Test that malformed XML raises ParseError."""
        f = tmp_path / "bad.xml"
        f.write_text("<root><unclosed>")
        with pytest.raises(ET.ParseError):
            XMLLoader(str(f)).load()

    def test_empty_xml(self, tmp_path):
        """Test loading XML with no text content."""
        f = tmp_path / "empty.xml"
        f.write_text(
            """<?xml version="1.0"?>
<root></root>"""
        )
        docs = XMLLoader(str(f)).load()
        assert len(docs) == 1
        assert docs[0].text == ""

    def test_xml_with_whitespace_only(self, tmp_path):
        """Test that whitespace-only text is properly handled."""
        f = tmp_path / "whitespace.xml"
        f.write_text(
            """<?xml version="1.0"?>
<root>
    <item>   </item>
    <item>
    
    </item>
</root>"""
        )
        docs = XMLLoader(str(f)).load()
        assert len(docs) == 1
        # Should be empty or minimal since we strip whitespace
        assert docs[0].text.strip() == ""

    def test_xml_with_cdata(self, tmp_path):
        """Test XML with CDATA sections."""
        f = tmp_path / "cdata.xml"
        f.write_text(
            """<?xml version="1.0"?>
<root>
    <content><![CDATA[Some <content> with special chars & symbols]]></content>
</root>"""
        )
        docs = XMLLoader(str(f)).load()
        assert len(docs) == 1
        assert "Some <content> with special chars & symbols" in docs[0].text

    def test_multiple_instances_of_same_tag(self, tmp_path):
        """Test extracting multiple instances of the same tag."""
        f = tmp_path / "multi.xml"
        f.write_text(
            """<?xml version="1.0"?>
<root>
    <item>First item</item>
    <item>Second item</item>
    <item>Third item</item>
</root>"""
        )
        docs = XMLLoader(str(f), tags=["item"]).load()
        assert len(docs) == 1
        assert "First item" in docs[0].text
        assert "Second item" in docs[0].text
        assert "Third item" in docs[0].text

    def test_encoding_param(self, tmp_path):
        """Test loading XML with custom encoding."""
        f = tmp_path / "encoded.xml"
        content = """<?xml version="1.0" encoding="utf-8"?>
<root>
    <text>Héllo wörld</text>
</root>"""
        f.write_text(content, encoding="utf-8")
        docs = XMLLoader(str(f), encoding="utf-8").load()
        assert len(docs) == 1
        assert "Héllo wörld" in docs[0].text

    def test_complex_document(self, tmp_path):
        """Test with a more complex, realistic XML document."""
        f = tmp_path / "complex.xml"
        f.write_text(
            """<?xml version="1.0"?>
<library>
    <book id="1">
        <title>The Great Gatsby</title>
        <author>F. Scott Fitzgerald</author>
        <year>1925</year>
        <summary>A novel about the American dream</summary>
    </book>
    <book id="2">
        <title>1984</title>
        <author>George Orwell</author>
        <year>1949</year>
        <summary>A dystopian social science fiction</summary>
    </book>
</library>"""
        )
        docs = XMLLoader(str(f)).load()
        assert len(docs) == 1
        assert "The Great Gatsby" in docs[0].text
        assert "F. Scott Fitzgerald" in docs[0].text
        assert "1984" in docs[0].text
        assert "George Orwell" in docs[0].text

    def test_specific_tags_complex_document(self, tmp_path):
        """Test extracting specific tags from complex document."""
        f = tmp_path / "complex.xml"
        f.write_text(
            """<?xml version="1.0"?>
<library>
    <book id="1">
        <title>The Great Gatsby</title>
        <author>F. Scott Fitzgerald</author>
        <year>1925</year>
        <summary>A novel about the American dream</summary>
    </book>
    <book id="2">
        <title>1984</title>
        <author>George Orwell</author>
        <year>1949</year>
        <summary>A dystopian social science fiction</summary>
    </book>
</library>"""
        )
        docs = XMLLoader(str(f), tags=["title", "author"]).load()
        assert len(docs) == 1
        assert "The Great Gatsby" in docs[0].text
        assert "F. Scott Fitzgerald" in docs[0].text
        assert "1984" in docs[0].text
        assert "George Orwell" in docs[0].text
        # Years and summaries should not be included
        assert "1925" not in docs[0].text
        assert "A novel about the American dream" not in docs[0].text

    def test_nonexistent_tag_filter(self, tmp_path):
        """Test that filtering by non-existent tags returns empty text."""
        f = tmp_path / "data.xml"
        f.write_text(
            """<?xml version="1.0"?>
<root>
    <item>Content</item>
</root>"""
        )
        docs = XMLLoader(str(f), tags=["nonexistent"]).load()
        assert len(docs) == 1
        assert docs[0].text == ""
