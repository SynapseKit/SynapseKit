"""Tests for v1.0.0 features: multimodal, image loader, API stability markers."""

from __future__ import annotations

import base64
import struct
import warnings
import zlib
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from synapsekit._api import deprecated, experimental, public_api
from synapsekit.llm.multimodal import AudioContent, ImageContent, MultimodalMessage
from synapsekit.loaders.image import ImageLoader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tiny_png() -> bytes:
    """Create a minimal valid 1x1 red PNG."""

    def chunk(ct: bytes, data: bytes) -> bytes:
        c = ct + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
        + chunk(b"IEND", b"")
    )


@pytest.fixture
def tiny_png_path(tmp_path: Path) -> Path:
    p = tmp_path / "test.png"
    p.write_bytes(make_tiny_png())
    return p


@pytest.fixture
def tiny_png_b64() -> str:
    return base64.b64encode(make_tiny_png()).decode("ascii")


# ===========================================================================
# ImageContent tests
# ===========================================================================


class TestImageContentFromFile:
    def test_loads_file(self, tiny_png_path: Path):
        ic = ImageContent.from_file(tiny_png_path)
        assert ic.source_type == "file"
        assert ic.media_type == "image/png"
        assert len(ic.data) > 0

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            ImageContent.from_file(tmp_path / "nope.png")

    def test_unknown_extension_defaults_png(self, tmp_path: Path):
        p = tmp_path / "image.unknownext"
        p.write_bytes(make_tiny_png())
        ic = ImageContent.from_file(p)
        assert ic.media_type == "image/png"

    def test_jpeg_detection(self, tmp_path: Path):
        p = tmp_path / "photo.jpg"
        p.write_bytes(b"\xff\xd8\xff\xe0")  # fake JPEG header
        ic = ImageContent.from_file(p)
        assert ic.media_type == "image/jpeg"


class TestImageContentFromUrl:
    def test_stores_url(self):
        ic = ImageContent.from_url("https://example.com/img.png")
        assert ic.source_type == "url"
        assert ic.url == "https://example.com/img.png"
        assert ic.data == ""

    def test_custom_media_type(self):
        ic = ImageContent.from_url("https://x.com/a.webp", media_type="image/webp")
        assert ic.media_type == "image/webp"


class TestImageContentFromBase64:
    def test_stores_data(self, tiny_png_b64: str):
        ic = ImageContent.from_base64(tiny_png_b64)
        assert ic.source_type == "base64"
        assert ic.data == tiny_png_b64
        assert ic.media_type == "image/png"

    def test_custom_media_type(self):
        ic = ImageContent.from_base64("abc", media_type="image/gif")
        assert ic.media_type == "image/gif"


class TestImageContentOpenAIFormat:
    def test_base64_format(self, tiny_png_b64: str):
        ic = ImageContent.from_base64(tiny_png_b64)
        fmt = ic.to_openai_format()
        assert fmt["type"] == "image_url"
        assert fmt["image_url"]["url"].startswith("data:image/png;base64,")

    def test_url_format(self):
        ic = ImageContent.from_url("https://example.com/img.png")
        fmt = ic.to_openai_format()
        assert fmt["image_url"]["url"] == "https://example.com/img.png"

    def test_file_format(self, tiny_png_path: Path):
        ic = ImageContent.from_file(tiny_png_path)
        fmt = ic.to_openai_format()
        assert "data:image/png;base64," in fmt["image_url"]["url"]


class TestImageContentAnthropicFormat:
    def test_base64_format(self, tiny_png_b64: str):
        ic = ImageContent.from_base64(tiny_png_b64)
        fmt = ic.to_anthropic_format()
        assert fmt["type"] == "image"
        assert fmt["source"]["type"] == "base64"
        assert fmt["source"]["media_type"] == "image/png"
        assert fmt["source"]["data"] == tiny_png_b64

    def test_url_format(self):
        ic = ImageContent.from_url("https://example.com/img.png")
        fmt = ic.to_anthropic_format()
        assert fmt["source"]["type"] == "url"
        assert fmt["source"]["url"] == "https://example.com/img.png"

    def test_file_format(self, tiny_png_path: Path):
        ic = ImageContent.from_file(tiny_png_path)
        fmt = ic.to_anthropic_format()
        assert fmt["source"]["type"] == "base64"
        assert len(fmt["source"]["data"]) > 0


# ===========================================================================
# AudioContent tests
# ===========================================================================


class TestAudioContentFromFile:
    def test_loads_file(self, tmp_path: Path):
        p = tmp_path / "clip.wav"
        p.write_bytes(b"RIFF" + b"\x00" * 40)
        ac = AudioContent.from_file(p)
        assert ac.media_type == "audio/wav"
        assert len(ac.data) > 0

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            AudioContent.from_file(tmp_path / "nope.wav")


class TestAudioContentFromBase64:
    def test_stores_data(self):
        ac = AudioContent.from_base64("YXVkaW8=", media_type="audio/mp3")
        assert ac.data == "YXVkaW8="
        assert ac.media_type == "audio/mp3"

    def test_default_media_type(self):
        ac = AudioContent.from_base64("YXVkaW8=")
        assert ac.media_type == "audio/wav"


# ===========================================================================
# MultimodalMessage tests
# ===========================================================================


class TestMultimodalMessageOpenAI:
    def test_text_only(self):
        mm = MultimodalMessage(text="Hello")
        msgs = mm.to_openai_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"][0] == {"type": "text", "text": "Hello"}

    def test_text_and_image(self, tiny_png_b64: str):
        ic = ImageContent.from_base64(tiny_png_b64)
        mm = MultimodalMessage(text="Describe", images=[ic])
        msgs = mm.to_openai_messages()
        content = msgs[0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"

    def test_multiple_images(self, tiny_png_b64: str):
        imgs = [ImageContent.from_base64(tiny_png_b64) for _ in range(3)]
        mm = MultimodalMessage(text="Compare", images=imgs)
        content = mm.to_openai_messages()[0]["content"]
        assert len(content) == 4  # 1 text + 3 images

    def test_custom_role(self):
        mm = MultimodalMessage(text="Hi", role="assistant")
        assert mm.to_openai_messages()[0]["role"] == "assistant"


class TestMultimodalMessageAnthropic:
    def test_text_only(self):
        mm = MultimodalMessage(text="Hello")
        msgs = mm.to_anthropic_messages()
        assert msgs[0]["content"][0] == {"type": "text", "text": "Hello"}

    def test_text_and_image(self, tiny_png_b64: str):
        ic = ImageContent.from_base64(tiny_png_b64)
        mm = MultimodalMessage(text="Describe", images=[ic])
        content = mm.to_anthropic_messages()[0]["content"]
        assert len(content) == 2
        assert content[1]["type"] == "image"
        assert content[1]["source"]["type"] == "base64"

    def test_images_only(self, tiny_png_b64: str):
        ic = ImageContent.from_base64(tiny_png_b64)
        mm = MultimodalMessage(images=[ic])
        content = mm.to_anthropic_messages()[0]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "image"


# ===========================================================================
# ImageLoader tests
# ===========================================================================


class TestImageLoaderSync:
    def test_no_llm_returns_metadata_doc(self, tiny_png_path: Path):
        loader = ImageLoader(tiny_png_path)
        docs = loader.load()
        assert len(docs) == 1
        assert docs[0].text == f"[Image: {tiny_png_path}]"
        assert docs[0].metadata["media_type"] == "image/png"
        assert docs[0].metadata["file_size"] > 0

    def test_missing_file_raises(self, tmp_path: Path):
        loader = ImageLoader(tmp_path / "nope.png")
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_with_llm_sync_still_returns_placeholder(self, tiny_png_path: Path):
        mock_llm = AsyncMock()
        loader = ImageLoader(tiny_png_path, llm=mock_llm)
        docs = loader.load()
        assert "[Image:" in docs[0].text


class TestImageLoaderAsync:
    @pytest.mark.asyncio
    async def test_no_llm_async(self, tiny_png_path: Path):
        loader = ImageLoader(tiny_png_path)
        docs = await loader.async_load()
        assert "[Image:" in docs[0].text

    @pytest.mark.asyncio
    async def test_with_mock_llm(self, tiny_png_path: Path):
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value="A small red pixel image.")
        loader = ImageLoader(tiny_png_path, llm=mock_llm)
        docs = await loader.async_load()
        assert docs[0].text == "A small red pixel image."
        assert "description_prompt" in docs[0].metadata

    @pytest.mark.asyncio
    async def test_missing_file_async(self, tmp_path: Path):
        loader = ImageLoader(tmp_path / "nope.png")
        with pytest.raises(FileNotFoundError):
            await loader.async_load()


# ===========================================================================
# API stability markers tests
# ===========================================================================


class TestPublicAPI:
    def test_marks_function(self):
        @public_api
        def my_func():
            return 42

        assert my_func._synapsekit_public_api is True
        assert my_func() == 42

    def test_marks_class(self):
        @public_api
        class MyClass:
            pass

        assert MyClass._synapsekit_public_api is True
        obj = MyClass()
        assert obj is not None


class TestExperimental:
    def test_function_warns(self):
        @experimental
        def beta_func():
            return "beta"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = beta_func()
            assert result == "beta"
            assert len(w) == 1
            assert issubclass(w[0].category, FutureWarning)
            assert "experimental" in str(w[0].message)

    def test_class_warns(self):
        @experimental
        class BetaClass:
            def __init__(self):
                self.value = 99

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj = BetaClass()
            assert obj.value == 99
            assert len(w) == 1
            assert issubclass(w[0].category, FutureWarning)

    def test_experimental_attr_on_function(self):
        @experimental
        def f():
            pass

        assert f._synapsekit_experimental is True

    def test_experimental_attr_on_class(self):
        @experimental
        class C:
            pass

        assert C._synapsekit_experimental is True


class TestDeprecated:
    def test_function_warns(self):
        @deprecated("old API", alternative="new_func")
        def old_func():
            return "old"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = old_func()
            assert result == "old"
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()
            assert "new_func" in str(w[0].message)

    def test_class_warns(self):
        @deprecated("use V2 instead")
        class OldClass:
            pass

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            OldClass()
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

    def test_deprecated_no_alternative(self):
        @deprecated("going away")
        def doomed():
            return "bye"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            doomed()
            msg = str(w[0].message)
            assert "going away" in msg
            assert "Use " not in msg

    def test_deprecated_attr_on_function(self):
        @deprecated("old")
        def f():
            pass

        assert f._synapsekit_deprecated is True

    def test_deprecated_attr_on_class(self):
        @deprecated("old")
        class C:
            pass

        assert C._synapsekit_deprecated is True
