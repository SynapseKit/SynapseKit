from __future__ import annotations

import asyncio
import configparser
import os

from .base import Document

_SENSITIVE_KEYWORDS = {"password", "secret", "token", "api_key", "key", "auth"}
_SUPPORTED_EXTENSIONS = {".env", ".ini", ".cfg", ".toml"}


# Simple keyword-based detection; not exhaustive but avoids exposing common secrets
def _is_sensitive(key: str) -> bool:
    lower = key.lower()
    return any(kw in lower for kw in _SENSITIVE_KEYWORDS)


def _redact(key: str, value: str) -> str:
    return "***" if _is_sensitive(key) else value


def _parse_env(content: str) -> list[tuple[str, str]]:
    pairs = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            k, v = key.strip(), value.strip()
            if not k:
                continue
            pairs.append((k, v))
    return pairs


def _parse_ini(content: str) -> dict[str, list[tuple[str, str]]]:
    parser = configparser.ConfigParser()
    parser.read_string(content)
    sections: dict[str, list[tuple[str, str]]] = {}
    for section in parser.sections():
        sections[section] = [(k.strip(), v.strip()) for k, v in parser.items(section) if k.strip()]
    return sections


def _flatten_toml(data: dict, prefix: str = "") -> list[tuple[str, str]]:
    pairs = []
    for k, v in data.items():
        full_key = f"{prefix}.{k.strip()}" if prefix else k.strip()
        if not full_key:
            continue
        if isinstance(v, dict):
            pairs.extend(_flatten_toml(v, full_key))
        else:
            pairs.append((full_key, str(v).strip() if v is not None else ""))
    return pairs


class ConfigLoader:
    """Load .env, .ini, .cfg, or .toml config files into Documents.

    Sensitive keys (password, secret, token, api_key, key, auth) are redacted.
    """

    def __init__(self, path: str) -> None:
        self._path = path

    def load(self) -> list[Document]:
        if not os.path.exists(self._path):
            raise FileNotFoundError(f"Config file not found: {self._path}")

        basename = os.path.basename(self._path).lower()
        ext = os.path.splitext(self._path)[1].lower()
        # dotfiles like ".env" have no extension; treat the whole filename as the ext.
        # Also handles ".env.local", ".env.staging", etc.
        if not ext or basename.startswith(".env"):
            ext = ".env" if basename.startswith(".env") else basename
            if ext and not ext.startswith("."):
                ext = f".{ext}"
        if ext not in _SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported config file type: {ext!r}")

        with open(self._path, encoding="utf-8") as f:
            content = f.read()

        if ext == ".env":
            return self._load_env(content)
        if ext in {".ini", ".cfg"}:
            return self._load_ini(content)
        return self._load_toml(content)

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)

    # ------------------------------------------------------------------
    # Private parsers
    # ------------------------------------------------------------------

    def _load_env(self, content: str) -> list[Document]:
        pairs = _parse_env(content)
        lines = [f"{k}: {_redact(k, v)}" for k, v in pairs]
        text = "\n".join(lines)
        return [Document(text=text, metadata={"source": self._path, "type": "env"})]

    def _load_ini(self, content: str) -> list[Document]:
        sections = _parse_ini(content)
        if not sections:
            return [Document(text="", metadata={"source": self._path, "type": "ini"})]
        docs = []
        for section, pairs in sections.items():
            lines = [f"[{section}]"] + [f"{k}: {_redact(k, v)}" for k, v in pairs]
            docs.append(
                Document(
                    text="\n".join(lines),
                    metadata={"source": self._path, "type": "ini", "section": section},
                )
            )
        return docs

    def _load_toml(self, content: str) -> list[Document]:
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ImportError:
                raise ImportError(
                    "TOML loading requires Python 3.11+ or: pip install tomli"
                ) from None

        data = tomllib.loads(content)
        if not isinstance(data, dict):
            return [Document(text="", metadata={"source": self._path, "type": "toml"})]
        pairs = _flatten_toml(data)
        lines = [f"{k}: {_redact(k, v)}" for k, v in pairs]
        text = "\n".join(lines)
        return [Document(text=text, metadata={"source": self._path, "type": "toml"})]
