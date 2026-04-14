from __future__ import annotations

import pytest

from synapsekit.loaders.base import Document
from synapsekit.loaders.config import ConfigLoader


class TestConfigLoaderErrors:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            ConfigLoader("/nonexistent/file.env").load()

    def test_unsupported_extension(self, tmp_path):
        p = tmp_path / "config.xml"
        p.write_text("<config/>")
        with pytest.raises(ValueError, match="Unsupported"):
            ConfigLoader(str(p)).load()

    async def test_aload(self, tmp_path):
        p = tmp_path / "config.ini"
        p.write_text("[app]\nname=synapsekit\n")
        docs = await ConfigLoader(str(p)).aload()
        assert len(docs) == 1
        assert "name: synapsekit" in docs[0].text


class TestEnvLoader:
    def test_basic_parsing(self, tmp_path):
        p = tmp_path / ".env"
        p.write_text("HOST=localhost\nPORT=8080\n")
        docs = ConfigLoader(str(p)).load()
        assert len(docs) == 1
        assert "HOST: localhost" in docs[0].text
        assert "PORT: 8080" in docs[0].text

    def test_skips_comments_and_blank_lines(self, tmp_path):
        p = tmp_path / ".env"
        p.write_text("# comment\n\nHOST=localhost\n")
        docs = ConfigLoader(str(p)).load()
        assert "comment" not in docs[0].text
        assert "HOST: localhost" in docs[0].text

    def test_redacts_sensitive_keys(self, tmp_path):
        p = tmp_path / ".env"
        p.write_text("API_KEY=supersecret\nDB_PASSWORD=hunter2\nHOST=localhost\n")
        docs = ConfigLoader(str(p)).load()
        assert "supersecret" not in docs[0].text
        assert "hunter2" not in docs[0].text
        assert "***" in docs[0].text
        assert "HOST: localhost" in docs[0].text

    def test_token_redacted(self, tmp_path):
        p = tmp_path / ".env"
        p.write_text("AUTH_TOKEN=abc123\n")
        docs = ConfigLoader(str(p)).load()
        assert "abc123" not in docs[0].text

    def test_env_local_treated_as_env(self, tmp_path):
        """Regression: .env.local / .env.staging must be parsed as .env, not raise ValueError."""
        p = tmp_path / ".env.local"
        p.write_text("STAGE=local\nDB_HOST=localhost\n")
        docs = ConfigLoader(str(p)).load()
        assert len(docs) == 1
        assert "STAGE: local" in docs[0].text
        assert docs[0].metadata["type"] == "env"

    def test_env_staging_treated_as_env(self, tmp_path):
        p = tmp_path / ".env.staging"
        p.write_text("STAGE=staging\n")
        docs = ConfigLoader(str(p)).load()
        assert docs[0].metadata["type"] == "env"

    def test_metadata_type(self, tmp_path):
        p = tmp_path / ".env"
        p.write_text("HOST=localhost\n")
        docs = ConfigLoader(str(p)).load()
        assert docs[0].metadata["type"] == "env"
        assert docs[0].metadata["source"] == str(p)

    def test_empty_file(self, tmp_path):
        p = tmp_path / ".env"
        p.write_text("")
        docs = ConfigLoader(str(p)).load()
        assert len(docs) == 1
        assert docs[0].text == ""


class TestIniLoader:
    INI_CONTENT = "[database]\nhost=localhost\nport=5432\n\n[app]\ndebug=true\n"

    def test_one_doc_per_section(self, tmp_path):
        p = tmp_path / "config.ini"
        p.write_text(self.INI_CONTENT)
        docs = ConfigLoader(str(p)).load()
        assert len(docs) == 2

    def test_section_in_text(self, tmp_path):
        p = tmp_path / "config.ini"
        p.write_text(self.INI_CONTENT)
        docs = ConfigLoader(str(p)).load()
        texts = [d.text for d in docs]
        assert any("[database]" in t for t in texts)

    def test_key_values_in_text(self, tmp_path):
        p = tmp_path / "config.ini"
        p.write_text(self.INI_CONTENT)
        docs = ConfigLoader(str(p)).load()
        db_doc = next(d for d in docs if d.metadata.get("section") == "database")
        assert "host: localhost" in db_doc.text
        assert "port: 5432" in db_doc.text

    def test_metadata_section(self, tmp_path):
        p = tmp_path / "config.ini"
        p.write_text(self.INI_CONTENT)
        docs = ConfigLoader(str(p)).load()
        sections = {d.metadata["section"] for d in docs}
        assert "database" in sections
        assert "app" in sections

    def test_redacts_password_in_ini(self, tmp_path):
        p = tmp_path / "config.ini"
        p.write_text("[db]\npassword=secret123\nhost=localhost\n")
        docs = ConfigLoader(str(p)).load()
        assert "secret123" not in docs[0].text
        assert "***" in docs[0].text

    def test_cfg_extension(self, tmp_path):
        p = tmp_path / "setup.cfg"
        p.write_text("[metadata]\nname=myapp\nversion=1.0\n")
        docs = ConfigLoader(str(p)).load()
        assert len(docs) == 1
        assert "name: myapp" in docs[0].text

    def test_empty_ini(self, tmp_path):
        p = tmp_path / "empty.ini"
        p.write_text("")
        docs = ConfigLoader(str(p)).load()
        assert len(docs) == 1
        assert docs[0].text == ""


class TestTomlLoader:
    TOML_CONTENT = '[database]\nhost = "localhost"\nport = 5432\n\n[app]\ndebug = true\n'

    def test_returns_single_document(self, tmp_path):
        p = tmp_path / "config.toml"
        p.write_text(self.TOML_CONTENT)
        docs = ConfigLoader(str(p)).load()
        assert len(docs) == 1
        assert isinstance(docs[0], Document)

    def test_flattens_nested_keys(self, tmp_path):
        p = tmp_path / "config.toml"
        p.write_text(self.TOML_CONTENT)
        docs = ConfigLoader(str(p)).load()
        assert "database.host: localhost" in docs[0].text
        assert "database.port: 5432" in docs[0].text

    def test_redacts_secret_in_toml(self, tmp_path):
        p = tmp_path / "config.toml"
        p.write_text('[auth]\napi_key = "topsecret"\nhost = "example.com"\n')
        docs = ConfigLoader(str(p)).load()
        assert "topsecret" not in docs[0].text
        assert "***" in docs[0].text

    def test_metadata_type_toml(self, tmp_path):
        p = tmp_path / "config.toml"
        p.write_text('[app]\nname = "test"\n')
        docs = ConfigLoader(str(p)).load()
        assert docs[0].metadata["type"] == "toml"
        assert docs[0].metadata["source"] == str(p)

    def test_empty_toml(self, tmp_path):
        p = tmp_path / "empty.toml"
        p.write_text("")
        docs = ConfigLoader(str(p)).load()
        assert len(docs) == 1
        assert docs[0].text == ""

    def test_aload_is_coroutine_function(self):
        import inspect

        assert inspect.iscoroutinefunction(ConfigLoader("/tmp/x.env").aload)
