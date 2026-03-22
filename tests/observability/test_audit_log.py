"""Tests for AuditLog (v1.3.0)."""

from __future__ import annotations

import pytest

from synapsekit.observability.audit_log import AuditEntry, AuditLog


class TestAuditLogMemory:
    def test_record_and_len(self):
        log = AuditLog(backend="memory")
        assert len(log) == 0
        entry = log.record(model="gpt-4o", input_text="hi", output_text="hello")
        assert len(log) == 1
        assert isinstance(entry, AuditEntry)
        assert entry.model == "gpt-4o"

    def test_query_all(self):
        log = AuditLog(backend="memory")
        log.record(model="gpt-4o", input_text="a", output_text="b", user="alice")
        log.record(model="gpt-4o-mini", input_text="c", output_text="d", user="bob")
        results = log.query()
        assert len(results) == 2

    def test_query_by_user(self):
        log = AuditLog(backend="memory")
        log.record(model="gpt-4o", input_text="a", output_text="b", user="alice")
        log.record(model="gpt-4o", input_text="c", output_text="d", user="bob")
        results = log.query(user="alice")
        assert len(results) == 1
        assert results[0].user == "alice"

    def test_query_by_model(self):
        log = AuditLog(backend="memory")
        log.record(model="gpt-4o", input_text="a", output_text="b")
        log.record(model="claude", input_text="c", output_text="d")
        results = log.query(model="gpt-4o")
        assert len(results) == 1

    def test_query_with_limit(self):
        log = AuditLog(backend="memory")
        for i in range(5):
            log.record(model="gpt-4o", input_text=f"q{i}", output_text=f"a{i}")
        results = log.query(limit=2)
        assert len(results) == 2

    def test_immutability_no_delete_api(self):
        log = AuditLog(backend="memory")
        assert not hasattr(log, "delete")
        assert not hasattr(log, "update")

    def test_entry_has_timestamp_and_id(self):
        log = AuditLog(backend="memory")
        entry = log.record(model="m", input_text="i", output_text="o")
        assert entry.entry_id
        assert entry.timestamp
        assert entry.user == "anonymous"


class TestAuditLogSQLite:
    def test_sqlite_record_and_query(self, tmp_path):
        db_path = str(tmp_path / "audit.db")
        log = AuditLog(backend="sqlite", path=db_path)
        log.record(model="gpt-4o", input_text="hi", output_text="hello", user="alice")
        log.record(model="gpt-4o", input_text="q2", output_text="a2", user="bob")
        assert len(log) == 2
        results = log.query(user="alice")
        assert len(results) == 1

    def test_sqlite_persistence(self, tmp_path):
        db_path = str(tmp_path / "audit.db")
        log1 = AuditLog(backend="sqlite", path=db_path)
        log1.record(model="m", input_text="i", output_text="o")
        # Re-open
        log2 = AuditLog(backend="sqlite", path=db_path)
        assert len(log2) == 1


class TestAuditLogJSONL:
    def test_jsonl_record_and_query(self, tmp_path):
        path = str(tmp_path / "audit.jsonl")
        log = AuditLog(backend="jsonl", path=path)
        log.record(model="gpt-4o", input_text="hi", output_text="hello", user="alice")
        assert len(log) == 1
        results = log.query(user="alice")
        assert len(results) == 1

    def test_jsonl_persistence(self, tmp_path):
        path = str(tmp_path / "audit.jsonl")
        log1 = AuditLog(backend="jsonl", path=path)
        log1.record(model="m", input_text="i", output_text="o")
        # Re-open
        log2 = AuditLog(backend="jsonl", path=path)
        assert len(log2) == 1
        results = log2.query()
        assert len(results) == 1

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="backend must be"):
            AuditLog(backend="invalid")
