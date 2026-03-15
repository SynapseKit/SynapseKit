from __future__ import annotations

from synapsekit.graph.checkpointers.json_file import JSONFileCheckpointer


class TestJSONFileCheckpointer:
    def test_save_and_load(self, tmp_path):
        cp = JSONFileCheckpointer(directory=str(tmp_path))
        cp.save("graph1", 3, {"count": 42, "items": ["a", "b"]})
        result = cp.load("graph1")
        assert result is not None
        step, state = result
        assert step == 3
        assert state == {"count": 42, "items": ["a", "b"]}

    def test_load_missing(self, tmp_path):
        cp = JSONFileCheckpointer(directory=str(tmp_path))
        assert cp.load("nonexistent") is None

    def test_delete(self, tmp_path):
        cp = JSONFileCheckpointer(directory=str(tmp_path))
        cp.save("graph1", 1, {"x": 1})
        cp.delete("graph1")
        assert cp.load("graph1") is None

    def test_delete_missing(self, tmp_path):
        cp = JSONFileCheckpointer(directory=str(tmp_path))
        cp.delete("nonexistent")  # should not raise

    def test_overwrite(self, tmp_path):
        cp = JSONFileCheckpointer(directory=str(tmp_path))
        cp.save("g", 1, {"v": 1})
        cp.save("g", 5, {"v": 2})
        step, state = cp.load("g")
        assert step == 5
        assert state == {"v": 2}
