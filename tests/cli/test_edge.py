from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pytest

from synapsekit.cli.edge import MODEL_REGISTRY, run_edge


def test_edge_list_outputs_registry(capsys, tmp_path: Path) -> None:
    args = argparse.Namespace(edge_command="list", cache_dir=str(tmp_path), output_format="table")

    run_edge(args)

    out = capsys.readouterr().out
    assert "llama-3.2-3b" in out
    assert "minilm-l6-onnx" in out


def test_edge_list_json(capsys, tmp_path: Path) -> None:
    args = argparse.Namespace(edge_command="list", cache_dir=str(tmp_path), output_format="json")

    run_edge(args)

    out = capsys.readouterr().out
    assert '"models"' in out
    assert '"downloaded": false' in out


def test_edge_pull_unknown_model(tmp_path: Path) -> None:
    args = argparse.Namespace(
        edge_command="pull",
        cache_dir=str(tmp_path),
        model="missing-model",
        force=False,
    )

    with pytest.raises(SystemExit, match="Unknown edge model"):
        run_edge(args)


def test_edge_pull_uses_huggingface_hub(capsys, tmp_path: Path) -> None:
    model = MODEL_REGISTRY["llama-3.2-3b"]
    target = tmp_path / model.name / model.local_filename
    target.parent.mkdir(parents=True)
    target.write_text("mock")

    class FakeHub:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def hf_hub_download(
            self,
            repo_id: str,
            filename: str,
            local_dir: str | None = None,
            local_dir_use_symlinks: bool = True,
            force_download: bool = False,
        ) -> str:
            self.calls.append({"repo_id": repo_id, "filename": filename})
            return str(target)

    fake_hub = FakeHub()
    original = sys.modules.get("huggingface_hub")
    sys.modules["huggingface_hub"] = fake_hub  # type: ignore[assignment]
    try:
        args = argparse.Namespace(
            edge_command="pull",
            cache_dir=str(tmp_path),
            model=model.name,
            force=True,
        )
        run_edge(args)
    finally:
        if original is None:
            del sys.modules["huggingface_hub"]
        else:
            sys.modules["huggingface_hub"] = original

    assert len(fake_hub.calls) == 1
    assert fake_hub.calls[0]["repo_id"] == model.repo_id
    assert model.name in capsys.readouterr().out


def test_edge_quantize_missing_binary(tmp_path: Path) -> None:
    input_model = tmp_path / "input.gguf"
    input_model.write_text("model")

    # Pass llama_quantize=None and make sure PATH has no 'llama-quantize' binary
    # by pointing to an empty directory
    empty_bin = tmp_path / "emptybin"
    empty_bin.mkdir()
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = str(empty_bin)
    try:
        args = argparse.Namespace(
            edge_command="quantize",
            input_model=str(input_model),
            output=str(tmp_path / "out.gguf"),
            quantization="Q4_K_M",
            llama_quantize=None,
        )
        with pytest.raises(SystemExit, match="llama-quantize"):
            run_edge(args)
    finally:
        os.environ["PATH"] = old_path


def test_edge_quantize_runs_external_binary(tmp_path: Path, capsys) -> None:
    input_model = tmp_path / "input.gguf"
    input_model.write_text("model")

    # Write a tiny executable that exits successfully
    fake_binary = tmp_path / "llama-quantize"
    fake_binary.write_text("#!/usr/bin/env python3\nimport sys\nsys.exit(0)\n")
    fake_binary.chmod(0o755)

    args = argparse.Namespace(
        edge_command="quantize",
        input_model=str(input_model),
        output=str(tmp_path / "out.gguf"),
        quantization="Q4_K_M",
        llama_quantize=str(fake_binary),
    )

    run_edge(args)

    assert "Q4_K_M" in capsys.readouterr().out
