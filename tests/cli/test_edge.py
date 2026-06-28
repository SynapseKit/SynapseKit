from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

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

    fake_hub = MagicMock()
    fake_hub.hf_hub_download.return_value = str(target)
    args = argparse.Namespace(
        edge_command="pull",
        cache_dir=str(tmp_path),
        model=model.name,
        force=True,
    )

    with patch.dict("sys.modules", {"huggingface_hub": fake_hub}):
        run_edge(args)

    assert fake_hub.hf_hub_download.call_args[1]["repo_id"] == model.repo_id
    assert model.name in capsys.readouterr().out


def test_edge_quantize_missing_binary(tmp_path: Path) -> None:
    input_model = tmp_path / "input.gguf"
    input_model.write_text("model")
    args = argparse.Namespace(
        edge_command="quantize",
        input_model=str(input_model),
        output=str(tmp_path / "out.gguf"),
        quantization="Q4_K_M",
        llama_quantize=None,
    )

    with patch("shutil.which", return_value=None):
        with pytest.raises(SystemExit, match="llama-quantize"):
            run_edge(args)


def test_edge_quantize_runs_external_binary(tmp_path: Path, capsys) -> None:
    input_model = tmp_path / "input.gguf"
    input_model.write_text("model")
    args = argparse.Namespace(
        edge_command="quantize",
        input_model=str(input_model),
        output=str(tmp_path / "out.gguf"),
        quantization="Q4_K_M",
        llama_quantize="llama-quantize",
    )

    with patch("subprocess.run") as run:
        run_edge(args)

    run.assert_called_once()
    assert "Q4_K_M" in capsys.readouterr().out
