"""``synapsekit edge`` commands for local model management."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_EDGE_CACHE = Path.home() / ".cache" / "synapsekit" / "edge"


@dataclass(frozen=True)
class EdgeModel:
    """Registry entry for a known edge model artifact."""

    name: str
    repo_id: str
    filename: str
    backend: str
    size: str
    description: str

    @property
    def local_filename(self) -> str:
        return self.filename.rsplit("/", 1)[-1]


MODEL_REGISTRY: dict[str, EdgeModel] = {
    "llama-3.2-3b": EdgeModel(
        name="llama-3.2-3b",
        repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
        filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        backend="llamacpp",
        size="~2.0GB",
        description="Meta Llama 3.2 3B Instruct quantized for llama.cpp",
    ),
    "phi-3.5-mini": EdgeModel(
        name="phi-3.5-mini",
        repo_id="bartowski/Phi-3.5-mini-instruct-GGUF",
        filename="Phi-3.5-mini-instruct-Q4_K_M.gguf",
        backend="llamacpp",
        size="~2.4GB",
        description="Microsoft Phi 3.5 Mini Instruct quantized for llama.cpp",
    ),
    "minilm-l6-onnx": EdgeModel(
        name="minilm-l6-onnx",
        repo_id="Xenova/all-MiniLM-L6-v2",
        filename="onnx/model.onnx",
        backend="onnx",
        size="~90MB",
        description="All-MiniLM-L6-v2 sentence embeddings exported to ONNX",
    ),
}


def _cache_root(args: Any) -> Path:
    return Path(getattr(args, "cache_dir", None) or DEFAULT_EDGE_CACHE).expanduser()


def _model_path(model: EdgeModel, cache_root: Path) -> Path:
    return cache_root / model.name / model.local_filename


def _metadata_path(model: EdgeModel, cache_root: Path) -> Path:
    return cache_root / model.name / "model.json"


def _print_registry(args: Any) -> None:
    cache_root = _cache_root(args)
    payload = []

    for model in MODEL_REGISTRY.values():
        path = _model_path(model, cache_root)
        record = asdict(model)
        record["downloaded"] = path.exists()
        record["path"] = str(path)
        payload.append(record)

    if getattr(args, "output_format", "table") == "json":
        print(json.dumps({"models": payload}, indent=2))
        return

    print("Known edge models:")
    for record in payload:
        status = "downloaded" if record["downloaded"] else "missing"
        print(
            f"  - {record['name']:<16} {record['backend']:<8} {record['size']:<8} "
            f"{status:<10} {record['description']}"
        )


def _resolve_model(name: str) -> EdgeModel:
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise SystemExit(f"Unknown edge model '{name}'. Known models: {known}") from exc


def _download_model(model: EdgeModel, cache_root: Path, force: bool) -> Path:
    destination_dir = cache_root / model.name
    destination_dir.mkdir(parents=True, exist_ok=True)
    target_path = _model_path(model, cache_root)

    if target_path.exists() and not force:
        return target_path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit(
            "huggingface-hub required for 'synapsekit edge pull'. "
            "Install it with: pip install synapsekit[huggingface]"
        ) from None

    downloaded = hf_hub_download(
        repo_id=model.repo_id,
        filename=model.filename,
        local_dir=str(destination_dir),
        local_dir_use_symlinks=False,
        force_download=force,
    )
    downloaded_path = Path(downloaded)

    metadata = asdict(model)
    metadata["path"] = str(downloaded_path)
    _metadata_path(model, cache_root).write_text(json.dumps(metadata, indent=2) + "\n")
    return downloaded_path


def _pull_model(args: Any) -> None:
    model = _resolve_model(args.model)
    path = _download_model(model, _cache_root(args), bool(getattr(args, "force", False)))
    print(json.dumps({"model": model.name, "backend": model.backend, "path": str(path)}, indent=2))


def _quantize(args: Any) -> None:
    llama_quantize = getattr(args, "llama_quantize", None) or shutil.which("llama-quantize")
    if not llama_quantize:
        raise SystemExit(
            "llama-quantize was not found. Build llama.cpp and pass "
            "--llama-quantize /path/to/llama-quantize."
        )

    input_path = Path(args.input_model).expanduser()
    if not input_path.exists():
        raise SystemExit(f"Input model not found: {input_path}")

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantization = getattr(args, "quantization", "Q4_K_M")

    cmd = [str(llama_quantize), str(input_path), str(output_path), quantization]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Quantization failed with exit code {exc.returncode}") from exc

    print(json.dumps({"input": str(input_path), "output": str(output_path), "quantization": quantization}, indent=2))


def run_edge(args: Any) -> None:
    """Dispatch ``synapsekit edge`` subcommands."""

    command = getattr(args, "edge_command", None)
    if command == "list":
        _print_registry(args)
        return
    if command == "pull":
        _pull_model(args)
        return
    if command == "quantize":
        _quantize(args)
        return
    raise SystemExit("Missing edge action. Use 'list', 'pull', or 'quantize'.")


def build_edge_parser(subparsers: Any) -> None:
    """Register the ``edge`` parser with the top-level CLI."""

    parser = subparsers.add_parser("edge", help="Manage local edge models")
    parser.add_argument("--cache-dir", default=None, help="Edge model cache directory")
    edge_sub = parser.add_subparsers(dest="edge_command")

    list_cmd = edge_sub.add_parser("list", help="List known edge models")
    list_cmd.add_argument(
        "--format",
        dest="output_format",
        choices=["table", "json"],
        default="table",
        help="Output format",
    )

    pull_cmd = edge_sub.add_parser("pull", help="Download a known edge model")
    pull_cmd.add_argument("model", help="Registry name, e.g. llama-3.2-3b")
    pull_cmd.add_argument("--force", action="store_true", help="Redownload existing files")

    quantize_cmd = edge_sub.add_parser("quantize", help="Quantize a GGUF model with llama.cpp")
    quantize_cmd.add_argument("input_model", help="Input GGUF/F16 model path")
    quantize_cmd.add_argument("output", help="Output GGUF model path")
    quantize_cmd.add_argument("--quantization", default="Q4_K_M", help="llama.cpp quantization type")
    quantize_cmd.add_argument(
        "--llama-quantize",
        default=None,
        help="Path to llama.cpp llama-quantize binary",
    )
