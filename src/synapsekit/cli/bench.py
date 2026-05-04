"""``synapsekit bench`` command: run/publish EvalHub community suites."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import json
import os
import shutil
import statistics
import subprocess
import tempfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..llm._factory import make_llm

EVALHUB_ROOT = Path(__file__).resolve().parents[1] / "evaluation" / "evalhub"
REGISTRY_INDEX = EVALHUB_ROOT / "index.json"
DEFAULT_REGISTRY_REPO = "SynapseKit/SynapseKit"
DEFAULT_REGISTRY_DIR = "src/synapsekit/evaluation/evalhub/community"


@dataclass
class BenchCase:
    id: str
    input: str
    ideal: str


@dataclass
class BenchSuite:
    name: str
    description: str
    baseline_score: float
    cases: list[BenchCase]


def run_bench(args: Any) -> None:
    """Run/list/publish EvalHub community suites."""
    if getattr(args, "list", False):
        _print_suite_list(_load_registry())
        return

    publish_path = getattr(args, "publish", None)
    if publish_path:
        _publish_suite(args)
        return

    suite_name = getattr(args, "suite", None)
    if suite_name:
        _run_suite(args, _load_registry())
        return

    raise SystemExit("Missing bench action. Use --list, --suite, or --publish")


def _load_registry() -> dict[str, BenchSuite]:
    if not REGISTRY_INDEX.exists():
        raise SystemExit(f"EvalHub registry index not found: {REGISTRY_INDEX}")

    try:
        index_data = json.loads(REGISTRY_INDEX.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - malformed file safety
        raise SystemExit(f"EvalHub registry index is invalid JSON: {exc}") from exc

    suites: dict[str, BenchSuite] = {}
    for entry in index_data.get("suites", []):
        suite_path = EVALHUB_ROOT / str(entry.get("path", ""))
        if not suite_path.exists():
            raise SystemExit(f"EvalHub suite file missing: {suite_path}")

        suite_data = json.loads(suite_path.read_text(encoding="utf-8"))
        name = str(suite_data.get("name") or entry.get("name") or "")
        if not name:
            raise SystemExit(f"Suite missing name in: {suite_path}")

        cases = [
            BenchCase(
                id=str(case.get("id", "")),
                input=str(case.get("input", "")),
                ideal=str(case.get("ideal", "")),
            )
            for case in suite_data.get("cases", [])
        ]

        suites[name] = BenchSuite(
            name=name,
            description=str(suite_data.get("description") or entry.get("description") or ""),
            baseline_score=float(
                suite_data.get("baseline_score")
                if suite_data.get("baseline_score") is not None
                else entry.get("baseline_score", 0.0)
            ),
            cases=cases,
        )

    return suites


def _print_suite_list(suites: dict[str, BenchSuite]) -> None:
    print("Available community suites:")
    for key in sorted(suites):
        suite = suites[key]
        print(
            f"  - {key} | {suite.description} | cases: {len(suite.cases)} | baseline: {suite.baseline_score:.2f}"
        )


def _load_callable(path_value: str):
    try:
        module_name, func_name = path_value.split(":", 1)
        module = importlib.import_module(module_name)
        fn = getattr(module, func_name)
    except Exception as exc:
        raise SystemExit(f"Failed to load callable '{path_value}': {exc}") from exc

    if not callable(fn):
        raise SystemExit(f"Loaded object '{path_value}' is not callable")
    return fn


def _resolve_provider(model: str, provider: str | None) -> str:
    if provider:
        return provider
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("gemini"):
        return "gemini"
    if model.startswith("command"):
        return "cohere"
    if model.startswith("mistral") or model.startswith("open-mistral"):
        return "mistral"
    if model.startswith("deepseek"):
        return "deepseek"
    if model.startswith("moonshot"):
        return "moonshot"
    if model.startswith("abab") or model.startswith("minimax"):
        return "minimax"
    if model.startswith("glm"):
        return "zhipu"
    if model.startswith("jamba"):
        return "ai21"
    if model.startswith("luminous") or model.startswith("pharia"):
        return "aleph-alpha"
    if model.startswith("@cf/") or model.startswith("@hf/"):
        return "cloudflare"
    if model.startswith("dbrx") or model.startswith("databricks"):
        return "databricks"
    if model.startswith("ernie"):
        return "ernie"
    if model.startswith("sambanova"):
        return "sambanova"
    if "/" in model:
        return "openrouter"
    return "openai"


def _api_key_env_for_provider(provider: str) -> str | None:
    mapping = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "cohere": "COHERE_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "moonshot": "MOONSHOT_API_KEY",
        "minimax": "MINIMAX_API_KEY",
        "zhipu": "ZHIPU_API_KEY",
        "ai21": "AI21_API_KEY",
        "aleph-alpha": "ALEPH_ALPHA_API_KEY",
        "cloudflare": "CLOUDFLARE_API_TOKEN",
        "databricks": "DATABRICKS_TOKEN",
        "ernie": "ERNIE_API_KEY",
        "sambanova": "SAMBANOVA_API_KEY",
        "groq": "GROQ_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "together": "TOGETHER_API_KEY",
        "fireworks": "FIREWORKS_API_KEY",
        "ollama": "OLLAMA_API_KEY",
    }
    return mapping.get(provider)


def _build_model_runner(args: Any):
    model = getattr(args, "model", None)
    if not model:
        raise SystemExit("--model is required for suite runs unless --pipeline/--agent is provided")

    provider = _resolve_provider(model, getattr(args, "provider", None))
    api_key = getattr(args, "api_key", None)
    if not api_key:
        env_name = _api_key_env_for_provider(provider)
        if env_name:
            api_key = os.getenv(env_name)
    if not api_key:
        env_name = _api_key_env_for_provider(provider)
        hint = f" or set {env_name}" if env_name else ""
        raise SystemExit(f"Missing API key for provider '{provider}'. Use --api-key{hint}.")

    llm = make_llm(
        model=model,
        api_key=api_key,
        provider=provider,
        system_prompt=getattr(args, "system_prompt", "You are a concise benchmark assistant."),
        temperature=float(getattr(args, "temperature", 0.0)),
        max_tokens=int(getattr(args, "max_tokens", 512)),
    )

    async def _runner(prompt: str) -> str:
        return await llm.generate(prompt)

    return _runner


def _resolve_runner(args: Any):
    pipeline_path = getattr(args, "pipeline", None)
    if pipeline_path:
        return _load_callable(pipeline_path)

    # Legacy compatibility with previous draft implementation.
    agent_path = getattr(args, "agent", None)
    if agent_path:
        return _load_callable(agent_path)

    return _build_model_runner(args)


async def _call_runner(runner: Any, prompt: str) -> str:
    if inspect.iscoroutinefunction(runner):
        result = await runner(prompt)
    else:
        result = runner(prompt)
        if inspect.isawaitable(result):
            result = await result
    return str(result)


def _simple_text_score(prediction: str, ideal: str) -> float:
    pred_tokens = {t for t in prediction.lower().split() if t}
    ideal_tokens = {t for t in ideal.lower().split() if t}
    if not ideal_tokens:
        return 0.0
    overlap = len(pred_tokens & ideal_tokens)
    return round(overlap / len(ideal_tokens), 4)


async def _evaluate_cases(
    cases: list[BenchCase], runner: Callable[[str], Any]
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case in cases:
        prediction = await _call_runner(runner, case.input)
        score = _simple_text_score(prediction, case.ideal)
        results.append(
            {
                "id": case.id,
                "score": score,
                "input": case.input,
                "ideal": case.ideal,
                "prediction": prediction,
            }
        )
    return results


def _run_suite(args: Any, suites: dict[str, BenchSuite]) -> None:
    suite_name = args.suite
    if suite_name not in suites:
        raise SystemExit(f"Unknown suite: {suite_name}")

    suite = suites[suite_name]
    limit = getattr(args, "limit", None)
    cases = suite.cases[:limit] if isinstance(limit, int) and limit > 0 else suite.cases
    runner = _resolve_runner(args)

    case_results = asyncio.run(_evaluate_cases(cases, runner))
    avg_score = statistics.fmean(r["score"] for r in case_results) if case_results else 0.0
    delta = round(avg_score - suite.baseline_score, 4)

    output_format = getattr(args, "output_format", "table")
    payload = {
        "suite": suite.name,
        "description": suite.description,
        "cases": len(case_results),
        "aggregate_score": round(avg_score, 4),
        "community_baseline": suite.baseline_score,
        "baseline_delta": delta,
        "model": getattr(args, "model", None),
        "results": case_results,
    }

    if output_format == "json":
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    print(f"Suite: {suite.name}")
    print(f"Description: {suite.description}")
    if getattr(args, "model", None):
        print(f"Model: {args.model}")
    print()
    print(f"{'Case':<16} {'Score':<8}")
    print("-" * 28)
    for row in case_results:
        print(f"{row['id']:<16} {row['score']:<8.4f}")
    print("-" * 28)
    print(f"Aggregate score: {avg_score:.4f}")
    print(f"Community baseline: {suite.baseline_score:.4f}")
    print(f"Delta vs baseline: {delta:+.4f}")


def _slugify_suite_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "/"} else "-" for ch in name)
    cleaned = cleaned.strip("/").replace("/", "-").lower()
    return cleaned or "evalhub-suite"


def _suite_registry_path(registry_dir: str, suite_name: str) -> Path:
    parts = [p for p in suite_name.split("/") if p]
    if len(parts) == 1:
        parts = ["community", parts[0]]
    return Path(registry_dir).joinpath(*parts)


def _bundle_suite_folder(publish_path: Path, suite_name: str, bundle_out: str | None) -> Path:
    if bundle_out:
        zip_path = Path(bundle_out)
    else:
        zip_path = Path.cwd() / f"{_slugify_suite_name(suite_name)}.zip"

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(p for p in publish_path.rglob("*") if p.is_file()):
            zf.write(file, arcname=str(file.relative_to(publish_path)))
    return zip_path


def _run_external(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            check=True,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit(f"Command not found: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        details = stderr or stdout or str(exc)
        raise SystemExit(f"Command failed: {' '.join(cmd)}\n{details}") from exc


def _submit_suite_pr(
    publish_path: Path,
    manifest: dict[str, Any],
    suite_name: str,
    registry_repo: str,
    registry_dir: str,
) -> str:
    _run_external(["gh", "auth", "status"])

    branch_name = (
        f"evalhub/{_slugify_suite_name(suite_name)}-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    )

    with tempfile.TemporaryDirectory(prefix="synapsekit-evalhub-") as tmpdir:
        repo_dir = Path(tmpdir) / "registry"
        _run_external(["gh", "repo", "clone", registry_repo, str(repo_dir), "--", "--depth", "1"])

        target_dir = repo_dir / _suite_registry_path(registry_dir, suite_name)
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(publish_path, target_dir)
        (target_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        _run_external(["git", "checkout", "-b", branch_name], cwd=repo_dir)
        _run_external(["git", "add", str(target_dir.relative_to(repo_dir))], cwd=repo_dir)
        _run_external(
            [
                "git",
                "commit",
                "-m",
                f"feat(evalhub): publish {suite_name}",
            ],
            cwd=repo_dir,
        )
        _run_external(["git", "push", "--set-upstream", "origin", branch_name], cwd=repo_dir)

        pr_body = (
            "Automated EvalHub suite submission generated by `synapsekit bench --publish`.\n\n"
            f"Suite: `{suite_name}`\n"
            f"Source folder: `{publish_path}`"
        )
        pr_result = _run_external(
            [
                "gh",
                "pr",
                "create",
                "--repo",
                registry_repo,
                "--title",
                f"feat(evalhub): add {suite_name}",
                "--body",
                pr_body,
                "--head",
                branch_name,
            ],
            cwd=repo_dir,
        )

    out = (pr_result.stdout or "").strip()
    return out.splitlines()[-1] if out else ""


def _publish_suite(args: Any) -> None:
    publish_path = Path(args.publish)
    if not publish_path.exists() or not publish_path.is_dir():
        raise SystemExit(f"Publish path not found or not a directory: {publish_path}")

    suite_name = getattr(args, "name", None)
    if not suite_name:
        raise SystemExit("--name is required with --publish (example: myorg/rag-finance)")

    files = [
        str(p.relative_to(publish_path))
        for p in sorted(publish_path.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    ]
    if not files:
        raise SystemExit(f"No files found in publish directory: {publish_path}")

    bundle_path = _bundle_suite_folder(
        publish_path=publish_path,
        suite_name=suite_name,
        bundle_out=getattr(args, "bundle_out", None),
    )

    manifest: dict[str, Any] = {
        "name": suite_name,
        "path": str(publish_path.resolve()),
        "files": files,
        "bundle": str(bundle_path.resolve()),
        "registry_repo": getattr(args, "registry_repo", DEFAULT_REGISTRY_REPO),
        "registry_dir": getattr(args, "registry_dir", DEFAULT_REGISTRY_DIR),
    }

    if not getattr(args, "no_submit", False):
        pr_url = _submit_suite_pr(
            publish_path=publish_path,
            manifest=manifest,
            suite_name=suite_name,
            registry_repo=manifest["registry_repo"],
            registry_dir=manifest["registry_dir"],
        )
        manifest["pull_request"] = pr_url

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def build_bench_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("bench", help="Run community eval suites (EvalHub)")
    p.add_argument("--suite", default=None, help="Suite name, e.g. community/customer-support")
    p.add_argument("--model", default=None, help="Model identifier, e.g. gpt-4o-mini")
    p.add_argument("--provider", default=None, help="Optional model provider override")
    p.add_argument("--api-key", default=None, help="Optional API key (else env var is used)")
    p.add_argument(
        "--pipeline", default=None, help="Callable import path, e.g. my_module:run_pipeline"
    )
    p.add_argument("--agent", default=None, help="Legacy alias of --pipeline")
    p.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    p.add_argument("--max-tokens", type=int, default=512, help="Generation max tokens")
    p.add_argument("--system-prompt", default="You are a concise benchmark assistant.")
    p.add_argument("--limit", type=int, default=None, help="Optional case limit")
    p.add_argument("--list", action="store_true", help="List available community suites")

    p.add_argument("--publish", default=None, help="Directory to package and publish")
    p.add_argument("--name", default=None, help="Published suite name, e.g. myorg/rag-finance")
    p.add_argument("--registry-repo", default=DEFAULT_REGISTRY_REPO, help="Registry GitHub repo")
    p.add_argument("--registry-dir", default=DEFAULT_REGISTRY_DIR, help="Registry path inside repo")
    p.add_argument("--bundle-out", default=None, help="Optional output zip path")
    p.add_argument("--no-submit", action="store_true", help="Package only, skip PR submission")

    p.add_argument(
        "--format",
        dest="output_format",
        choices=["table", "json"],
        default="table",
        help="Output format",
    )
