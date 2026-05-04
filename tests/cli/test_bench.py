import argparse
from unittest.mock import MagicMock, patch

import pytest

from synapsekit.cli.bench import run_bench


def test_run_bench_list(capsys):
    args = argparse.Namespace(list=True, suite=None, publish=None)
    run_bench(args)

    captured = capsys.readouterr()
    assert "Available community suites:" in captured.out
    assert "community/customer-support" in captured.out
    assert "cases: 20" in captured.out
    assert "community/qa-hotpotqa" in captured.out


@patch("synapsekit.cli.bench.importlib.import_module")
def test_run_bench_suite_with_pipeline_success(mock_import_module, capsys):
    mock_module = MagicMock()

    def mock_pipeline(prompt):
        return "accurate concise response"

    mock_module.mock_pipeline = mock_pipeline
    mock_import_module.return_value = mock_module

    args = argparse.Namespace(
        list=False,
        publish=None,
        suite="community/customer-support",
        pipeline="my_module:mock_pipeline",
        agent=None,
        model=None,
        provider=None,
        api_key=None,
        temperature=0.0,
        max_tokens=128,
        system_prompt="You are concise.",
        limit=1,
        output_format="table",
    )
    run_bench(args)

    captured = capsys.readouterr()
    assert "Suite: community/customer-support" in captured.out
    assert "Aggregate score:" in captured.out
    assert "Community baseline:" in captured.out


@patch("synapsekit.cli.bench.make_llm")
def test_run_bench_suite_with_model_success(mock_make_llm, capsys):
    mock_llm = MagicMock()

    async def fake_generate(prompt):
        return "high quality response"

    mock_llm.generate = fake_generate
    mock_make_llm.return_value = mock_llm

    args = argparse.Namespace(
        list=False,
        publish=None,
        suite="community/summarization",
        pipeline=None,
        agent=None,
        model="gpt-4o-mini",
        provider="openai",
        api_key="sk-test",
        temperature=0.0,
        max_tokens=128,
        system_prompt="You are concise.",
        limit=1,
        output_format="json",
    )
    run_bench(args)

    out = capsys.readouterr().out
    assert '"suite": "community/summarization"' in out
    assert '"aggregate_score"' in out


def test_run_bench_unknown_suite():
    args = argparse.Namespace(
        list=False,
        publish=None,
        suite="community/does-not-exist",
        pipeline=None,
        agent=None,
        model="gpt-4o-mini",
        provider="openai",
        api_key="sk-test",
        temperature=0.0,
        max_tokens=128,
        system_prompt="You are concise.",
        limit=None,
        output_format="table",
    )
    with pytest.raises(SystemExit) as exc_info:
        run_bench(args)
    assert "Unknown suite" in str(exc_info.value)


def test_run_bench_suite_requires_model_or_pipeline():
    args = argparse.Namespace(
        list=False,
        publish=None,
        suite="community/customer-support",
        pipeline=None,
        agent=None,
        model=None,
        provider=None,
        api_key=None,
        temperature=0.0,
        max_tokens=128,
        system_prompt="You are concise.",
        limit=None,
        output_format="table",
    )
    with pytest.raises(SystemExit) as exc_info:
        run_bench(args)
    assert "--model is required" in str(exc_info.value)


def test_run_bench_publish_requires_name(tmp_path):
    args = argparse.Namespace(list=False, publish=str(tmp_path), name=None, suite=None)
    with pytest.raises(SystemExit) as exc_info:
        run_bench(args)
    assert "--name is required" in str(exc_info.value)


def test_run_bench_publish_no_submit_prints_manifest_and_bundle(tmp_path, capsys):
    folder = tmp_path / "my_evals"
    folder.mkdir()
    (folder / "suite.json").write_text("{}", encoding="utf-8")

    args = argparse.Namespace(
        list=False,
        publish=str(folder),
        name="myorg/rag-finance",
        suite=None,
        bundle_out=str(tmp_path / "myorg-rag-finance.zip"),
        registry_repo="SynapseKit/SynapseKit",
        registry_dir="src/synapsekit/evaluation/evalhub/community",
        no_submit=True,
    )
    run_bench(args)

    out = capsys.readouterr().out
    assert '"name": "myorg/rag-finance"' in out
    assert '"suite.json"' in out
    assert '"bundle"' in out


@patch("synapsekit.cli.bench._submit_suite_pr")
def test_run_bench_publish_submit_attaches_pr(mock_submit, tmp_path, capsys):
    mock_submit.return_value = "https://github.com/SynapseKit/SynapseKit/pull/999"

    folder = tmp_path / "my_evals"
    folder.mkdir()
    (folder / "suite.json").write_text("{}", encoding="utf-8")

    args = argparse.Namespace(
        list=False,
        publish=str(folder),
        name="myorg/rag-finance",
        suite=None,
        bundle_out=str(tmp_path / "myorg-rag-finance.zip"),
        registry_repo="SynapseKit/SynapseKit",
        registry_dir="src/synapsekit/evaluation/evalhub/community",
        no_submit=False,
    )
    run_bench(args)

    out = capsys.readouterr().out
    assert '"pull_request": "https://github.com/SynapseKit/SynapseKit/pull/999"' in out


def test_run_bench_missing_action():
    args = argparse.Namespace(list=False, publish=None, suite=None)
    with pytest.raises(SystemExit) as exc_info:
        run_bench(args)
    assert "Missing bench action" in str(exc_info.value)


@patch("synapsekit.cli.bench.importlib.import_module")
def test_run_bench_limit_restricts_cases(mock_import_module, capsys):
    """--limit N should evaluate at most N cases, not all cases in the suite."""
    mock_module = MagicMock()

    def mock_pipeline(prompt):
        return "response"

    mock_module.mock_pipeline = mock_pipeline
    mock_import_module.return_value = mock_module

    args = argparse.Namespace(
        list=False,
        publish=None,
        suite="community/qa-hotpotqa",  # 50 cases
        pipeline="my_module:mock_pipeline",
        agent=None,
        model=None,
        provider=None,
        api_key=None,
        temperature=0.0,
        max_tokens=128,
        system_prompt="",
        limit=3,
        output_format="json",
    )
    run_bench(args)

    import json

    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["cases"] == 3
