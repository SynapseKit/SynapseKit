from __future__ import annotations

import argparse

from synapsekit.agents import AgentConfigPatch, AgentEvolutionAuditLog
from synapsekit.cli.agent import run_agent


def test_agent_cli_inspect_evolution_table(tmp_path, capsys):
    path = tmp_path / "evolution.jsonl"
    patch = AgentConfigPatch(
        patch_type="prompt_rewrite",
        description="Improve accuracy",
        changes={"system_prompt": "better"},
        status="canary",
        eval_score=0.93,
        rollout_pct=5.0,
        metadata={"agent_id": "assistant"},
    )
    AgentEvolutionAuditLog(path).append(patch)

    args = argparse.Namespace(
        agent_command="inspect-evolution",
        agent_id="assistant",
        audit_path=str(path),
        limit=20,
        output_format="table",
    )
    run_agent(args)

    captured = capsys.readouterr()
    assert "Improve accuracy" in captured.out
    assert "canary" in captured.out


def test_agent_cli_inspect_evolution_json(tmp_path, capsys):
    path = tmp_path / "evolution.jsonl"
    patch = AgentConfigPatch(
        patch_type="prompt_rewrite",
        description="JSON output",
        changes={"system_prompt": "better"},
        status="blocked",
        metadata={"agent_id": "assistant"},
    )
    AgentEvolutionAuditLog(path).append(patch)

    args = argparse.Namespace(
        agent_command="inspect-evolution",
        agent_id="all",
        audit_path=str(path),
        limit=20,
        output_format="json",
    )
    run_agent(args)

    captured = capsys.readouterr()
    assert '"description": "JSON output"' in captured.out


def test_agent_cli_missing_log(capsys, tmp_path):
    args = argparse.Namespace(
        agent_command="inspect-evolution",
        agent_id="assistant",
        audit_path=str(tmp_path / "missing.jsonl"),
        limit=20,
        output_format="table",
    )
    run_agent(args)

    captured = capsys.readouterr()
    assert "No evolution audit log found" in captured.out
