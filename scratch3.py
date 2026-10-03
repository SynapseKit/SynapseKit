import json
from synapsekit.cli.agent import _run_inspect_evolution
from types import SimpleNamespace

# Create a fake audit log with a null before/after
with open("fake_audit.jsonl", "w") as f:
    f.write(json.dumps({"patch_type": "prompt_rewrite", "description": "test", "changes": {}, "before": None, "after": None, "metadata": None}) + "\n")

args = SimpleNamespace(audit_path="fake_audit.jsonl", agent_id="test", limit=10, output_format="text")
_run_inspect_evolution(args)
