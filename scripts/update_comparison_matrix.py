from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

DOC_PATH = Path(__file__).resolve().parents[1] / "docs" / "comparison-matrix.md"

REPOS = {
    "SynapseKit": "synapsekit/synapsekit",
    "LangChain": "langchain-ai/langchain",
    "LlamaIndex": "run-llama/llama_index",
    "Haystack": "deepset-ai/haystack",
}


def fetch_pushed_at(repo: str) -> str:
    url = f"https://api.github.com/repos/{repo}"
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(url, headers=headers)
    with urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    pushed_at = data.get("pushed_at", "")
    return pushed_at[:10] if pushed_at else ""


def update_last_updated(text: str, today: str) -> str:
    pattern = r"\*\*Last updated:\*\* \d{4}-\d{2}-\d{2}"
    replacement = f"**Last updated:** {today}"
    return re.sub(pattern, replacement, text)


def update_maintenance_table(lines: list[str], dates: dict[str, str]) -> list[str]:
    for i, line in enumerate(lines):
        for name, date in dates.items():
            if line.startswith(f"| {name} |"):
                parts = [part.strip() for part in line.strip().strip("|").split("|")]
                if len(parts) >= 4:
                    parts[2] = date
                    lines[i] = "| " + " | ".join(parts) + " |"
                break
    return lines


def main() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    text = update_last_updated(text, today)

    dates = {name: fetch_pushed_at(repo) for name, repo in REPOS.items()}
    lines = update_maintenance_table(text.splitlines(), dates)
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Updated comparison-matrix.md")


if __name__ == "__main__":
    main()
