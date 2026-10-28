"""
GitHub PR Auto-Reviewer Agent
==============================

This example demonstrates how to build an agent that connects to GitHub,
fetches the details and unified diff of a specific Pull Request, analyzes the
code for bugs, logic errors, or style issues, and generates a review summary.

It wires a ``ReActAgent`` to the ``GitHubAPITool``, which exposes the
``get_pull_request`` and ``get_pull_request_diff`` actions the agent needs.

Prerequisites:
    pip install synapsekit[openai]

Usage:
    export OPENAI_API_KEY=sk-...
    export GITHUB_TOKEN=ghp_...            # recommended (higher rate limits, private repos)
    python examples/github_pr_reviewer.py SynapseKit/SynapseKit 1056
"""

import asyncio
import os
import sys

from synapsekit import GitHubAPITool, ReActAgent
from synapsekit.llm.base import LLMConfig
from synapsekit.llm.openai import OpenAILLM

_REVIEWER_INSTRUCTIONS = (
    "You are an expert Senior Software Engineer and Code Reviewer. "
    "Review the pull request {owner}/{repo} #{pull_number} thoroughly.\n\n"
    "Use the github_api tool to do this:\n"
    "1. action='get_pull_request' with owner='{owner}', repo='{repo}', "
    "pull_number={pull_number} to read the PR title and description.\n"
    "2. action='get_pull_request_diff' with the same arguments to read the "
    "actual code diff.\n\n"
    "Then analyze the diff for:\n"
    "  - Logic bugs or unhandled edge cases\n"
    "  - Performance issues\n"
    "  - Security vulnerabilities\n"
    "  - Clean-code practices\n\n"
    "Produce a structured, markdown-formatted review as your final answer, with "
    "sections: 'Summary', 'Major Issues', 'Minor Suggestions', and an overall "
    "'Verdict' (Approve / Request Changes)."
)


async def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Please set OPENAI_API_KEY")

    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        print(
            "Warning: GITHUB_TOKEN is not set. You may hit rate limits or be "
            "unable to access private repos.",
            file=sys.stderr,
        )

    # Target repo / PR: "owner/repo" and a PR number (defaults to a public example).
    target = sys.argv[1] if len(sys.argv) > 1 else "SynapseKit/SynapseKit"
    pull_number = int(sys.argv[2]) if len(sys.argv) > 2 else 1056
    owner, repo = target.split("/", 1)

    # A smarter model is worth it for code review.
    llm = OpenAILLM(LLMConfig(model="gpt-4o", api_key=api_key, provider="openai"))
    agent = ReActAgent(
        llm=llm,
        tools=[GitHubAPITool(token=github_token)],
        max_iterations=8,
    )

    print(f"\n{'=' * 60}")
    print(f"Reviewing PR #{pull_number} on {target}")
    print(f"{'=' * 60}\n")

    query = _REVIEWER_INSTRUCTIONS.format(owner=owner, repo=repo, pull_number=pull_number)

    try:
        review = await agent.run(query)
    except Exception as e:
        raise SystemExit(f"Error during review: {e}") from e

    print(f"\n{'=' * 60}")
    print("FINAL PR REVIEW REPORT")
    print(f"{'=' * 60}")
    print(review)


if __name__ == "__main__":
    asyncio.run(main())
