"""
GitHub PR Auto-Reviewer Agent
==============================

This example demonstrates how to build an advanced agent that can connect to GitHub,
fetch the details of a specific Pull Request (including its diffs and files),
analyze the code for bugs, logic errors, or style issues, and generate a
review summary.

Prerequisites:
    pip install synapsekit[openai]

Usage:
    export OPENAI_API_KEY=sk-...
    export GITHUB_TOKEN=ghp_...
    python examples/github_pr_reviewer.py
"""

import asyncio
import os
import sys

from synapsekit import GitHubAPITool, ReActAgent


async def main():
    # 1. Ensure API keys are set
    api_key = os.environ.get("OPENAI_API_KEY")
    github_token = os.environ.get("GITHUB_TOKEN")

    if not api_key:
        print("Error: Please set OPENAI_API_KEY environment variable")
        sys.exit(1)

    if not github_token:
        print(
            "Warning: GITHUB_TOKEN is not set. You may hit rate limits or be unable to access private repos."
        )
        # Some GitHub APIs work without a token, but it's highly recommended.

    print("Initializing GitHub PR Reviewer Agent...")

    # 2. Initialize the Agent with GitHub capabilities
    agent = ReActAgent(
        model="gpt-4o",  # Using a smarter model for code review
        api_key=api_key,
        tools=[GitHubAPITool(token=github_token)],
        system_prompt=(
            "You are an expert Senior Software Engineer and Code Reviewer. "
            "Your task is to review GitHub Pull Requests thoroughly. "
            "When given a repository and a PR number: "
            "1. Fetch the PR details and read the description. "
            "2. Fetch the files changed and the diffs in the PR. "
            "3. Analyze the code changes for: "
            "   - Logic bugs or edge cases "
            "   - Performance issues "
            "   - Security vulnerabilities "
            "   - Clean code practices "
            "4. Provide a structured, markdown-formatted PR review as your final answer. "
            "Include a 'Summary', 'Major Issues', 'Minor Suggestions', and an overall 'Verdict' (Approve/Request Changes)."
        ),
        verbose=True,
    )

    # Note: Replace this with any public repository and PR number you want to test!
    target_repo = "SynapseKit/SynapseKit"
    pr_number = 1056

    query = (
        f"Please review Pull Request #{pr_number} in the repository '{target_repo}'. "
        "Fetch the diff, analyze the code, and give me your professional review."
    )

    print(f"\n{'=' * 60}")
    print(f"Agent Task: Reviewing PR #{pr_number} on {target_repo}")
    print(f"{'=' * 60}\n")

    # 3. Let the Agent execute the review!
    try:
        review_result = await agent.run(query)

        print(f"\n{'=' * 60}")
        print("FINAL PR REVIEW REPORT:")
        print(f"{'=' * 60}")
        print(review_result)

    except Exception as e:
        print(f"Error during review: {e}")


if __name__ == "__main__":
    asyncio.run(main())
