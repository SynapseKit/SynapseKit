"""
Graph Workflow with Conditional Edges
======================================

This example demonstrates SynapseKit's StateGraph for building workflows
with conditional routing, state management, and visualization.

Prerequisites:
    pip install synapsekit[openai]

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/graph_workflow.py
"""

import asyncio
import os
from typing import Literal

from synapsekit import StateGraph, END


# Define workflow state
class ReviewState:
    """State for content review workflow"""
    content: str = ""
    word_count: int = 0
    is_approved: bool = False
    feedback: str = ""


# Define node functions
async def count_words(state: ReviewState) -> ReviewState:
    """Count words in content"""
    words = state.content.split()
    state.word_count = len(words)
    print(f"📊 Word count: {state.word_count}")
    return state


async def check_length(state: ReviewState) -> ReviewState:
    """Check if content meets length requirements"""
    if state.word_count >= 10:
        state.is_approved = True
        state.feedback = "Content length is acceptable."
    else:
        state.is_approved = False
        state.feedback = f"Content too short. Need {10 - state.word_count} more words."
    
    print(f"✅ Approved: {state.is_approved}")
    print(f"💬 Feedback: {state.feedback}")
    return state


async def approve_content(state: ReviewState) -> ReviewState:
    """Approve content for publication"""
    print("✨ Content approved for publication!")
    return state


async def reject_content(state: ReviewState) -> ReviewState:
    """Reject content and request revision"""
    print("❌ Content rejected. Revision needed.")
    return state


# Conditional routing function
def should_approve(state: ReviewState) -> Literal["approve", "reject"]:
    """Route based on approval status"""
    return "approve" if state.is_approved else "reject"


async def main():
    # Build the workflow graph
    graph = StateGraph(ReviewState)

    # Add nodes
    graph.add_node("count", count_words)
    graph.add_node("check", check_length)
    graph.add_node("approve", approve_content)
    graph.add_node("reject", reject_content)

    # Add edges
    graph.set_entry_point("count")
    graph.add_edge("count", "check")
    
    # Add conditional edge based on approval
    graph.add_conditional_edges(
        "check",
        should_approve,
        {
            "approve": "approve",
            "reject": "reject",
        }
    )
    
    # Both approval and rejection lead to end
    graph.add_edge("approve", END)
    graph.add_edge("reject", END)

    # Compile the graph
    workflow = graph.compile()

    # Visualize the graph (optional)
    print("Graph structure:")
    print(workflow.get_mermaid())
    print("\n" + "="*60 + "\n")

    # Test with different inputs
    test_contents = [
        "Hello world!",  # Too short
        "This is a longer piece of content that meets the minimum word count requirement.",  # Long enough
    ]

    for i, content in enumerate(test_contents, 1):
        print(f"Test {i}: {content[:50]}...")
        print("-" * 60)
        
        result = await workflow.run(ReviewState(content=content))
        print(f"\nFinal state - Approved: {result.is_approved}")
        print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
