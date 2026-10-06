## Overview
This PR introduces the `SemanticCompressorMemory`, an advanced memory backend that automatically summarizes older parts of a conversation into dense, structured blocks when the context window reaches a predefined token threshold.

## Why?
Long-running agents (like coding assistants or roleplay bots) eventually fill up the context window. 
Traditionally, memory managers either use a naive sliding window (which deletes the oldest messages causing "amnesia") or simply crash. 

The `SemanticCompressorMemory` solves this by using a fast, cheap LLM (like Llama 3 8B or GPT-4o-mini) to dynamically merge the existing summary with new exchanges, preserving core instructions, critical entities, and ongoing state indefinitely—while staying perfectly within token bounds and saving massive API costs on larger models.

## Features
- **Dynamic Compression**: Tracks token limits natively and intelligently grabs the oldest half of the active context to compress into a dense block when a configurable threshold is breached.
- **Cost Saving**: Prevents expensive LLMs from constantly reading identical, verbose older histories.
- **Extensible**: Simply pass any `BaseLLM` to the `llm` argument of the compressor.

## Changes
- Built `src/synapsekit/memory/semantic_compressor.py`.
- Exported in `src/synapsekit/memory/__init__.py`.
- Wrote full unit test coverage using a mocked compressor VLM in `tests/memory/test_semantic_compressor.py`.
