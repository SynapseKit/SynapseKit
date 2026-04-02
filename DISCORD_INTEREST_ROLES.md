# SynapseKit Discord Interest Roles (Buttons in #roles)

Draft for issue #393.

## Tool
Use **Carl-bot** (Reaction Roles / Button Roles).

Carl-bot dashboard: https://carl.gg

---

## Message to pin in #roles

```
Pick what you're building with SynapseKit.
You can select multiple.

🗂  RAG Builder        — pipelines, loaders, retrievers
🤖  Agent Builder      — tools, ReAct, function calling
🔀  Graph Builder      — StateGraph, workflows, HITL
🧠  LLM Explorer       — providers, caching, cost routing
📖  Docs / Writer      — docs improvements, examples
```

---

## Setup steps

1. Carl-bot dashboard → Reaction Roles → Create new.
2. Paste the message above, add emoji → role mappings.
3. Set mode to **Toggle** (click once to add, again to remove).
4. Post in `#roles` with **send-as-bot** enabled.
5. Lock `#roles` so only the bot can post (members interact via buttons only).

---

## Why this matters

Lets us @-mention specific groups (e.g., **@RAG Builder**) instead of @everyone.
