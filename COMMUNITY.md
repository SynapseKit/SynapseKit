# SynapseKit Community Discord Structure

Draft for issue #388.

## Proposed structure

```text
📢 COMMUNITY
  #announcements       — releases, blog posts, major news (admin-only post)
  #changelog           — auto-posted from GitHub releases
  #roadmap             — pinned roadmap, updated manually

👋 START HERE
  #rules               — short, clear rules
  #introductions       — new members introduce themselves
  #roles               — self-assign interest roles (react or button)

💬 GENERAL
  #general             — anything goes
  #show-and-tell       — share what you built with SynapseKit
  #off-topic

🛠 HELP
  #rag-help            — RAG pipelines, loaders, retrievers
  #agents-help         — tools, ReAct, function calling
  #graphs-help         — StateGraph, workflows
  #llm-providers-help  — provider-specific questions
  #general-help        — anything else

👩‍💻 CONTRIBUTORS
  #contributors-chat   — PRs, reviews, coordination
  #good-first-issues   — auto-posted when issues get the label
  #pr-feed             — auto-posted new PRs

📊 STATS (read-only)
  #github-activity     — stars, forks, new issues via webhook
```

## Permissions

- `#announcements` and `#changelog` — members can read, only admins post
- `#contributors-chat` — visible to Contributor role and above
- `#roles` — members interact (buttons/reactions), no free-text messages

## Ordering note

Set up the channel structure first, before configuring bots, so webhooks can target the correct channels from the start.
