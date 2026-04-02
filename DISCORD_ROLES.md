# SynapseKit Discord Roles & Permissions

Draft for issue #389.

## Role hierarchy

| Role | Color | Who gets it | How |
|---|---|---|---|
| Admin | Red | Amit, Abhay | Manual |
| Moderator | Orange | Trusted members later | Manual |
| Contributor | Green | Anyone with a merged PR | Manual for now; see #391 for automation |
| Member | Blue/grey | Everyone on join | Auto via Carl-bot welcome |
| Bot | Dark grey | All bots | Manual |

## Interest roles

These are self-assign in `#roles` and do not unlock channels:

- 🗂 RAG Builder
- 🤖 Agent Builder
- 🔀 Graph Builder
- 🧠 LLM Explorer
- 📖 Docs / Writer

## Setup steps

1. Create the roles in Server Settings → Roles.
2. Set the role colors.
3. Order them as: Admin > Moderator > Contributor > Member > @everyone.
4. Lock `#contributors-chat` to Contributor+ only.
5. Lock `#announcements` and `#changelog` to admin-only posting.

## Permissions

- `#contributors-chat` — visible to Contributor, Moderator, and Admin.
- `#announcements` — visible to everyone, posting restricted to Admin only.
- `#changelog` — visible to everyone, posting restricted to Admin only.
