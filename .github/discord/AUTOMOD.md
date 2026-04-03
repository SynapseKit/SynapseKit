# SynapseKit Discord Automod & Spam Protection

Draft for issue #394.

## Tool
Use **Carl-bot** (Automod) — free, no code.

Carl-bot dashboard: https://carl.gg

---

## Automod rules to configure

| Rule | Action | Target |
|---|---|---|
| New account < 7 days old | Kick + DM warning | @everyone |
| Posting links in first 10 minutes | Delete + warn | New members |
| More than 5 messages in 5 seconds | Mute 10 min + log | @everyone |
| Discord invite links | Delete + warn | @everyone (except Moderator+) |
| Spam caps (ALL CAPS > 70%) | Delete | @everyone |
| Known phishing domains | Ban + log | @everyone |

---

## Logging

Create a private **#mod-log** channel (Admin + Moderator only) and point Carl-bot automod logs there.

---

## Slowmode

- **#general** — 5s slowmode
- **#introductions** — 10s slowmode
- **Help channels** — no slowmode

---

## Setup path

Carl-bot dashboard → Automod → enable each rule. Takes about 10 minutes.
