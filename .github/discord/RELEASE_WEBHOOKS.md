# SynapseKit Discord Release Webhooks

Draft for issue #390.

## What this does

Automatically posts new GitHub releases to Discord so `#changelog` stays up to date without manual copy-pasting.

## Setup

1. In GitHub, go to the repo → Settings → Webhooks → Add webhook.
2. Set the payload URL to the Discord channel webhook URL.
3. Use content type `application/json`.
4. Select individual events and enable only `Releases`.
5. Save the webhook.

## Discord result

A published release should show something like:

> 🚀 **New release: v1.4.6**
> Subgraph error handling — retry, fallback, skip
> [View release](https://github.com/SynapseKit/SynapseKit/releases/tag/v1.4.6)

## Optional extra

If you want a wider announcement, point the same GitHub release webhook at `#announcements` too.

## Channel target

`#changelog` is the main target for this automation.
