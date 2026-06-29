# Computer Use Safety

Computer-use agents can click, type, navigate, submit forms, and operate apps. Treat them as automation running with the same permissions as the host user.

## Required Defaults

- Run risky tasks in an isolated browser profile, VM, container, or throwaway remote desktop.
- Keep credential managers, password dialogs, keychains, payment apps, and admin consoles in `SafetyPolicy.forbidden_apps`.
- Use `allowed_domains` for browser workflows and require confirmation before navigating outside the expected domain set.
- Require human confirmation for destructive or externally visible actions such as delete, send, purchase, submit, transfer, pay, or password entry.
- Record sessions for debugging and auditability when handling business workflows.

## Safe Policy Example

```python
from synapsekit import SafetyPolicy

safety = SafetyPolicy(
    confirm_before=["delete", "send", "purchase", "submit", "transfer", "pay"],
    forbidden_apps=["1password", "bitwarden", "keychain", "lastpass"],
    allowed_domains=["example.com", "internal.example.com"],
    blocked_domains=["bank.example", "admin.example"],
    record_session=True,
)
```

## Human Confirmation

`ComputerUseAgent` fails closed when a `SafetyPolicy` requires confirmation and no confirmation callback is configured.

```python
async def confirm(action, observation) -> bool:
    return await show_review_dialog(action, observation)
```

Use confirmation for any step that changes external state: sending messages, submitting orders, deleting records, updating production data, or entering secrets.

## Session Recording

`SessionRecorder` writes JSONL events containing observations, normalized actions, safety decisions, outputs, and errors. Use recordings to replay provider behavior in tests and to diagnose bad screen grounding.

Do not store recordings that contain secrets, regulated data, private messages, or customer information unless your retention and access controls allow it.

## Provider Isolation

Provider adapters normalize Anthropic, OpenAI, and open-source action payloads into `ComputerAction`. Unknown or malformed actions are rejected instead of being executed. Prefer model prompts that request visual grounding such as "click the button labeled Submit" over raw pixel-only instructions when the screen provider supports semantic selectors.

## Production Checklist

- Use least-privilege OS users and application accounts.
- Disable access to password managers and browser-saved credentials.
- Restrict browser downloads and file-system access.
- Log safety decisions and review blocked actions.
- Start with deterministic browser or VNC environments before local desktop automation.
- Add workflow-specific tests for malicious instructions such as "delete all files" or "send credentials".
