from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DelegationLevel = Literal["draft", "draft_with_approval", "never_send_auto"]


@dataclass
class DelegationPolicy:
    commit_messages: DelegationLevel = "draft"
    pr_descriptions: DelegationLevel = "draft"
    pr_reviews: DelegationLevel = "draft_with_approval"
    emails: DelegationLevel = "never_send_auto"

    def get_level(self, channel: str) -> DelegationLevel:
        if channel in ("commit_messages", "commit", "commit_message"):
            return self.commit_messages
        elif channel in ("pr_descriptions", "pr_description", "pr_desc"):
            return self.pr_descriptions
        elif channel in ("pr_reviews", "pr_review", "review"):
            return self.pr_reviews
        elif channel in ("emails", "email"):
            return self.emails
        return "draft_with_approval"


@dataclass
class DraftResult:
    content: str
    channel: str
    delegation_level: DelegationLevel
    requires_approval: bool
    twin_version: int
    attribution: str
    confidence: float = 0.0
    reference_samples_used: int = 0

    def to_dict(self) -> dict[str, str | bool | int | float]:
        return {
            "content": self.content,
            "channel": self.channel,
            "delegation_level": self.delegation_level,
            "requires_approval": self.requires_approval,
            "twin_version": self.twin_version,
            "attribution": self.attribution,
            "confidence": self.confidence,
            "reference_samples_used": self.reference_samples_used,
        }
