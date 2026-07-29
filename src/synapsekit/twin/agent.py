from __future__ import annotations

import logging
from collections.abc import Sequence

from synapsekit.llm._factory import make_llm
from synapsekit.llm.base import BaseLLM
from synapsekit.twin.delegation import (
    ApprovalRequiredError,
    AutoSendForbiddenError,
    DelegationPolicy,
    DraftResult,
)
from synapsekit.twin.style_profile import LearnedPatterns, StyleProfile
from synapsekit.twin.voice_matcher import VoiceMatcher, VoiceMatchResult

logger = logging.getLogger(__name__)


class DigitalTwinAgent:
    """Agent that learns human workflow patterns and drafts content in their voice."""

    def __init__(
        self,
        profile_path: str = "~/.synapsekit/twin/style.md",
        delegation: DelegationPolicy | None = None,
        reference_samples: Sequence[str] | None = None,
        *,
        llm: BaseLLM | None = None,
        model: str = "gpt-4o-mini",
        api_key: str = "",
        provider: str | None = None,
    ) -> None:
        self.profile = StyleProfile(profile_path)
        self.delegation = delegation or DelegationPolicy()
        self.reference_samples: list[str] = list(reference_samples) if reference_samples else []

        self.llm: BaseLLM | None = None
        if llm is not None:
            self.llm = llm
        else:
            try:
                self.llm = make_llm(
                    model,
                    api_key,
                    provider,
                    "You are a digital twin drafting messages in the user's authentic style.",
                    0.3,
                    1024,
                )
            except Exception as err:
                logger.debug("Could not initialize default LLM for DigitalTwinAgent: %s", err)
                self.llm = None

        self.voice_matcher = VoiceMatcher(llm=self.llm)

    async def learn(self, samples: Sequence[str]) -> LearnedPatterns:
        """Learn writing patterns from a set of human-written text samples."""
        self.reference_samples.extend(samples)
        return await self.profile.update_from_samples(self.reference_samples)

    async def draft_commit_message(self, diff: str) -> DraftResult:
        """Draft a git commit message matching user voice."""
        prompt = (
            f"Draft a concise git commit message for the following diff. "
            f"Style: tone={self.profile.patterns.tone}, structure={self.profile.patterns.structure}.\n\n"
            f"Diff:\n{diff}"
        )
        return await self.draft("commit_messages", prompt)

    async def draft_pr_description(self, diff: str, title: str = "") -> DraftResult:
        """Draft a PR description matching user voice."""
        title_context = f"Title: {title}\n" if title else ""
        prompt = (
            f"Draft a PR description for the following changes.\n{title_context}"
            f"Style preferences:\n"
            f"- Tone: {self.profile.patterns.tone}\n"
            f"- Structure: {self.profile.patterns.structure}\n"
            f"- Preferred vocabulary: {self.profile.patterns.vocabulary}\n\n"
            f"Diff:\n{diff}"
        )
        return await self.draft("pr_descriptions", prompt)

    async def draft_review(self, diff: str, context: str = "") -> DraftResult:
        """Draft a code review matching user voice."""
        ctx_text = f"Context: {context}\n" if context else ""
        prompt = (
            f"Draft a code review comment in the user's style.\n{ctx_text}"
            f"Review style: {self.profile.patterns.review_style}, tone: {self.profile.patterns.tone}.\n\n"
            f"Diff:\n{diff}"
        )
        return await self.draft("pr_reviews", prompt)

    async def draft(self, channel: str, prompt_or_content: str) -> DraftResult:
        """Generic drafting method enforcing delegation policy and voice matching."""
        level = self.delegation.get_level(channel)
        requires_approval = not self.delegation.can_auto_send(channel)

        generated_text = ""
        if self.llm is not None:
            try:
                chunks: list[str] = []
                async for chunk in self.llm.stream(prompt_or_content):
                    chunks.append(chunk)
                generated_text = "".join(chunks).strip()
            except Exception as err:
                logger.warning("LLM stream failed during drafting: %s", err)

        if not generated_text:
            generated_text = self._fallback_draft(channel, prompt_or_content)

        # Evaluate match
        match_res = await self.voice_matcher.evaluate(
            generated_text, self.reference_samples, self.profile.patterns
        )

        attribution = f"drafted by twin v{self.profile.version}"

        return DraftResult(
            content=generated_text,
            channel=channel,
            delegation_level=level,
            requires_approval=requires_approval,
            twin_version=self.profile.version,
            attribution=attribution,
            confidence=match_res.score,
            reference_samples_used=len(self.reference_samples),
        )

    async def evaluate_voice_match(self, candidate: str) -> VoiceMatchResult:
        """Evaluate voice match score of a candidate text against profile and samples."""
        return await self.voice_matcher.evaluate(
            candidate, self.reference_samples, self.profile.patterns
        )

    async def send(
        self,
        draft_result: DraftResult,
        channel: str | None = None,
        *,
        approved: bool = False,
    ) -> DraftResult:
        """Enforce the delegation gate before a draft may be dispatched.

        - ``never_send_auto`` channels always raise ``AutoSendForbiddenError``.
        - ``draft_with_approval`` channels require ``approved=True`` (an explicit
          human approval token) or raise ``ApprovalRequiredError``.
        - ``draft`` channels may be sent freely.

        The default (``approved=False``) is safe: any gated channel refuses.
        Returns the ``draft_result`` unchanged on success so callers may chain.
        """
        target = channel or draft_result.channel

        if self.delegation.is_send_forbidden(target):
            raise AutoSendForbiddenError(
                f"Channel '{target}' is gated as never_send_auto; "
                "auto-sending is forbidden and requires a manual human send."
            )

        if self.delegation.requires_human_approval(target) and not approved:
            raise ApprovalRequiredError(
                f"Channel '{target}' requires explicit human approval; "
                "pass approved=True to send."
            )

        return draft_result

    def _fallback_draft(self, channel: str, prompt: str) -> str:
        patterns = self.profile.patterns
        if channel in ("commit_messages", "commit"):
            return f"feat: update implementation ({patterns.tone} style)"
        elif channel in ("pr_descriptions", "pr"):
            return (
                f"## Summary\n"
                f"Automated draft by DigitalTwinAgent.\n\n"
                f"## Key Changes\n"
                f"- Implementation details updated according to style profile v{self.profile.version}.\n"
            )
        elif channel in ("pr_reviews", "review"):
            return f"LGTM! Changes look good from a {patterns.review_style} perspective."
        return f"Draft content for {channel}"
