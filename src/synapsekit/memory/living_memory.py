"""Living Memory — bidirectional agent memory file management."""

from __future__ import annotations

import inspect
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .diff_engine import FileDiffEngine
from .file_router import MemoryFileRouter
from .living_types import MemoryPatch, PatchStatus
from .patch_store import OccurrenceTracker, PatchStore
from .pii_filter import MemoryPIIFilter

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM prompt for proposing memory patches
# ---------------------------------------------------------------------------
_PROPOSAL_PROMPT = """\
You are a Memory Writer agent. Analyze the session transcript below and propose \
updates to the user's memory files.

## Current Memory Files
{file_contents}

## Session Transcript
{transcript}

## Instructions
1. Identify facts, preferences, corrections, or decisions that should be persisted.
2. For each proposed change, output a JSON object with:
   - "file_path": which file to update
   - "section": which section heading the change belongs under (or "new" for a new section)
   - "fact_key": a short stable identifier for this fact (snake_case, ≤40 chars)
   - "proposed_addition": the markdown text to add or modify
   - "rationale": why this should be recorded
   - "evidence": quote from the transcript supporting this change
3. Only propose changes for durable, reusable information — not one-off requests.
4. Do NOT propose changes that are already present in the memory files.
5. Return a JSON array of proposed changes. If no changes needed, return [].
"""


class LivingMemory:
    """Orchestrate bidirectional memory file management.

    Observes agent sessions, proposes signed diffs to memory files,
    and manages the review / apply / revert lifecycle.

    Parameters
    ----------
    paths:
        Glob patterns or explicit paths to managed memory files.
    proposer:
        An LLM instance used to generate patch proposals.  Should support
        ``generate(prompt)`` or ``agenerate(prompt)``.
    require_approval:
        If True (default), patches are stored as ``"pending"`` for human
        review.  If False, patches are auto-applied after PII filtering.
    sign:
        Whether to cryptographically sign patches.
    signature_secret:
        Secret used for HMAC-style patch signing.
    store_path:
        Path for the JSONL patch store file.
    occurrence_path:
        Path for the occurrence tracker JSON file.
    occurrence_threshold:
        Minimum times a fact must be observed before proposing a patch.
    pii_filter:
        Custom PII filter instance.  Uses default if None.
    file_router:
        Custom file router instance.  Uses default if None.
    """

    def __init__(
        self,
        paths: list[str],
        proposer: Any | None = None,
        *,
        require_approval: bool = True,
        sign: bool = True,
        signature_secret: str = "",
        store_path: str = ".synapsekit_memory_patches.jsonl",
        occurrence_path: str | None = ".synapsekit_memory_occurrences.json",
        occurrence_threshold: int = 3,
        pii_filter: MemoryPIIFilter | None = None,
        file_router: MemoryFileRouter | None = None,
    ) -> None:
        self._raw_paths = paths
        self._proposer = proposer
        self._require_approval = require_approval
        self._sign = sign
        self._secret = signature_secret
        self._occurrence_threshold = occurrence_threshold

        self._store = PatchStore(store_path)
        self._tracker = OccurrenceTracker(occurrence_path)
        self._pii_filter = pii_filter or MemoryPIIFilter()
        self._diff = FileDiffEngine()
        self._router = file_router or MemoryFileRouter(
            primary_path=paths[0] if paths else "./CLAUDE.md"
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def managed_paths(self) -> list[str]:
        """Resolve glob patterns to actual file paths."""
        resolved: list[str] = []
        seen: set[str] = set()
        for raw in self._raw_paths:
            path = Path(raw)
            if "*" in raw or "?" in raw:
                parent = path.parent
                pattern = path.name
                if parent.exists():
                    for match in sorted(parent.glob(pattern)):
                        if match.is_file():
                            norm = str(match)
                            if norm not in seen:
                                seen.add(norm)
                                resolved.append(norm)
            elif path.is_file():
                norm = str(path)
                if norm not in seen:
                    seen.add(norm)
                    resolved.append(norm)
        return resolved

    # ------------------------------------------------------------------
    # Session analysis
    # ------------------------------------------------------------------

    async def propose_from_session(
        self,
        session_id: str,
        *,
        transcript: str | None = None,
        session_records: list[dict[str, Any]] | None = None,
    ) -> list[MemoryPatch]:
        """Analyze a session and propose memory file patches.

        Returns a list of :class:`MemoryPatch` objects.  Their status will
        be ``"pending"`` when *require_approval* is True, otherwise
        ``"applied"``.
        """
        if self._proposer is None:
            _log.warning("No proposer LLM configured — skipping patch proposal")
            return []

        # Build transcript from records if not provided directly
        if transcript is None and session_records:
            transcript = self._format_session_records(session_records)
        if not transcript:
            return []

        file_contents = self._read_managed_files()
        if not file_contents:
            _log.warning("No managed memory files found at configured paths")
            return []

        # Build the proposal prompt
        files_section = self._format_file_contents(file_contents)
        prompt = _PROPOSAL_PROMPT.format(
            file_contents=files_section,
            transcript=transcript[:8000],  # Cap transcript length
        )

        # Get proposals from LLM
        raw_proposals = await self._generate_proposals(prompt)
        if not raw_proposals:
            return []

        # Process each proposal through the pipeline
        patches: list[MemoryPatch] = []
        for proposal in raw_proposals:
            patch = await self._process_single_proposal(
                proposal=proposal,
                file_contents=file_contents,
                session_id=session_id,
            )
            if patch is not None:
                patches.append(patch)

        return patches

    # ------------------------------------------------------------------
    # Review / apply / revert
    # ------------------------------------------------------------------

    def review(self, patch: MemoryPatch | str) -> dict[str, Any]:
        """Prepare a patch for interactive review.

        Returns a dict with the diff, rationale, and review metadata
        suitable for display in a CLI or editor.
        """
        if isinstance(patch, str):
            resolved = self._store.get(patch)
            if resolved is None:
                raise KeyError(f"Patch {patch!r} not found")
            patch = resolved

        stats = self._diff.count_changed_lines(patch.unified_diff)
        return {
            "patch_id": patch.patch_id,
            "file_path": patch.file_path,
            "status": patch.status,
            "rationale": patch.rationale,
            "evidence": patch.evidence_refs,
            "category": patch.category,
            "diff": patch.unified_diff,
            "stats": stats,
            "created_at": patch.created_at,
            "signature_valid": patch.verify(self._secret) if patch.signature else None,
        }

    def apply(self, patch: MemoryPatch | str) -> MemoryPatch:
        """Apply an approved patch to the target file."""
        if isinstance(patch, str):
            resolved = self._store.get(patch)
            if resolved is None:
                raise KeyError(f"Patch {patch!r} not found")
            patch = resolved

        if patch.status not in ("pending", "approved"):
            raise ValueError(
                f"Cannot apply patch {patch.patch_id!r} with status {patch.status!r}"
            )

        # Validate the file hasn't changed
        applicable, reason = self._diff.validate_patch_applicable(
            patch.file_path, patch.before_content
        )
        if not applicable:
            patch.status = "conflict"
            patch.metadata["conflict_reason"] = reason
            self._store.update(patch)
            raise RuntimeError(
                f"Cannot apply patch {patch.patch_id}: {reason}"
            )

        self._apply_patch_to_file(patch)
        return patch

    def revert(self, patch_id: str) -> MemoryPatch:
        """Revert a previously applied patch."""
        patch = self._store.get(patch_id)
        if patch is None:
            raise KeyError(f"Patch {patch_id!r} not found")

        if patch.status != "applied":
            raise ValueError(
                f"Cannot revert patch {patch_id!r} with status {patch.status!r}"
            )

        self._diff.revert_to_content(patch.file_path, patch.before_content)
        patch.status = "reverted"
        patch.reverted_at = datetime.now(timezone.utc).isoformat()
        if self._sign:
            patch.sign(self._secret)
        self._store.update(patch)

        _log.info("Reverted patch %s on %s", patch_id, patch.file_path)
        return patch

    def pending_patches(self) -> list[MemoryPatch]:
        """Return all patches awaiting review."""
        return self._store.pending_patches()

    def patch_history(
        self,
        *,
        status: PatchStatus | None = None,
        limit: int | None = None,
    ) -> list[MemoryPatch]:
        """Return patch history, newest first."""
        return self._store.list_by_status(status, limit=limit)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_managed_files(self) -> dict[str, str]:
        """Read current content of all managed memory files."""
        contents: dict[str, str] = {}
        for file_path in self.managed_paths:
            try:
                contents[file_path] = Path(file_path).read_text(encoding="utf-8")
            except OSError as exc:
                _log.warning("Could not read %s: %s", file_path, exc)
        return contents

    async def _generate_proposals(self, prompt: str) -> list[dict[str, Any]]:
        """Call the proposer LLM and parse the JSON response."""
        try:
            if hasattr(self._proposer, "agenerate"):
                raw = await self._proposer.agenerate(prompt)
            elif hasattr(self._proposer, "generate"):
                result = self._proposer.generate(prompt)
                if inspect.isawaitable(result):
                    raw = await result
                else:
                    raw = result
            else:
                _log.error("Proposer LLM has no generate/agenerate method")
                return []
        except Exception:
            _log.exception("LLM proposal generation failed")
            return []

        return self._parse_proposal_json(str(raw))

    async def _process_single_proposal(
        self,
        proposal: dict[str, Any],
        file_contents: dict[str, str],
        session_id: str,
    ) -> MemoryPatch | None:
        """Transform a raw LLM proposal into a validated MemoryPatch."""
        fact_key = str(proposal.get("fact_key", ""))
        if not fact_key:
            return None

        evidence = str(proposal.get("evidence", ""))
        self._tracker.record_occurrence(fact_key, session_id, evidence)

        # Check occurrence threshold
        if not self._tracker.has_reached_threshold(
            fact_key, self._occurrence_threshold
        ):
            current_count = self._tracker.get_count(fact_key)
            _log.debug(
                "Fact %r has not reached threshold (%d/%d) — deferring",
                fact_key,
                current_count,
                self._occurrence_threshold,
            )
            return None

        # Determine target file
        content_text = str(proposal.get("proposed_addition", ""))
        category = self._router.categorize(content_text)
        target_path = self._router.resolve_target_path(
            category, list(file_contents.keys())
        )

        # Build the patch
        before = file_contents.get(target_path, "")
        after = self._insert_content(
            before,
            content_text,
            section=str(proposal.get("section", "")),
        )

        if before == after:
            return None

        # PII filter
        pii_result = self._pii_filter.filter_content(after)
        after = pii_result.filtered_content

        unified = self._diff.generate_unified_diff(before, after, target_path)
        if not unified.strip():
            return None

        patch = MemoryPatch(
            file_path=target_path,
            before_content=before,
            after_content=after,
            unified_diff=unified,
            rationale=str(proposal.get("rationale", "")),
            evidence_refs=[evidence] if evidence else [],
            session_id=session_id,
            category=category,
            status="pending" if self._require_approval else "applied",
            metadata={
                "fact_key": fact_key,
                "pii_filtered": not pii_result.is_clean,
                "diff_stats": self._diff.count_changed_lines(unified),
            },
        )

        if self._sign:
            patch.sign(self._secret)

        self._store.save(patch)

        # Auto-apply if approval not required
        if not self._require_approval:
            self._apply_patch_to_file(patch)

        return patch

    def _apply_patch_to_file(self, patch: MemoryPatch) -> None:
        """Write the patch content to disk and update status."""
        self._diff.apply_patch(patch.file_path, patch.after_content)
        patch.status = "applied"
        patch.applied_at = datetime.now(timezone.utc).isoformat()
        if self._sign:
            patch.sign(self._secret)
        self._store.update(patch)
        _log.info("Applied patch %s to %s", patch.patch_id, patch.file_path)

    @staticmethod
    def _insert_content(
        existing: str,
        addition: str,
        section: str = "",
    ) -> str:
        """Insert new content into the appropriate section of a file."""
        addition = addition.strip()
        if not addition:
            return existing

        lines = existing.split("\n")

        if section and section != "new":
            # Find the section heading and insert after it
            section_lower = section.lower().strip("# ").strip()
            insert_idx = None
            for i, line in enumerate(lines):
                stripped = line.strip().lower().lstrip("# ").strip()
                if stripped == section_lower:
                    # Find the end of this section (next heading or EOF)
                    insert_idx = i + 1
                    while insert_idx < len(lines):
                        next_line = lines[insert_idx].strip()
                        if next_line.startswith("#"):
                            break
                        insert_idx += 1
                    break

            if insert_idx is not None:
                # If inserting at the end of the file and it has a trailing newline (empty string in lines),
                # insert before it to preserve the single trailing newline properly.
                if insert_idx == len(lines) and len(lines) > 1 and lines[-1] == "":
                    lines.insert(len(lines) - 1, addition)
                else:
                    lines.insert(insert_idx, addition)
                return "\n".join(lines)

        # Default: append at end
        if existing and not existing.endswith("\n"):
            return existing + "\n\n" + addition + "\n"
        return existing + "\n" + addition + "\n"

    @staticmethod
    def _format_file_contents(contents: dict[str, str]) -> str:
        """Format managed file contents for the LLM prompt."""
        parts: list[str] = []
        for path, content in contents.items():
            parts.append(f"### File: `{path}`\n```markdown\n{content}\n```")
        return "\n\n".join(parts)

    @staticmethod
    def _format_session_records(records: list[dict[str, Any]]) -> str:
        """Format session records into a readable transcript."""
        lines: list[str] = []
        for record in records:
            role = record.get("role", "unknown")
            content = record.get("content", "")
            lines.append(f"[{role}]: {content}")
        return "\n".join(lines)

    @staticmethod
    def _parse_proposal_json(text: str) -> list[dict[str, Any]]:
        """Extract JSON array from LLM response text."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("[")
            end = cleaned.rfind("]")
            if start == -1 or end == -1 or end <= start:
                return []
            try:
                parsed = json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                return []

        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            return []
        return [item for item in parsed if isinstance(item, dict)]
