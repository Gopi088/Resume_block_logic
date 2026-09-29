"""B5: LLM Fallback Resolution.

B5 answers: "What section is this AMBIGUOUS block?" It runs ONLY for blocks
B4 escalated (needs_llm=True). Accepted blocks are never sent, never modified.

Cost and scope discipline:
- One batched LLM call per document (all escalated blocks in a single prompt).
- Each block carries limited context only: its text/heading, B3 prediction +
  alternatives, B4 escalation reasons, and neighbouring blocks' labels.
- The LLM returns a JSON mapping; every entry is validated. Invalid entries,
  missing entries, and transport failures all become EXPLICITLY unresolved
  blocks — never silent guesses.
- No API key / no provider package -> loud, actionable RuntimeError at client
  construction (operator error), not per-block ambiguity.
- No B6 merging here: accepted + resolved are combined downstream.

Seams: LLMClient.complete(prompt) is the only impure boundary. Production
uses AnthropicLLMClient; tests use ScriptedLLMClient (deterministic double).
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

from .models import (
    BlockResolution,
    IntegrityReport,
    LLMConfig,
    LLMResolutionResult,
    SectionLabel,
    SegmentationResult,
    ValidationResult,
)

ALLOWED_SECTIONS = [s.value for s in SectionLabel]

PROMPT_TEMPLATE = """You are validating resume blocks. For EACH item below, decide its section.

Choose exactly one section per block from:
contact, summary, experience, education, skills, projects,
certifications, awards, publications, languages, volunteering,
interests, references, other, unknown.

Rules:
- Do not invent content. Classify each supplied block only.
- Each item shows the block text, the ML prediction with alternatives, the
  reasons it was escalated for review, and neighbouring blocks' labels.
- Prefer EXPERIENCE when a block contains a job title/company/date range
  and responsibilities, even if there is no WORK EXPERIENCE heading.
- Use "unknown" only when the block is genuinely unclassifiable.
- Reply with ONLY a JSON object mapping block id (string) to an object
  {{"section": ..., "confidence": 0.0-1.0, "reason": "short factual explanation"}}.

Schema example:
{{"B000003": {{"section": "experience", "confidence": 0.9, "reason": "..."}}}}

Items:
{items}
"""

REASON_NO_ANSWER = "llm_no_answer_for_block"
REASON_INVALID = "llm_invalid_answer"
REASON_CALL_FAILED = "llm_call_failed"
REASON_BUDGET = "llm_budget_exceeded"


# ---------------------------------------------------------------- clients

class LLMClient(ABC):
    """Transport seam: prompt in, raw text out. Parsing lives here, not in clients."""

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Return the model's raw text response (may raise on transport errors)."""
        raise NotImplementedError


class ScriptedLLMClient(LLMClient):
    """Deterministic test double. NEVER used in production.

    `responses` maps block_id -> {"section":..., "confidence":..., "reason":...}
    (or a raw string to simulate malformed output). Every prompt is recorded
    in `prompts` for assertions.
    """

    def __init__(self, responses: dict[str, dict[str, Any]] | str):
        self.responses = responses
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if isinstance(self.responses, str):
            return self.responses
        return json.dumps(self.responses)


class AnthropicLLMClient(LLMClient):
    """Production transport over the Anthropic messages API.

    Key comes ONLY from the ANTHROPIC_API_KEY environment variable.
    """

    ENV_VAR = "ANTHROPIC_API_KEY"

    def __init__(self, config: LLMConfig):
        import os

        self.config = config
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "resolve with Anthropic needs the anthropic package: pip install anthropic"
            ) from exc
        api_key = os.getenv(self.ENV_VAR)
        if not api_key:
            raise RuntimeError(
                f"resolve with Anthropic needs {self.ENV_VAR} in the environment "
                "(escalated blocks stay explicitly unresolved without it)."
            )
        self._client = anthropic.Anthropic(api_key=api_key, timeout=config.timeout_seconds)

    def complete(self, prompt: str) -> str:
        resp = self._client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return str(resp.content[0].text)


# ---------------------------------------------------------------- prompt + parsing (pure)

def build_resolution_prompt(
    items: list[dict[str, Any]],
) -> str:
    """Render the batched prompt. Pure function of the supplied items."""
    return PROMPT_TEMPLATE.format(items=json.dumps(items, ensure_ascii=False))


def _strip_fences(text: str) -> str:
    return re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.M).strip()


def parse_resolution_response(
    text: str,
) -> tuple[dict[str, dict[str, Any]], list[str], str]:
    """Parse raw LLM output.

    Returns (entries, invalid_ids, error). entries maps block_id -> validated
    {"section": SectionLabel, "confidence": float, "reason": str}.
    invalid_ids were present but failed validation (bad section/confidence).
    error is non-empty iff the whole response was unusable.
    """
    try:
        data = json.loads(_strip_fences(text))
    except (json.JSONDecodeError, ValueError) as exc:
        return {}, [], f"unparseable LLM response: {exc}"
    if not isinstance(data, dict):
        return {}, [], "LLM response is not a JSON object"
    entries: dict[str, dict[str, Any]] = {}
    invalid_ids: list[str] = []
    for block_id, value in data.items():
        bid = str(block_id)
        valid = False
        if isinstance(value, dict):
            try:
                section = SectionLabel(value.get("section"))
            except ValueError:
                section = None
            conf = value.get("confidence")
            if section is not None and not isinstance(conf, bool) \
                    and isinstance(conf, (int, float)) and 0.0 <= float(conf) <= 1.0:
                reason = value.get("reason", "")
                entries[bid] = {
                    "section": section,
                    "confidence": float(conf),
                    "reason": reason if isinstance(reason, str) else "",
                }
                valid = True
        if not valid:
            invalid_ids.append(bid)
    return entries, invalid_ids, ""


# ---------------------------------------------------------------- orchestration

def _request_items(
    seg: SegmentationResult,
    val: ValidationResult,
    escalated_ids: list[str],
) -> list[dict[str, Any]]:
    by_block = {b.block_id: b for b in seg.blocks}
    verdicts = {v.block_id: v for v in val.verdicts}
    items = []
    for bid in escalated_ids:
        block = by_block[bid]
        verdict = verdicts[bid]
        # Neighbour context: predicted labels of ADJACENT verdicts in document
        # order (limited context, not full text).
        order = [v.block_id for v in val.verdicts]
        idx = order.index(bid)
        prev_label = val.verdicts[idx - 1].predicted_section.value if idx > 0 else None
        next_label = (
            val.verdicts[idx + 1].predicted_section.value
            if idx + 1 < len(val.verdicts) else None
        )
        items.append({
            "id": bid,
            "heading": block.display_lines[0].text if block.display_lines and block.display_lines[0].is_header else None,
            "text": block.text,
            "ml_prediction": verdict.predicted_section.value,
            "ml_confidence": verdict.confidence,
            "alternatives": [
                {"section": a.section.value, "confidence": a.confidence}
                for a in verdict.alternatives
            ],
            "escalation_reasons": list(verdict.reasons),
            "previous_block_label": prev_label,
            "next_block_label": next_label,
        })
    return items


def resolve_escalated(
    seg: SegmentationResult,
    val: ValidationResult,
    client: LLMClient,
    config: LLMConfig | None = None,
) -> "LLMResolutionResult":
    """Resolve B4-escalated blocks via the LLM client. Accepted blocks untouched."""
    config = config or LLMConfig()
    escalated = [v for v in val.verdicts if v.needs_llm]
    escalated_ids = [v.block_id for v in escalated]
    by_verdict = {v.block_id: v for v in val.verdicts}
    line_ids = {b.block_id: list(b.line_ids) for b in seg.blocks}

    resolutions: list[BlockResolution] = []
    error = ""
    unknown_in_response: list[str] = []

    if not escalated_ids:
        return LLMResolutionResult(
            document_id=seg.document_id, resolutions=[],
            n_sent=0, n_resolved=0, n_unresolved=0,
            provider=config.provider, model=config.model,
            prompt_version=config.prompt_version, error="",
            integrity=IntegrityReport(passed=True, violations=[],
                                      checks={"no_escalated_blocks": True}),
        )

    send_ids = escalated_ids[: config.max_escalated_blocks]
    excess_ids = escalated_ids[config.max_escalated_blocks :]

    entries: dict[str, dict[str, Any]] = {}
    invalid_ids: list[str] = []
    try:
        prompt = build_resolution_prompt(_request_items(seg, val, send_ids))
        raw = client.complete(prompt)
        entries, invalid_ids, parse_error = parse_resolution_response(raw)
        if parse_error:
            error = parse_error
    except Exception as exc:  # transport failure: explicit, never a guess
        error = f"{REASON_CALL_FAILED}: {type(exc).__name__}: {str(exc)[:300]}"

    unknown_in_response = [bid for bid in entries if bid not in set(escalated_ids)]

    for bid in send_ids:
        verdict = by_verdict[bid]
        if bid in entries and not error:
            e = entries[bid]
            resolutions.append(BlockResolution(
                block_id=bid, document_id=seg.document_id,
                resolved_section=e["section"], confidence=e["confidence"],
                reason=e["reason"], resolved=True, source="llm",
                source_line_ids=line_ids.get(bid, []),
                b4_reasons=list(verdict.reasons),
                model=config.model, prompt_version=config.prompt_version,
            ))
            continue
        # Explicitly unresolved: whole-call failure, invalid entry, or silence.
        if error:
            reason = error
        elif bid in invalid_ids:
            reason = REASON_INVALID
        else:
            reason = REASON_NO_ANSWER
        resolutions.append(BlockResolution(
            block_id=bid, document_id=seg.document_id,
            resolved_section=None, confidence=0.0, reason=reason,
            resolved=False, source="unresolved",
            source_line_ids=line_ids.get(bid, []),
            b4_reasons=list(verdict.reasons),
            model="", prompt_version=config.prompt_version,
        ))

    for bid in excess_ids:
        verdict = by_verdict[bid]
        resolutions.append(BlockResolution(
            block_id=bid, document_id=seg.document_id,
            resolved_section=None, confidence=0.0, reason=REASON_BUDGET,
            resolved=False, source="unresolved",
            source_line_ids=line_ids.get(bid, []),
            b4_reasons=list(verdict.reasons),
            model="", prompt_version=config.prompt_version,
        ))

    n_resolved = sum(1 for r in resolutions if r.resolved)
    integrity = build_resolution_integrity(
        escalated_ids, send_ids, resolutions, unknown_in_response, val)
    return LLMResolutionResult(
        document_id=seg.document_id, resolutions=resolutions,
        n_sent=len(send_ids), n_resolved=n_resolved,
        n_unresolved=len(resolutions) - n_resolved,
        provider=config.provider, model=config.model,
        prompt_version=config.prompt_version, error=error,
        integrity=integrity,
    )


def build_resolution_integrity(
    escalated_ids: list[str],
    sent_ids: list[str],
    resolutions: list[BlockResolution],
    unknown_in_response: list[str],
    val: ValidationResult,
) -> IntegrityReport:
    """B5 integrity: every escalated block resolved-or-explicit; accepted untouched."""
    violations: list[str] = []
    checks: dict[str, bool] = {}

    ok_cover = [r.block_id for r in resolutions] == escalated_ids
    checks["every_escalated_block_accounted_in_order"] = ok_cover
    if not ok_cover:
        violations.append(
            f"resolution coverage mismatch: {len(escalated_ids)} escalated, "
            f"{len(resolutions)} resolutions."
        )

    accepted_ids = {v.block_id for v in val.verdicts if not v.needs_llm}
    leaked = sorted({r.block_id for r in resolutions} & accepted_ids)
    checks["no_accepted_block_sent_or_modified"] = not leaked
    if leaked:
        violations.append(f"resolutions touch accepted blocks: {leaked}")

    ok_state = all(
        (r.resolved and r.resolved_section is not None and r.source == "llm")
        or (not r.resolved and r.resolved_section is None and r.source == "unresolved"
            and bool(r.reason))
        for r in resolutions
    )
    checks["resolved_state_consistent"] = ok_state
    if not ok_state:
        violations.append("resolved/unresolved state inconsistent (section/source/reason).")

    ok_conf = all(0.0 <= r.confidence <= 1.0 for r in resolutions)
    checks["confidence_range_valid"] = ok_conf
    if not ok_conf:
        violations.append("resolution confidence outside [0, 1].")

    checks["no_unknown_response_blocks"] = not unknown_in_response
    if unknown_in_response:
        violations.append(f"LLM answered for unknown blocks (ignored): {sorted(unknown_in_response)}")

    return IntegrityReport(passed=not violations, violations=violations, checks=checks)


__all__ = [
    "AnthropicLLMClient",
    "LLMClient",
    "ScriptedLLMClient",
    "build_resolution_integrity",
    "build_resolution_prompt",
    "parse_resolution_response",
    "resolve_escalated",
]
