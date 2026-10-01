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
    SemanticSpanResolution,
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
- Each item shows the block text, its B3 semantic spans with source line
  ranges and ML probabilities, the reasons it was escalated for review, and
  neighbouring blocks' labels. A mixed block may contain more than one section.
- Prefer EXPERIENCE when a block contains a job title/company/date range
  and responsibilities, even if there is no WORK EXPERIENCE heading.
- Use "unknown" only when the block is genuinely unclassifiable.
- For a mixed block, also return "span_verifications": [{{"start_line_id": ..., "end_line_id": ..., "section": ..., "confidence": 0.0-1.0, "reason": ...}}] for each supplied span.
- Do not merge spans. Keep each span's source line range intact.
- Reply with ONLY a JSON object mapping block id (string) to an object
  {{"section": ..., "confidence": 0.0-1.0, "reason": "short factual explanation", "span_verifications": []}}.

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
    OPENROUTER_ENV_VAR = "OPENROUTER_API_KEY"
    OPENROUTER_BASE_URL = "https://openrouter.ai/api"
    DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
    DEFAULT_OPENROUTER_MODEL = "anthropic/claude-haiku-4.5"

    def __init__(self, config: LLMConfig):
        import os

        self.config = config
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "LLM verification needs the anthropic package: pip install anthropic"
            ) from exc
        openrouter_key = os.getenv(self.OPENROUTER_ENV_VAR)
        if openrouter_key:
            self.provider = "openrouter"
            api_key = openrouter_key
            self.base_url = self.OPENROUTER_BASE_URL
            self.model = (self.DEFAULT_OPENROUTER_MODEL
                          if config.model == self.DEFAULT_ANTHROPIC_MODEL else config.model)
        else:
            self.provider = "anthropic"
            api_key = os.getenv(self.ENV_VAR)
            self.base_url = None
            self.model = config.model
            if not api_key:
                raise RuntimeError(
                    f"Set {self.OPENROUTER_ENV_VAR} for OpenRouter or {self.ENV_VAR} for Anthropic. "
                    "Unverified blocks remain explicitly unresolved without an LLM key."
                )
        client_options = {"api_key": api_key, "timeout": config.timeout_seconds}
        if self.base_url:
            client_options["base_url"] = self.base_url
        self._client = anthropic.Anthropic(**client_options)

    def complete(self, prompt: str) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.config.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return str(resp.content[0].text)


class NvidiaNIMLLMClient(LLMClient):
    """OpenAI-compatible client for NVIDIA's hosted NIM inference endpoint."""

    ENV_VAR = "NVIDIA_API_KEY"
    BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
    DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"

    def __init__(self, config: LLMConfig):
        import os
        self.config = config
        self.model = config.model if config.model != AnthropicLLMClient.DEFAULT_ANTHROPIC_MODEL else self.DEFAULT_MODEL
        self.api_key = os.getenv(self.ENV_VAR)
        if not self.api_key:
            raise RuntimeError(f"Set {self.ENV_VAR} in the environment to use NVIDIA NIM.")

    def complete(self, prompt: str) -> str:
        import socket
        import time
        import urllib.error
        import urllib.request

        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.config.max_tokens,
            "temperature": 0.1,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode("utf-8")
        request = urllib.request.Request(
            self.BASE_URL, data=body,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            method="POST")
        payload = None
        retryable_statuses = {429, 500, 502, 503, 504}
        max_retries = self.config.transport_retries
        for attempt in range(max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                if exc.code not in retryable_statuses or attempt == max_retries:
                    raise RuntimeError(
                        f"NVIDIA NIM HTTP {exc.code} after {attempt + 1} attempt(s): {detail}"
                    ) from exc
                retry_after = (exc.headers.get("Retry-After") if exc.headers else None)
                try:
                    delay = min(8.0, max(0.0, float(retry_after))) if retry_after else float(2 ** attempt)
                except ValueError:
                    delay = float(2 ** attempt)
                time.sleep(delay)
            except (TimeoutError, socket.timeout) as exc:
                if attempt == max_retries:
                    raise RuntimeError(
                        f"NVIDIA NIM timed out after {attempt + 1} attempt(s) "
                        f"({self.config.timeout_seconds:g}s per attempt)."
                    ) from exc
                time.sleep(min(8.0, float(2 ** attempt)))
            except urllib.error.URLError as exc:
                if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                    if attempt == max_retries:
                        raise RuntimeError(
                            f"NVIDIA NIM timed out after {attempt + 1} attempt(s) "
                            f"({self.config.timeout_seconds:g}s per attempt)."
                        ) from exc
                    time.sleep(min(8.0, float(2 ** attempt)))
                    continue
                raise RuntimeError(f"NVIDIA NIM connection failed: {exc.reason}") from exc
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("NVIDIA NIM returned no chat completion content.") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("NVIDIA NIM returned empty chat completion content.")
        return content


class OpenRouterLLMClient(LLMClient):
    """OpenRouter API client for hosted models (NVIDIA Nemotron, Claude, Llama, etc.)."""

    ENV_VAR = "OPENROUTER_API_KEY"
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"

    def __init__(self, config: LLMConfig):
        import os
        self.config = config
        self.model = config.model if config.model and config.model != AnthropicLLMClient.DEFAULT_ANTHROPIC_MODEL else self.DEFAULT_MODEL
        self.api_key = os.getenv(self.ENV_VAR)
        if not self.api_key:
            raise RuntimeError(f"Set {self.ENV_VAR} in the environment to use OpenRouter.")

    def complete(self, prompt: str) -> str:
        import socket
        import time
        import urllib.error
        import urllib.request

        req_max_tokens = min(self.config.max_tokens, 1500)
        payload_dict: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": req_max_tokens,
            "temperature": 0.1,
            "reasoning": {
                "effort": "low",
                "exclude": True,
            },
            "chat_template_kwargs": {"enable_thinking": False},
        }
        body = json.dumps(payload_dict).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/resume-parser",
            "X-Title": "Resume Section Classifier",
        }
        request = urllib.request.Request(self.BASE_URL, data=body, headers=headers, method="POST")
        payload = None
        retryable_statuses = {429, 500, 502, 503, 504}
        max_retries = self.config.transport_retries
        for attempt in range(max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"), strict=False)
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                if exc.code == 402 and attempt < max_retries:
                    m_afford = re.search(r"can only afford (\d+)", detail)
                    if m_afford:
                        affordable = max(256, int(m_afford.group(1)) - 100)
                        payload_dict["max_tokens"] = affordable
                        body = json.dumps(payload_dict).encode("utf-8")
                        request = urllib.request.Request(self.BASE_URL, data=body, headers=headers, method="POST")
                        time.sleep(0.5)
                        continue
                if exc.code not in retryable_statuses or attempt == max_retries:
                    raise RuntimeError(
                        f"OpenRouter HTTP {exc.code} after {attempt + 1} attempt(s): {detail}"
                    ) from exc
                retry_after = (exc.headers.get("Retry-After") if exc.headers else None)
                try:
                    delay = min(8.0, max(0.0, float(retry_after))) if retry_after else float(2 ** attempt)
                except ValueError:
                    delay = float(2 ** attempt)
                time.sleep(delay)
            except (TimeoutError, socket.timeout) as exc:
                if attempt == max_retries:
                    raise RuntimeError(
                        f"OpenRouter timed out after {attempt + 1} attempt(s) "
                        f"({self.config.timeout_seconds:g}s per attempt)."
                    ) from exc
                time.sleep(min(8.0, float(2 ** attempt)))
            except urllib.error.URLError as exc:
                if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                    if attempt == max_retries:
                        raise RuntimeError(
                            f"OpenRouter timed out after {attempt + 1} attempt(s) "
                            f"({self.config.timeout_seconds:g}s per attempt)."
                        ) from exc
                    time.sleep(min(8.0, float(2 ** attempt)))
                    continue
                raise RuntimeError(f"OpenRouter connection failed: {exc.reason}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("OpenRouter returned non-dict JSON response.")
        if "error" in payload:
            raise RuntimeError(f"OpenRouter API error: {payload['error']}")
        try:
            choice = payload["choices"][0]
            message = choice.get("message", {})
            content = message.get("content")
            if not content or not str(content).strip():
                content = message.get("reasoning") or message.get("reasoning_content") or choice.get("text")
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("OpenRouter returned no chat completion content.") from exc
        if not isinstance(content, str) or not content.strip():
            finish_reason = choice.get("finish_reason") if "choice" in locals() and isinstance(choice, dict) else None
            raise RuntimeError(f"OpenRouter returned empty chat completion content (finish_reason={finish_reason}).")
        return content


# ---------------------------------------------------------------- prompt + parsing (pure)

def build_resolution_prompt(
    items: list[dict[str, Any]],
) -> str:
    """Render the batched prompt. Pure function of the supplied items."""
    return PROMPT_TEMPLATE.format(items=json.dumps(items, ensure_ascii=False))


def _strip_fences(text: str) -> str:
    # Match markdown code block
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text.strip(), re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.M).strip()


def parse_json_object(text: str) -> dict:
    raw = _strip_fences(text)
    decoder = json.JSONDecoder(strict=False)
    try:
        value = json.loads(raw, strict=False)
    except json.JSONDecodeError:
        # Decode the first balanced object, ignoring surrounding prose/fences.
        start = raw.find("{")
        if start < 0:
            raise ValueError("LLM response contains no JSON object")
        value, _ = decoder.raw_decode(raw[start:])
    if not isinstance(value, dict):
        raise ValueError("LLM response is not a JSON object")
    return value


def complete_json(client, prompt: str) -> dict:
    for attempt in range(2):
        raw = client.complete(prompt if attempt == 0 else prompt +
                              "\nYour previous response was invalid. Return ONLY one strict valid JSON object. Escape quotes and newlines inside strings; no Markdown or commentary.")
        try:
            return parse_json_object(raw)
        except (ValueError, json.JSONDecodeError):
            if attempt:
                raise


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
        data = parse_json_object(text)
    except ValueError as exc:
        return {}, [], f"unparseable LLM response: {exc}"
    entries: dict[str, dict[str, Any]] = {}
    invalid_ids: list[str] = []
    for block_id, value in data.items():
        bid = str(block_id)
        valid = False
        if isinstance(value, dict):
            try:
                section = SectionLabel(value.get("section"))
                conf = value.get("confidence")
                if (isinstance(conf, bool) or not isinstance(conf, (int, float))
                        or not 0.0 <= float(conf) <= 1.0):
                    raise ValueError("invalid block confidence")
                raw_spans = value.get("span_verifications", [])
                if not isinstance(raw_spans, list):
                    raise ValueError("span_verifications must be an array")
                span_verifications = []
                for raw_span in raw_spans:
                    if not isinstance(raw_span, dict):
                        raise ValueError("span verification must be an object")
                    span_section = SectionLabel(raw_span.get("section"))
                    span_confidence = raw_span.get("confidence")
                    start_id, end_id = raw_span.get("start_line_id"), raw_span.get("end_line_id")
                    if (not isinstance(start_id, str) or not isinstance(end_id, str)
                            or isinstance(span_confidence, bool)
                            or not isinstance(span_confidence, (int, float))
                            or not 0.0 <= float(span_confidence) <= 1.0):
                        raise ValueError("invalid span verification fields")
                    span_verifications.append({
                        "start_line_id": start_id, "end_line_id": end_id,
                        "section": span_section, "confidence": float(span_confidence),
                        "reason": raw_span.get("reason", "") if isinstance(raw_span.get("reason", ""), str) else "",
                    })
                reason = value.get("reason", "")
                entries[bid] = {
                    "section": section, "confidence": float(conf),
                    "reason": reason if isinstance(reason, str) else "",
                    "span_verifications": span_verifications,
                }
                valid = True
            except (ValueError, TypeError):
                valid = False
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
            "candidate_semantic_spans": verdict.evidence.get("semantic_spans", []),
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
            raw = client.complete(prompt + "\nReturn ONLY strict valid JSON. Escape all quotes and newlines in strings. No Markdown or commentary.")
            entries, invalid_ids, parse_error = parse_resolution_response(raw)
            error = parse_error
    except Exception as exc:  # transport failure: explicit, never a guess
        error = f"{REASON_CALL_FAILED}: {type(exc).__name__}: {str(exc)[:300]}"

    unknown_in_response = [bid for bid in entries if bid not in set(escalated_ids)]

    for bid in send_ids:
        verdict = by_verdict[bid]
        if bid in entries and not error:
            e = entries[bid]
            requested_spans = verdict.evidence.get("semantic_spans", [])
            span_results = e.get("span_verifications", [])
            expected_ranges = {(x.get("start_line_id"), x.get("end_line_id")) for x in requested_spans}
            returned_ranges = {(x["start_line_id"], x["end_line_id"]) for x in span_results}
            invalid_span_response = bool(span_results and returned_ranges != expected_ranges)
            if not invalid_span_response:
                if not span_results and requested_spans:
                    span_resolutions = [
                        SemanticSpanResolution(
                            start_line_id=s.get("start_line_id", ""),
                            end_line_id=s.get("end_line_id", ""),
                            section=SectionLabel(e["section"]),
                            confidence=e["confidence"],
                            reason=e["reason"],
                        )
                        for s in requested_spans
                    ]
                else:
                    span_resolutions = [SemanticSpanResolution(**span) for span in span_results]
                resolutions.append(BlockResolution(
                    block_id=bid, document_id=seg.document_id,
                    resolved_section=e["section"], confidence=e["confidence"],
                    reason=e["reason"], resolved=True, source="llm",
                    source_line_ids=line_ids.get(bid, []),
                    b4_reasons=list(verdict.reasons),
                    model=config.model, prompt_version=config.prompt_version,
                    span_resolutions=span_resolutions,
                ))
                continue
            invalid_ids.append(bid)
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
