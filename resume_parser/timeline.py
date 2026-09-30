"""B9: Timeline + Gap Detection.

B9 answers: "What is the chronological timeline and where are the gaps?" It
is the final stage: no further parsing, no ML, no LLM, no forecasting.

What B9 does:
- Places every B8-dated entry on one timeline, sorted by start date
  (ties broken by end date, then entry id — fully deterministic).
- Resolves comparison dates granularity-aware: year-only starts -> Jan 1,
  year-only ends -> Dec 31; month precision uses month boundaries; ongoing
  ends ("Present") resolve to the reference date, recorded per event as
  effective_end_date while end_date stays None (lexical fact preserved).
- Detects employment gaps over EXPERIENCE events only: walking start-sorted
  experience ranges with running max-end coverage, any uncovered stretch of
  >= policy.min_gap_days becomes a CareerGap. Overlapping/concurrent jobs
  extend coverage (never gaps). Education and other dated sections appear in
  the timeline but never bridge employment gaps (documented v1 scope).
- Lists undated entries explicitly (undated_entry_ids). They are not placed,
  not dropped, and never treated as zero-length events.
- Inherits the reference date from B8's recorded value when the policy does
  not pin one, so Present resolves identically across stages.

What B9 does NOT do: durations, totals, scores, ratings, predictions,
recommendations, or cross-entry date invention. A gap is a measured
uncovered stretch between stated employment periods — nothing more.
"""

from __future__ import annotations

import calendar
from datetime import date

from .models import (
    CareerGap,
    DateExtractionResult,
    EntrySegmentationResult,
    IntegrityReport,
    SectionLabel,
    TimelineEvent,
    TimelinePolicy,
    TimelineResult,
)


def _start_cmp(year: int, month: int | None, day: int | None) -> date:
    """Earliest possible calendar date for a start endpoint."""
    return date(year, month or 1, day or 1)


def _end_cmp(year: int, month: int | None, day: int | None) -> date:
    """Latest possible calendar date for an end endpoint."""
    m = month or 12
    if day is not None:
        d = day
    else:
        d = calendar.monthrange(year, m)[1]
    return date(year, m, d)


def build_timeline(
    dated: DateExtractionResult,
    entries: EntrySegmentationResult,
    policy: TimelinePolicy | None = None,
) -> TimelineResult:
    """Assemble the career timeline and detect employment gaps. Pure function."""
    policy = policy or TimelinePolicy()
    reference = policy.reference_date or dated.reference_date
    by_entry = {e.entry_id: e for e in entries.entries}

    events: list[TimelineEvent] = []
    undated: list[str] = []
    for ed in dated.entry_dates:
        entry = by_entry.get(ed.entry_id)
        if entry is None or ed.primary_range is None:
            # No parseable date (or unknown entry): explicit undated listing.
            undated.append(ed.entry_id)
            continue
        pr = ed.primary_range
        if pr.is_ongoing:
            end = None
        elif pr.end is not None:
            end = _end_cmp(pr.end.year, pr.end.month, pr.end.day)
        else:
            # Broken upstream invariant (end None but not ongoing): explicit.
            undated.append(ed.entry_id)
            continue
        events.append(TimelineEvent(
            entry_id=ed.entry_id, block_id=ed.block_id,
            document_id=dated.document_id, section=ed.section,
            trusted=ed.trusted, source=entry.source,
            start_date=_start_cmp(pr.start.year, pr.start.month, pr.start.day),
            start_granularity=pr.start.granularity,
            end_date=end,
            end_granularity=pr.end.granularity if pr.end is not None else None,
            is_ongoing=pr.is_ongoing,
            effective_end_date=end if end is not None else reference,
            raw_range=pr.raw, text=entry.text,
        ))

    events.sort(key=lambda e: (e.start_date, e.effective_end_date, e.entry_id))

    gaps = _detect_gaps(dated.document_id, events, policy.min_gap_days)

    return TimelineResult(
        document_id=dated.document_id,
        reference_date=reference,
        events=events,
        undated_entry_ids=undated,
        gaps=gaps,
        n_events=len(events),
        n_undated=len(undated),
        n_gaps=len(gaps),
        total_gap_days=sum(g.gap_days for g in gaps),
        longest_gap_days=max((g.gap_days for g in gaps), default=0),
        integrity=build_timeline_integrity(dated, entries, events, undated, gaps, reference, policy),
    )


def _detect_gaps(
    document_id: str,
    events: list[TimelineEvent],
    min_gap_days: int,
) -> list[CareerGap]:
    """Employment gaps over EXPERIENCE events with running max-end coverage."""
    exp = sorted(
        (e for e in events if e.section == SectionLabel.EXPERIENCE),
        key=lambda e: (e.start_date, e.effective_end_date, e.entry_id),
    )
    gaps: list[CareerGap] = []
    covered_until: date | None = None
    before_id = ""
    for ev in exp:
        if covered_until is not None and ev.start_date > covered_until:
            gap_days = (ev.start_date - covered_until).days - 1
            if gap_days >= min_gap_days:
                start = date.fromordinal(covered_until.toordinal() + 1)
                end = date.fromordinal(ev.start_date.toordinal() - 1)
                gaps.append(CareerGap(
                    gap_id=f"G{len(gaps):06d}", document_id=document_id,
                    start_date=start, end_date=end, gap_days=gap_days,
                    gap_months_approx=round(gap_days / 30.44, 1),
                    before_entry_id=before_id, after_entry_id=ev.entry_id,
                ))
        if covered_until is None or ev.effective_end_date > covered_until:
            covered_until = ev.effective_end_date
            before_id = ev.entry_id
    return gaps


def build_timeline_integrity(
    dated: DateExtractionResult,
    entries: EntrySegmentationResult,
    events: list[TimelineEvent],
    undated: list[str],
    gaps: list[CareerGap],
    reference: date,
    policy: TimelinePolicy,
) -> IntegrityReport:
    """B9 integrity: dated/undated partition, chronological order, gap math."""
    violations: list[str] = []
    checks: dict[str, bool] = {}

    dated_ids = [e.entry_id for e in dated.entry_dates]
    event_ids = [e.entry_id for e in events]
    ok_part = sorted(event_ids + undated) == sorted(dated_ids) \
        and len(set(event_ids)) == len(event_ids) \
        and not (set(event_ids) & set(undated))
    checks["dated_undated_partition_complete"] = ok_part
    if not ok_part:
        violations.append(
            f"timeline partition mismatch: {len(dated_ids)} dated entries, "
            f"{len(event_ids)} events + {len(undated)} undated."
        )

    order_keys = [(e.start_date, e.effective_end_date, e.entry_id) for e in events]
    ok_order = order_keys == sorted(order_keys)
    checks["events_chronological"] = ok_order
    if not ok_order:
        violations.append("events not in chronological order.")

    ok_ref = True
    for e in events:
        want_end = e.end_date if e.end_date is not None else reference
        if e.effective_end_date != want_end or (e.is_ongoing == (e.end_date is not None)):
            ok_ref = False
            violations.append(f"{e.entry_id}: effective end / ongoing mismatch.")
    checks["ongoing_resolves_to_reference"] = ok_ref

    ok_gaps, seen_gaps = True, set()
    exp_events = {e.entry_id: e for e in events if e.section == SectionLabel.EXPERIENCE}
    for g in gaps:
        if g.gap_id in seen_gaps:
            ok_gaps = False
            violations.append(f"duplicate gap_id {g.gap_id}.")
        seen_gaps.add(g.gap_id)
        before, after = exp_events.get(g.before_entry_id), exp_events.get(g.after_entry_id)
        if before is None or after is None:
            ok_gaps = False
            violations.append(f"{g.gap_id}: references non-experience or unknown entries.")
            continue
        want_days = (g.end_date - g.start_date).days + 1
        if g.gap_days != want_days or g.gap_days < 1:
            ok_gaps = False
            violations.append(f"{g.gap_id}: gap_days {g.gap_days} != date math {want_days}.")
        if g.gap_days < policy.min_gap_days:
            ok_gaps = False
            violations.append(f"{g.gap_id}: below min_gap_days {policy.min_gap_days}.")
        if not (before.effective_end_date < g.start_date <= g.end_date < after.start_date):
            ok_gaps = False
            violations.append(f"{g.gap_id}: gap not strictly between referenced events.")
        if abs(g.gap_months_approx - round(g.gap_days / 30.44, 1)) > 1e-9:
            ok_gaps = False
            violations.append(f"{g.gap_id}: gap_months_approx inconsistent.")
    checks["gaps_valid"] = ok_gaps

    ok_ids = [g.gap_id for g in gaps] == [f"G{i:06d}" for i in range(len(gaps))]
    checks["gap_ids_unique_contiguous_ordered"] = ok_ids
    if not ok_ids:
        violations.append("gap_ids not unique/contiguous/ordered.")

    if dated.document_id != entries.document_id:
        violations.append(
            f"dates document {dated.document_id} != entries document {entries.document_id}.")
    checks["document_ids_match"] = dated.document_id == entries.document_id

    return IntegrityReport(passed=not violations, violations=violations, checks=checks)


__all__ = [
    "build_timeline",
    "build_timeline_integrity",
]
