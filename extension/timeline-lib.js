/* Career Timeline — pure presentation logic (no DOM, no chrome APIs).
   Tested with: node --test extension/timeline-lib.test.js
   All inputs are real parser fields; nothing is invented. */
(function (global) {
  "use strict";

  /* ---------------- named thresholds ---------------- */
  var CONF_HIGH = 0.8;      // HIGH >= 0.8
  var CONF_MEDIUM = 0.5;    // MEDIUM 0.5–0.8, LOW < 0.5
  var GAP_MIN_MONTHS = 3;   // gaps shorter than this are not shown

  var MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  function classifyConfidence(c) {
    c = Number(c);
    if (!(c >= 0)) return "low";
    if (c >= CONF_HIGH) return "high";
    if (c >= CONF_MEDIUM) return "medium";
    return "low";
  }

  /* section (parser) -> entry type. The parser has no explicit
     job|education|unclassified type, so section is the safest
     existing field: experience=>job, education=>education,
     everything else=>unclassified. */
  function entryType(section) {
    if (section === "experience") return "job";
    if (section === "education") return "education";
    return "unclassified";
  }

  /* Precision for one timeline event. Prefers the parser's own
     granularity fields (timeline.events supplement, keyed by entry_id);
     falls back to the raw_range text shape. 'inferred' never occurs in
     parser output — the code path exists as a named placeholder. */
  function precisionOf(ev, rawRange) {
    function norm(g) {
      if (g === "month" || g === "year" || g === "day") return g;
      return null;
    }
    var start = norm(ev && ev.start_granularity);
    var end = norm(ev && ev.end_granularity);
    if (start || end) {
      return {
        start: start || "month",
        end: end || (ev && ev.is_ongoing ? "month" : "month"),
        inferred: false
      };
    }
    var raw = String(rawRange || "").trim();
    if (/^\d{4}\s*[–—\-/]\s*\d{4}$/.test(raw) || /^\d{4}$/.test(raw)) {
      return { start: "year", end: "year", inferred: false };
    }
    if (/present|current|~|approx|circa/i.test(raw)) {
      return { start: "month", end: "month", inferred: true };
    }
    return { start: "month", end: "month", inferred: false };
  }

  function ym(dateIso) {
    var p = String(dateIso || "").split("-");
    return { y: parseInt(p[0], 10), m: parseInt(p[1] || "1", 10) };
  }

  /* Honest date rendering. Year precision never shows months.
     Zero-length ranges collapse to a single month. */
  function fmtRange(ev, precision) {
    precision = precision || { start: "month", end: "month", inferred: false };
    function fmt(iso, prec) {
      if (!iso) return null;
      var d = ym(iso);
      if (prec === "year" || !d.m) return String(d.y);
      return MONTHS[d.m - 1] + " " + d.y;
    }
    var approx = precision.inferred ? "~" : "";
    if (!ev.start_date && !ev.end_date) return { text: "Dates unclear", approx: true };
    if (ev.start_date && ev.end_date && ev.start_date.slice(0, 7) === ev.end_date.slice(0, 7)) {
      return { text: approx + fmt(ev.start_date, precision.start), approx: precision.inferred };
    }
    if (!ev.start_date) return { text: approx + "– " + fmt(ev.end_date, precision.end), approx: precision.inferred };
    if (!ev.end_date || ev.is_ongoing) return { text: approx + fmt(ev.start_date, precision.start) + " – Present", approx: precision.inferred };
    return { text: approx + fmt(ev.start_date, precision.start) + " – " + fmt(ev.end_date, precision.end), approx: precision.inferred };
  }

  function monthsBetween(startIso, endIso) {
    var a = ym(startIso), b = ym(endIso);
    return (b.y - a.y) * 12 + (b.m - a.m) + 1; // inclusive calendar months
  }

  function durationWords(totalMonths) {
    totalMonths = Math.max(0, Math.round(totalMonths));
    if (totalMonths < 1) return "less than 1 mo";
    var y = Math.floor(totalMonths / 12), m = totalMonths % 12;
    if (y === 0) return m + (m === 1 ? " mo" : " mos");
    if (m === 0) return y + (y === 1 ? " yr" : " yrs");
    return y + (y === 1 ? " yr " : " yrs ") + m + (m === 1 ? " mo" : " mos");
  }

  /* Merge overlapping/contiguous month ranges. Returns union ranges. */
  function mergeRanges(ranges) {
    var rs = ranges.filter(function (r) { return r && r.start; })
      .map(function (r) { return { start: r.start.slice(0, 7), end: (r.end || r.start).slice(0, 7) }; })
      .sort(function (a, b) { return a.start < b.start ? -1 : 1; });
    var out = [];
    rs.forEach(function (r) {
      var last = out[out.length - 1];
      if (last && r.start <= last.end) {
        if (r.end > last.end) last.end = r.end;
      } else if (last && addMonths(last.end, 1) === r.start) {
        last.end = r.end; // contiguous counts as continuous employment
      } else {
        out.push({ start: r.start, end: r.end });
      }
    });
    return out;
  }

  function addMonths(yyyymm, n) {
    var y = parseInt(yyyymm.slice(0, 4), 10), m = parseInt(yyyymm.slice(5, 7), 10) + n;
    while (m > 12) { m -= 12; y++; }
    while (m < 1) { m += 12; y--; }
    return y + "-" + (m < 10 ? "0" + m : m);
  }

  function unionMonths(ranges) {
    var total = 0;
    mergeRanges(ranges).forEach(function (r) {
      total += monthsBetween(r.start + "-01", r.end + "-01");
    });
    return total;
  }

  /* Snapshot totals: ONLY type=job AND confidence>=MEDIUM. Overlapping
     ranges merged so concurrent roles are never double counted. */
  function snapshotTotals(jobs) {
    var qualifying = jobs.filter(function (j) {
      return entryType(j.section) === "job" && Number(j.confidence) >= CONF_MEDIUM && j.start_date;
    });
    if (!qualifying.length) return { computable: false };
    var byEmployer = {};
    qualifying.forEach(function (j) {
      var key = (j.organization || "").trim().toLowerCase();
      if (!key) return;
      (byEmployer[key] = byEmployer[key] || { name: (j.organization || "").trim(), ranges: [] })
        .ranges.push({ start: j.start_date, end: j.is_ongoing ? null : (j.end_date || j.start_date) });
    });
    var employers = Object.keys(byEmployer).map(function (k) {
      var e = byEmployer[k];
      return { name: e.name, months: unionMonths(e.ranges.map(function (r) {
        return { start: r.start, end: r.end || null };
      })) };
    }).sort(function (a, b) { return b.months - a.months; });
    // Union across all qualifying ranges; open-ended ends use latest known end.
    var latestEnd = null;
    qualifying.forEach(function (j) {
      var e = j.is_ongoing ? null : (j.end_date || j.start_date);
      if (e && (!latestEnd || e > latestEnd)) latestEnd = e;
    });
    var total = unionMonths(qualifying.map(function (j) {
      return { start: j.start_date, end: j.is_ongoing ? (latestEnd || j.start_date) : (j.end_date || j.start_date) };
    }));
    return {
      computable: true,
      totalMonths: total,
      employers: employers,
      employerCount: employers.length,
      jobsUsed: qualifying.length
    };
  }

  /* Group jobs by employer (newest employer first by latest start),
     roles newest-first inside. */
  function groupByEmployer(jobs) {
    var groups = {}, order = [];
    var sorted = jobs.slice().sort(function (a, b) {
      return (a.start_date || "") < (b.start_date || "") ? 1 : -1;
    });
    sorted.forEach(function (j) {
      var key = ((j.organization || "").trim().toLowerCase()) || "__unknown__";
      if (!groups[key]) {
        groups[key] = { name: (j.organization || "").trim() || null, roles: [] };
        order.push(key);
      }
      groups[key].roles.push(j);
    });
    return order.map(function (k) { return groups[k]; });
  }

  /* Promotion/role-change tag between consecutive roles at one employer
     (older -> newer). Conservative: seniority lexicon with explicit
     junior->senior ladders; anything else with a changed title is a
     "Role change"; same title => no tag. */
  var SENIORITY = ["intern", "trainee", "junior", "associate", "analyst", "specialist",
                   "engineer", "developer", "consultant", "senior", "lead", "principal",
                   "manager", "director", "vp", "president"];
  function seniorityRank(title) {
    var t = String(title || "").toLowerCase();
    var rank = -1;
    SENIORITY.forEach(function (word, i) {
      if (new RegExp("\\b" + word + "s?\\b").test(t) && i > rank) rank = i;
    });
    return rank;
  }
  function progressionTag(olderTitle, newerTitle) {
    var a = String(olderTitle || "").trim().toLowerCase();
    var b = String(newerTitle || "").trim().toLowerCase();
    if (!a || !b || a === b) return null;
    var ra = seniorityRank(a), rb = seniorityRank(b);
    if (ra >= 0 && rb >= 0 && rb > ra) return "Promoted";
    return "Role change";
  }

  /* Gaps: only between consecutive HIGH/MEDIUM jobs, both month-precision,
     gap >= GAP_MIN_MONTHS. Rendered inline, neutral wording. */
  function computeGaps(jobsNewestFirst) {
    var trusted = jobsNewestFirst.filter(function (j) {
      return entryType(j.section) === "job" && Number(j.confidence) >= CONF_MEDIUM;
    });
    // chronological, oldest first
    var asc = trusted.slice().sort(function (a, b) {
      return (a.start_date || "") < (b.start_date || "") ? -1 : 1;
    });
    var gaps = [];
    for (var i = 0; i + 1 < asc.length; i++) {
      var prev = asc[i], next = asc[i + 1];
      if (prev.is_ongoing) continue;
      var pEnd = prev.end_date, nStart = next.start_date;
      if (!pEnd || !nStart) continue;
      var pPrec = precisionOf(prev._timeline, prev.raw_range);
      var nPrec = precisionOf(next._timeline, next.raw_range);
      if (pPrec.end !== "month" || nPrec.start !== "month") continue; // uncertain neighbour: no gap
      var gapStart = addMonths(pEnd.slice(0, 7), 1);
      var gapEnd = addMonths(nStart.slice(0, 7), -1);
      var months = monthsBetween(gapStart + "-01", gapEnd + "-01");
      if (months >= GAP_MIN_MONTHS) {
        gaps.push({
          afterEntryId: prev.entry_id, beforeEntryId: next.entry_id,
          startYm: gapStart, endYm: gapEnd, months: months
        });
      }
    }
    return gaps;
  }

  /* Trust chip from overall confidence + low-entry count. */
  function trustLevel(overallConfidence, lowCount) {
    if (!(overallConfidence >= 0)) return { level: "manual", label: "Needs manual check" };
    if (overallConfidence < 0.5) return { level: "manual", label: "Needs manual check" };
    if (overallConfidence < CONF_HIGH || lowCount > 0) return { level: "check", label: "Check details" };
    return { level: "reliable", label: "Reliable" };
  }

  /* Anchor resolution strategy report helper: classifies an evidence object
     into text-anchor | page-only | none, using the real parser shape
     {line_ids, start_line, end_line, pages[], excerpt}. */
  function anchorKind(evidence, linesById) {
    if (!evidence) return { kind: "none" };
    var ids = evidence.line_ids || [];
    var texts = ids.map(function (id) {
      var ln = (linesById || {})[id];
      return ln && ln.display_text ? String(ln.display_text).trim() : "";
    }).filter(Boolean);
    if (texts.length) return { kind: "text", pages: evidence.pages || [], lineIds: ids };
    if (evidence.pages && evidence.pages.length) return { kind: "page", pages: evidence.pages };
    return { kind: "none" };
  }

  global.CareerTimeline = {
    CONF_HIGH: CONF_HIGH, CONF_MEDIUM: CONF_MEDIUM, GAP_MIN_MONTHS: GAP_MIN_MONTHS,
    classifyConfidence: classifyConfidence,
    entryType: entryType,
    precisionOf: precisionOf,
    fmtRange: fmtRange,
    monthsBetween: monthsBetween,
    durationWords: durationWords,
    mergeRanges: mergeRanges,
    unionMonths: unionMonths,
    snapshotTotals: snapshotTotals,
    groupByEmployer: groupByEmployer,
    progressionTag: progressionTag,
    computeGaps: computeGaps,
    trustLevel: trustLevel,
    anchorKind: anchorKind
  };
})(typeof window !== "undefined" ? window : globalThis);
