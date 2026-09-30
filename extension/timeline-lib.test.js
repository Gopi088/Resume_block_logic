/* Unit tests for timeline-lib.js. Run: node --test extension/timeline-lib.test.js */
"use strict";
const { describe, it } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

require("./timeline-lib.js");
const T = globalThis.CareerTimeline;
const fix = (n) => JSON.parse(fs.readFileSync(path.join(__dirname, "fixtures", n), "utf8"));

describe("thresholds", () => {
  it("classifies confidence bands", () => {
    assert.equal(T.classifyConfidence(0.9), "high");
    assert.equal(T.classifyConfidence(0.8), "high");
    assert.equal(T.classifyConfidence(0.5), "medium");
    assert.equal(T.classifyConfidence(0.49), "low");
    assert.equal(T.classifyConfidence(null), "low");
  });
  it("maps parser sections to entry types", () => {
    assert.equal(T.entryType("experience"), "job");
    assert.equal(T.entryType("education"), "education");
    for (const s of ["skills", "summary", "other", "projects", undefined]) {
      assert.equal(T.entryType(s), "unclassified");
    }
  });
});

describe("honest dates", () => {
  it("shows year-only precision as years", () => {
    const r = T.fmtRange({ start_date: "2008-09-01", end_date: "2009-06-30" },
                         { start: "year", end: "year", inferred: false });
    assert.equal(r.text, "2008 – 2009");
  });
  it("collapses zero-length ranges to a single month", () => {
    const r = T.fmtRange({ start_date: "2015-02-01", end_date: "2015-02-28" },
                         { start: "month", end: "month", inferred: false });
    assert.equal(r.text, "Feb 2015");
  });
  it("marks inferred dates with ~", () => {
    const r = T.fmtRange({ start_date: "2020-01-01", end_date: null, is_ongoing: true },
                         { start: "month", end: "month", inferred: true });
    assert.ok(r.text.startsWith("~") && r.approx);
  });
  it("renders durations in words", () => {
    assert.equal(T.durationWords(0), "less than 1 mo");
    assert.equal(T.durationWords(1), "1 mo");
    assert.equal(T.durationWords(7), "7 mos");
    assert.equal(T.durationWords(12), "1 yr");
    assert.equal(T.durationWords(40), "3 yrs 4 mos");
  });
});

describe("snapshot totals (fixture 1 + fixture 2)", () => {
  it("merges overlapping ranges without double counting", () => {
    const f1 = fix("f1.json").review.events;
    const t = T.snapshotTotals(f1);
    assert.ok(t.computable);
    // Mar 2015–May 2019 (51) + Sep 2020–Dec 2024 (52); open end capped at latest known end
    assert.equal(t.totalMonths, 103);
    assert.equal(t.employerCount, 3);
    const initech = t.employers.find((e) => e.name === "Initech");
    assert.equal(initech.months, 52); // Sep 2020–Dec 2024
  });
  it("uses only medium+ jobs; fragments never count (fixture 2)", () => {
    const f2 = fix("f2.json").review.events;
    const t = T.snapshotTotals(f2);
    assert.ok(t.computable);
    // 4 unclassified rows (conf 0.3–0.4) excluded; only real jobs counted
    assert.ok(t.jobsUsed < f2.length);
    assert.equal(t.jobsUsed, 7); // 4 Northwind + Mystery + Consultant + Advisor; education + fragments excluded
    assert.equal(t.employerCount, 3); // Northwind, Mystery Corp, Freelance
  });
  it("needs-manual-check fixture hides totals (fixture 3)", () => {
    const f3 = fix("f3.json").review;
    const level = T.trustLevel(f3.accuracy.percent / 100,
      f3.events.filter((e) => e.confidence < T.CONF_MEDIUM).length);
    assert.equal(level.level, "manual");
  });
});

describe("employer grouping + progression", () => {
  it("groups Northwind's 4 roles newest-first with promotion tags", () => {
    const f2 = fix("f2.json").review.events.filter((e) => e.section === "experience" && e.confidence >= 0.5);
    const groups = T.groupByEmployer(f2);
    const nw = groups.find((g) => g.name === "Northwind");
    assert.equal(nw.roles.length, 4);
    assert.equal(nw.roles[0].title, "Analytics Manager");
    const tags = [];
    for (let i = nw.roles.length - 1; i > 0; i--) {
      tags.push(T.progressionTag(nw.roles[i].title, nw.roles[i - 1].title));
    }
    assert.deepEqual(tags, ["Promoted", "Promoted", "Promoted"]);
  });
  it("uses Role change when seniority is unclear, none for identical titles", () => {
    assert.equal(T.progressionTag("Analyst", "Banana Wizard"), "Role change");
    assert.equal(T.progressionTag("Analyst", "Analyst"), null);
  });
});

describe("gaps", () => {
  it("finds the real f1 gap inline (15 months)", () => {
    const f1 = fix("f1.json");
    const tl = {};
    f1.timeline.events.forEach((e) => { tl[e.entry_id] = e; });
    const jobs = f1.review.events.map((e) => Object.assign({ _timeline: tl[e.entry_id] }, e))
      .sort((a, b) => (a.start_date < b.start_date ? 1 : -1));
    const gaps = T.computeGaps(jobs);
    assert.equal(gaps.length, 1);
    assert.equal(gaps[0].startYm, "2019-06");
    assert.equal(gaps[0].endYm, "2020-08");
    assert.equal(gaps[0].months, 15);
  });
  it("produces no false gaps in fixture 2", () => {
    const f2 = fix("f2.json");
    const tl = {};
    f2.timeline.events.forEach((e) => { tl[e.entry_id] = e; });
    const jobs = f2.review.events.map((e) => Object.assign({ _timeline: tl[e.entry_id] }, e))
      .sort((a, b) => (a.start_date < b.start_date ? 1 : -1));
    assert.deepEqual(T.computeGaps(jobs), []);
  });
  it("skips gaps touching year-precision neighbours", () => {
    const jobs = [
      { entry_id: "a", section: "experience", confidence: 0.9, start_date: "2008-01-01", end_date: "2010-12-31", raw_range: "2008 – 2010", is_ongoing: false, _timeline: { start_granularity: "year", end_granularity: "year" } },
      { entry_id: "b", section: "experience", confidence: 0.9, start_date: "2012-01-01", end_date: "2013-01-31", raw_range: "Jan 2012", is_ongoing: false, _timeline: { start_granularity: "month", end_granularity: "month" } },
    ];
    assert.deepEqual(T.computeGaps(jobs), []);
  });
});

describe("anchors (fixture 6)", () => {
  it("classifies text, page-only and missing anchors", () => {
    const f6 = fix("f6.json");
    const linesById = {};
    f6.lines.forEach((l) => { linesById[l.line_id] = l; });
    const kinds = f6.review.events.map((e) => T.anchorKind(e.evidence, linesById).kind);
    assert.deepEqual(kinds, ["page", "none", "page"]);
    // E000000's dangling ids: texts unresolvable, pages present -> 'page'
    // fallback; genuinely unresolvable anchors surface at runtime as the
    // inline 'Couldn't locate' message (panel harness covers that path).
    const dangling = T.anchorKind({ line_ids: ["L999990"], pages: [0], excerpt: "x" }, linesById);
    assert.equal(dangling.kind, "page");
  });
});

describe("field sanity gate", () => {
  it("accepts clean titles", () => {
    for (const t of ["Software Developer", "Programmer", "Data Analyst",
                     "Lead Analyst", "Assistant Manager", "Sr. Manager"]) {
      assert.equal(T.isValidTitle(t), true, t);
    }
  });
  it("rejects bullet fragments", () => {
    assert.equal(T.isValidTitle("Served as a key Business Analyst at Australia's largest fund"), false);
    assert.equal(T.isValidTitle("Worked as Software UI developer for a User project"), false);
    assert.equal(T.isValidTitle("Providing 24/7 escalation of Production interruptions"), false);
    assert.equal(T.isValidTitle("Responsible for incident management"), false);
    assert.equal(T.isValidTitle("Managed treasury operations including margin monitoring"), false);
    assert.equal(T.isValidTitle("Production support monitoring of application and addressing the issues"), false); // >8 words
    assert.equal(T.isValidTitle("Assistant Manager (Data Analyst and Business Intelligence) Project Description…"), false);
    assert.equal(T.isValidTitle("Something ended."), false);
    assert.equal(T.isValidTitle("• Led sessions"), false);
    assert.equal(T.isValidTitle(""), false);
    assert.equal(T.isValidTitle(null), false);
  });
  it("rejects city/state/country employers and location matches", () => {
    assert.equal(T.isValidEmployer("Sydney, NSW, Australia", "Sydney, NSW, Australia"), false);
    assert.equal(T.isValidEmployer("Pune, Maharashtra, India", "Pune"), false);
    assert.equal(T.isValidEmployer("Chennai, Tamil Nadu, India", null), false);
    assert.equal(T.isValidEmployer("Tata Consultancy Services", "Pune, Maharashtra, India"), true);
    assert.equal(T.isValidEmployer("Akal Information System Pvt Ltd, New Delhi", null), true);
    assert.equal(T.isValidEmployer("NISG DELHI", null), true);
    assert.equal(T.isValidEmployer("", null), false);
  });
  it("sanitizeRow nulls double-invalid rows for the Other group", () => {
    const bad = T.sanitizeRow({ title: "Served as a key Business Analyst at a fund",
      organization: "Sydney, NSW, Australia", location: "Sydney, NSW, Australia", confidence: 0.85 });
    assert.equal(bad, null);
    const half = T.sanitizeRow({ title: "Production support monitoring of application and addressing the issues",
      organization: "Tata Consultancy Services", location: "Pune", confidence: 0.9 });
    assert.equal(half.employer, "Tata Consultancy Services");
    assert.equal(half.title, null);
    assert.equal(half.uncertain, true);
    assert.deepEqual(half.reasons, ["Title unclear"]);
  });
});
