"""Build preview fixtures f1..f6 for the Career Timeline panel.

Each fixture is a minimal document shaped exactly like the real backend
payload (review + timeline + entry_dates + lines). Synthetic on purpose:
they are TEST data for previewing edge states, never production data.
Run: .venv/bin/python extension/fixtures/make_fixtures.py
Preview: sidepanel.html?fixture=2
"""
import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)))


def _lines(texts, start_page=0):
    lines, raw = [], []
    for i, t in enumerate(texts):
        lid = f"L{i:06d}"
        lines.append({"line_id": lid, "index": i, "document_id": "doc_fix",
                      "raw_text": t, "normalized_text": t,
                      "display_text": t, "page_index": start_page + (i // 6),
                      "line_kind": "text"})
        raw.append(t)
    return lines, "\n".join(raw)


def _ev(ids, pages, raw_range=None, excerpt=""):
    if not ids:
        return None
    return {"line_ids": ids, "start_line": ids[0], "end_line": ids[-1],
            "pages": pages, "raw_range": raw_range, "date_chars": None,
            "excerpt": excerpt, "unavailable": False}


def build(name, rows, accuracy, gaps=None, undated=None, candidate=None,
          note=None, reference_date="2026-09-30"):
    """rows: dicts with keys: section,title,org,loc,start,end,ongoing,raw,
    conf,gran(start,end month|year),page,line(texts list)."""
    lines, events, tl_events, edates = [], [], [], []
    li = 0
    for i, r in enumerate(rows):
        eid = f"E{i:06d}"
        lids = []
        for t in r["lines"]:
            lid = f"L{li:06d}"
            lines.append({"line_id": lid, "index": li, "document_id": "doc_fix",
                          "raw_text": t, "normalized_text": t, "display_text": t,
                          "page_index": r.get("page", 0), "line_kind": "text"})
            lids.append(lid)
            li += 1
        excerpt = "\n".join(r["lines"])[:600]
        ev = _ev(lids, [r.get("page", 0)], r.get("raw"), excerpt)
        band = "High" if r["conf"] >= 0.8 else ("Medium" if r["conf"] >= 0.5 else "Low")
        events.append({"kind": "event", "entry_id": eid, "block_id": f"B{i:06d}",
                       "section": r["section"], "title": r.get("title"),
                       "organization": r.get("org"), "location": r.get("loc"),
                       "start_date": r.get("start"), "end_date": r.get("end"),
                       "is_ongoing": bool(r.get("ongoing")), "raw_range": r.get("raw"),
                       "confidence": r["conf"], "confidence_band": band,
                       "trusted": r["conf"] >= 0.5, "needs_review": r["conf"] < 0.5,
                       "evidence": ev})
        tl_events.append({"entry_id": eid, "block_id": f"B{i:06d}",
                          "document_id": "doc_fix", "section": r["section"],
                          "trusted": r["conf"] >= 0.5,
                          "start_date": r.get("start"), "start_granularity": r.get("gran", ("month", "month"))[0],
                          "end_date": r.get("end"), "end_granularity": r.get("gran", ("month", "month"))[1],
                          "is_ongoing": bool(r.get("ongoing")),
                          "effective_end_date": r.get("end") or reference_date,
                          "raw_range": r.get("raw"), "text": excerpt})
        edates.append({"entry_id": eid, "block_id": f"B{i:06d}", "document_id": "doc_fix",
                       "section": r["section"], "trusted": r["conf"] >= 0.5,
                       "ranges": [], "single_dates": [],
                       "primary_range": {"raw": r.get("raw") or ""} if r.get("raw") else None,
                       "has_dates": bool(r.get("start"))})
    doc = {"ok": True, "document_id": "doc_fix", "filename": name + ".pdf",
           "sha256": "fix" + name, "timeline": {"document_id": "doc_fix",
           "reference_date": reference_date, "events": tl_events,
           "undated_entry_ids": [], "gaps": gaps or [], "n_events": len(tl_events),
           "n_undated": 0, "n_gaps": len(gaps or [])},
           "entries": {"entries": []}, "entry_dates": {"entry_dates": edates},
           "final_sections": {}, "lines": lines,
           "review": {"candidate": candidate or {"name": "Test Candidate", "sha256": "fix" + name},
                      "accuracy": accuracy, "events": events,
                      "gaps": gaps or [], "undated": undated or []},
           "counts": {}}
    if note:
        doc["_note"] = note  # harnesses preload this into storage; panel reads store, not payload
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(doc, f, ensure_ascii=False)
    print("wrote", name, len(events), "events")


J = lambda section, title, org, start, end=None, **kw: dict(
    {"section": section, "title": title, "org": org, "start": start, "end": end,
     "conf": 0.9, "gran": ("month", "month")}, **kw)

# F1 — clean resume: 5 high-confidence jobs, one real gap, valid anchors.
build("f1", [
    J("experience", "Junior Analyst", "Northwind", "2015-03-01", "2017-02-28",
      raw="Mar 2015 – Feb 2017", page=0,
      lines=["Junior Analyst | Northwind", "Mar 2015 – Feb 2017"]),
    J("experience", "Analyst", "Northwind", "2017-03-01", "2019-05-31",
      raw="Mar 2017 – May 2019", page=0,
      lines=["Analyst | Northwind", "Mar 2017 – May 2019"]),
    J("experience", "Senior Analyst", "Initech", "2020-09-01", "2022-12-31",
      raw="Sep 2020 – Dec 2022", page=1,
      lines=["Senior Analyst | Initech", "Sep 2020 – Dec 2022"]),
    J("experience", "Lead Analyst", "Initech", "2023-01-01", "2024-12-31",
      raw="Jan 2023 – Dec 2024", page=1,
      lines=["Lead Analyst | Initech", "Jan 2023 – Dec 2024"]),
    J("experience", "Analytics Manager", "Globex", "2025-01-01", None,
      ongoing=True, raw="Jan 2025 – Present", page=1,
      lines=["Analytics Manager | Globex", "Jan 2025 – Present"]),
], {"percent": 92, "label": "Accurate", "trust_rate": 0.9},
    gaps=[{"kind": "gap", "gap_id": "G000000", "start_date": "2019-06-01",
           "end_date": "2020-08-31", "gap_months_approx": 15.0,
           "before_entry_id": "E000001", "after_entry_id": "E000002",
           "needs_review": True}])

# F2 — problem resume: fragments, education, promotion path, missing org,
# year-only rows, zero-length range.
build("f2", [
    J("experience", "Analyst", "Northwind", "2009-03-01", "2011-02-28",
      raw="Mar 2009 – Feb 2011", conf=0.9, page=0,
      lines=["Analyst | Northwind", "Mar 2009 – Feb 2011"]),
    J("experience", "Senior Analyst", "Northwind", "2011-03-01", "2013-02-28",
      raw="Mar 2011 – Feb 2013", conf=0.9, page=0,
      lines=["Senior Analyst | Northwind", "Mar 2011 – Feb 2013"]),
    J("experience", "Lead Analyst", "Northwind", "2013-03-01", "2015-02-28",
      raw="Mar 2013 – Feb 2015", conf=0.9, page=0,
      lines=["Lead Analyst | Northwind", "Mar 2013 – Feb 2015"]),
    J("experience", "Analytics Manager", "Northwind", "2015-03-01", "2018-08-31",
      raw="Mar 2015 – Aug 2018", conf=0.9, page=1,
      lines=["Analytics Manager | Northwind", "Mar 2015 – Aug 2018"]),
    J("experience", None, "Mystery Corp", "2018-09-01", "2020-12-31",
      raw="Sep 2018 – Dec 2020", conf=0.6, page=1,
      lines=["Mystery Corp", "Sep 2018 – Dec 2020"]),
    J("education", "BSc Computer Science", "State University", "2004-09-01", "2008-06-30",
      raw="2004 – 2008", conf=0.9, gran=("year", "year"), page=0,
      lines=["BSc Computer Science, State University, 2008"]),
    J("education", "MSc Data Science", "City University", "2008-09-01", "2009-06-30",
      raw="2008", conf=0.55, gran=("year", "year"), page=0,
      lines=["MSc Data Science 2008"]),
    J("skills", "led cross-functional workshops driving adoption", None, "2021-01-01", "2021-12-31",
      raw="2021", conf=0.4, gran=("year", "year"), page=1,
      lines=["led cross-functional workshops driving adoption 2021"]),
    J("other", "improved reporting pipelines and dashboards", None, "2022-01-01", "2022-06-30",
      raw="Jan 2022 – Jun 2022", conf=0.3, page=1,
      lines=["improved reporting pipelines and dashboards"]),
    J("skills", "managed stakeholder expectations across regions", None, "2022-07-01", "2022-07-31",
      raw="Jul 2022", conf=0.35, page=1,
      lines=["managed stakeholder expectations Jul 2022"]),
    J("other", "PowerBI, SQL, Python", None, "2023-01-01", "2023-01-31",
      raw="Jan 2023", conf=0.3, page=1,
      lines=["PowerBI, SQL, Python Jan 2023"]),
    J("experience", "Consultant", "Freelance", "2015-02-01", "2015-02-28",
      raw="Feb 2015", conf=0.9, page=0,
      lines=["Consultant | Freelance", "Feb 2015"]),
    J("experience", "Advisor", "Freelance", "2008-01-01", "2008-12-31",
      raw="2008", conf=0.9, gran=("year", "year"), page=0,
      lines=["Advisor | Freelance 2008"]),
], {"percent": 55, "label": "Check carefully", "trust_rate": 0.5})

# F3 — very low confidence.
build("f3", [
    J("experience", "Something Analyst", None, "2019-01-01", "2020-01-31",
      raw="Jan 2019 – Jan 2020", conf=0.2, page=0,
      lines=["Something Analyst", "Jan 2019 – Jan 2020"]),
    J("experience", None, None, "2021-01-01", "2021-06-30",
      raw="2021", conf=0.1, gran=("year", "year"), page=0,
      lines=["unclear fragment 2021"]),
], {"percent": 7, "label": "Low — verify", "trust_rate": 0.05})

# F4 — empty / unreadable.
build("f4", [], {"percent": 0, "label": "Low — verify", "trust_rate": 0.0},
      candidate={"name": "Unreadable resume", "sha256": "fixf4"})

# F5 — human-checked (panel reads the note from storage; harness preloads it).
build("f5", [
    J("experience", "Analyst", "Northwind", "2018-01-01", "2021-12-31",
      raw="Jan 2018 – Dec 2021", conf=0.9, page=0,
      lines=["Analyst | Northwind", "Jan 2018 – Dec 2021"]),
    J("experience", "Senior Analyst", "Globex", "2022-01-01", None,
      ongoing=True, raw="Jan 2022 – Present", conf=0.9, page=0,
      lines=["Senior Analyst | Globex", "Jan 2022 – Present"]),
], {"percent": 88, "label": "Accurate", "trust_rate": 0.85},
    note={"note": "Career history verified against the resume.", "by": "Amelia R.",
          "at": "2026-09-30T10:00:00"})

# F6 — anchor edge cases.
build("f6", [
    J("experience", "Analyst", "Northwind", "2018-01-01", "2021-12-31",
      raw="Jan 2018 – Dec 2021", conf=0.9, page=0,
      lines=["Analyst | Northwind", "Jan 2018 – Dec 2021"]),
    J("experience", "Manager", "Globex", "2022-01-01", None,
      ongoing=True, raw="Jan 2022 – Present", conf=0.9, page=1,
      lines=["Manager | Globex", "Jan 2022 – Present"]),
    J("education", "BSc", "State Uni", "2014-01-01", "2017-12-31",
      raw="2014 – 2017", conf=0.9, gran=("year", "year"), page=2,
      lines=["BSc State Uni 2014 – 2017"]),
], {"percent": 90, "label": "Accurate", "trust_rate": 0.9})

# Mutate f6 anchors: E000000 -> dangling line ids (resolution fails),
# E000001 -> no line ids at all (control hidden),
# E000002 -> page-only anchor.
_f6 = json.load(open(os.path.join(OUT, "f6.json")))
_evs = _f6["review"]["events"]
_evs[0]["evidence"]["line_ids"] = ["L999990", "L999991"]
_evs[0]["evidence"]["start_line"] = "L999990"
_evs[0]["evidence"]["end_line"] = "L999991"
_evs[1]["evidence"] = None
_evs[2]["evidence"]["line_ids"] = []
_evs[2]["evidence"]["start_line"] = None
_evs[2]["evidence"]["end_line"] = None
_evs[2]["evidence"]["pages"] = [2]
json.dump(_f6, open(os.path.join(OUT, "f6.json"), "w"), ensure_ascii=False)
print("f6 anchors mutated")
# F7 — Sydney/Pune/Chennai mirror: bullet-fragment titles, city-as-org rows.
build("f7", [
    J("experience", "IT Support Analyst Level 2", "Tata Consultancy Services", "2018-05-01", "2019-12-31",
      raw="May 2018 – Dec 2019", conf=0.9, page=0, loc="Pune, Maharashtra, India",
      lines=["IT Support Analyst Level 2 | Tata Consultancy Services", "Pune, Maharashtra, India", "May 2018 – Dec 2019"]),
    J("experience", "Production support monitoring of application and addressing the issues reported for transactions",
      "Pune, Maharashtra, India", "2020-01-01", "2023-01-31",
      raw="Jan 2020 – Jan 2023", conf=0.85, page=0, loc="Pune, Maharashtra, India",
      lines=["Production support monitoring of application", "Pune, Maharashtra, India", "Jan 2020 – Jan 2023"]),
    J("experience", "Served as a key Business Analyst at Australia's largest wealth management fund",
      "Sydney, NSW, Australia", "2024-02-01", "2025-08-31",
      raw="Feb 2024 – Aug 2025", conf=0.8, page=1, loc="Sydney, NSW, Australia",
      lines=["Served as a key Business Analyst", "Sydney, NSW, Australia", "Feb 2024 – Aug 2025"]),
    J("experience", "Providing 24/7 escalation of Production service interruptions and changes",
      "Chennai, Tamil Nadu, India", "2020-06-01", "2022-12-31",
      raw="Jun 2020 – Dec 2022", conf=0.75, page=1, loc="Chennai, Tamil Nadu, India",
      lines=["Providing 24/7 escalation", "Chennai, Tamil Nadu, India", "Jun 2020 – Dec 2022"]),
    J("experience", "Software UI Developer", "Brightline Systems", "2016-11-01", "2017-04-30",
      raw="Nov 2016 – Apr 2017", conf=0.9, page=0, loc="Chennai, Tamil Nadu, India",
      lines=["Software UI Developer | Brightline Systems", "Nov 2016 – Apr 2017"]),
    J("experience", "Senior Software UI Developer", "Brightline Systems", "2017-05-01", "2018-04-30",
      raw="May 2017 – Apr 2018", conf=0.9, page=0, loc="Chennai, Tamil Nadu, India",
      lines=["Senior Software UI Developer | Brightline Systems", "May 2017 – Apr 2018"]),
    J("education", "BSc Computer Science", "State University", "2012-09-01", "2016-06-30",
      raw="2012 – 2016", conf=0.9, gran=("year", "year"), page=0,
      lines=["BSc Computer Science, State University, 2016"]),
], {"percent": 82, "label": "Accurate", "trust_rate": 0.8},
    candidate={"name": "Ankur Sharma", "sha256": "fixf7"})
print("done")
