"""B10 contract tests: the recruiter review projection consumes real parser
output (never invented facts) and the HTTP API preserves the B0–B9 contract."""

from __future__ import annotations

import json

from backend.server import build_review_projection
from parse_resume import build_output

MODEL = "model_real_v2.pkl"
RESUME = "my_resumes/A.Bhargava.docx"


def _full_output() -> dict:
    return build_output(
        RESUME, include_eval=False, include_text=False, blocks_only=False,
        model_path=MODEL,
    )


def test_review_projection_shape():
    out = _full_output()
    review = build_review_projection(out)
    assert set(review) == {"candidate", "accuracy", "events", "gaps", "undated"}

    cand = review["candidate"]
    assert cand["name"] and cand["document_id"] == out["document_id"]
    assert cand["sha256"] == out["sha256"]

    acc = review["accuracy"]
    assert 0 <= acc["percent"] <= 100
    assert acc["label"]
    # Honest low score on this fixture: trust_rate is 0.25, never faked high.
    assert acc["percent"] < 60

    # Events mirror the B9 timeline exactly — no invented entries.
    assert len(review["events"]) == out["timeline"]["n_events"]
    by_id = {e["entry_id"] for e in out["timeline"]["events"]}
    for item in review["events"]:
        assert item["entry_id"] in by_id
        assert item["confidence_band"] in {"High", "Medium", "Low"}
        assert item["evidence"] is None or item["evidence"]["line_ids"]
        # No raw technical internals leak to the UI.
        blob = json.dumps(item)
        for tech in ("ml_accepted", "ml_unresolved", "llm", "spacy", "transformer",
                     "classification", "segmentation"):
            assert tech not in blob.lower()

    # Gaps mirror B9; they carry no fabricated source span.
    assert len(review["gaps"]) == out["timeline"]["n_gaps"] >= 1
    for g in review["gaps"]:
        assert g["evidence"] is None
        assert g["needs_review"] is True

    # Undated entries are a subset of B9's explicit undated list.
    assert set(u["entry_id"] for u in review["undated"]) <= set(
        out["timeline"]["undated_entry_ids"]
    )


def test_evidence_points_at_exact_lines():
    out = _full_output()
    review = build_review_projection(out)
    lines = {ln["line_id"]: ln for ln in out["lines"]}
    for item in review["events"]:
        ev = item["evidence"]
        assert ev and ev["start_line"] in lines
        assert ev["end_line"] in lines
        if ev["raw_range"]:
            assert ev["raw_range"] in ev["excerpt"]


def test_api_endpoints():
    from fastapi.testclient import TestClient

    from backend.server import app

    client = TestClient(app)
    assert client.get("/health").json()["ok"] is True

    with open(RESUME, "rb") as f:
        resp = client.post(
            "/api/parse",
            files={"file": ("A.Bhargava.docx", f, "application/octet-stream")},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["timeline"]["n_events"] >= 1
    assert "review" in body and body["review"]["candidate"]["name"]


def test_shared_notes_roundtrip(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import backend.server as srv

    monkeypatch.setattr(srv, "NOTES_PATH", str(tmp_path / "notes.json"))
    client = TestClient(srv.app)

    assert client.get("/api/notes/abc123").json() == {"ok": True, "notes": {}}

    # Only the note text is mandatory — author is auto-stamped, never typed.
    bad = client.post("/api/notes/abc123", json={"item_id": "E000001", "note": "  "})
    assert bad.json()["ok"] is False

    good = client.post(
        "/api/notes/abc123",
        json={"item_id": "E000001", "note": "Dates match p.1", "by": "Zoya"},
    )
    entries = good.json()["notes"]["E000001"]
    assert len(entries) == 1
    assert entries[0]["note"] == "Dates match p.1"
    assert entries[0]["by"] == "Zoya"
    assert entries[0]["at"]

    again = client.get("/api/notes/abc123").json()["notes"]
    assert len(again["E000001"]) == 1


def test_pipe_header_borrowed_from_same_block():
    from backend.server import _last_pipe, _pipe_header, _resolve_header

    header = {"entry_id": "E000003", "index": 3, "block_id": "B000003",
              "section": "experience",
              "text": "PROFESSIONAL EXPERIENCE\nData Management Analyst | Wells Fargo\nHyderabad, India"}
    dated = {"entry_id": "E000004", "index": 4, "block_id": "B000003",
             "section": "experience",
             "text": "Feb 2025 – Present\n• Support data governance initiatives."}
    tailed = {"entry_id": "E000005", "index": 5, "block_id": "B000003",
              "section": "experience",
              "text": "Sep 2021 – Dec 2024\n• Enhanced ETL processes."}
    tailed_prev = dict(dated)
    tailed_prev["text"] += "\nSoftware Engineer | Cognizant"
    ordered = [header, dated, tailed]

    # Own pre-date header wins.
    assert _pipe_header(header["text"]) == ("Data Management Analyst", "Wells Fargo")
    # Post-date pipes never qualify as the entry's own header.
    assert _pipe_header(tailed_prev["text"]) == (None, None)
    # Dated entries borrow the closest preceding same-block header.
    assert _resolve_header(dated, ordered) == ("Data Management Analyst", "Wells Fargo")
    assert _resolve_header(tailed, [header, tailed_prev, tailed]) == (
        "Software Engineer", "Cognizant")
    # Never across blocks.
    other = dict(tailed)
    other["block_id"] = "B000009"
    assert _resolve_header(other, ordered) == (None, None)
    assert _last_pipe("a\nLead | DevOps\nmulti | pipe | list") == ("Lead", "DevOps")


def test_parse_url_file_scheme():
    import os as _os

    from fastapi.testclient import TestClient

    from backend.server import app

    client = TestClient(app)
    path = _os.path.abspath("my_resumes/_Jayesh Yadav-CV.pdf")
    body = client.post("/api/parse-url", json={"url": "file://" + path}).json()
    assert body["ok"] is True
    assert "JAYESH" in (body["review"]["candidate"]["name"] or "").upper()
    assert body["filename"] == "_Jayesh Yadav-CV.pdf"
    assert body["timeline"]["n_events"] >= 1

    assert client.post("/api/parse-url", json={"url": ""}).json()["ok"] is False
    assert client.post("/api/parse-url", json={"url": "ftp://x/y.pdf"}).json()["ok"] is False
    assert client.post(
        "/api/parse-url", json={"url": "file:///does/not/exist.pdf"}
    ).json()["ok"] is False


def test_resolve_file_url_windows_mapping(tmp_path, monkeypatch):
    import os as _os

    from backend.server import _resolve_file_url

    real = tmp_path / "r.pdf"
    real.write_bytes(b"%PDF")
    monkeypatch.setattr(
        _os.path, "isfile", lambda p: p in (str(real), "/mnt/c/r/x.pdf")
    )
    assert _resolve_file_url("file://" + str(real)) == str(real)
    assert _resolve_file_url("file:///C:/r/x.pdf") == "/mnt/c/r/x.pdf"
    try:
        _resolve_file_url("file:///nope/x.pdf")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unreachable file")
