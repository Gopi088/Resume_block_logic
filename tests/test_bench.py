"""Tests for the controlled sectioning benchmark (bench_sectioning.py)."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import poc_sectioning as P
from bench_sectioning import (
    MAKERS,
    PER_STRATUM,
    STRATA,
    cmd_eval,
    cmd_gen,
    load_items,
    main,
    rules_predict,
)


def _gen(tmp_path: Path, seed: int = 1234) -> Path:
    out = tmp_path / "bench"
    cmd_gen(Namespace(out=str(out), seed=seed))
    return out


def test_suite_structure_and_manifest(tmp_path):
    out = _gen(tmp_path)
    txts = sorted(out.glob("bench_*.txt"))
    assert len(txts) == len(STRATA) * PER_STRATUM == 24
    man = json.loads((out / "suite.json").read_text())
    assert len(man) == 24
    for s, _ in STRATA:
        assert sum(1 for v in man.values() if v == s) == PER_STRATUM, s
    for t in txts:
        assert t.with_suffix(".labels.csv").exists()


def test_gen_is_deterministic(tmp_path):
    a = _gen(tmp_path / "a")
    b = _gen(tmp_path / "b")
    for f in sorted(a.glob("bench_*")):
        assert f.read_bytes() == (b / f.name).read_bytes(), f.name


def test_gold_covers_every_nonblank_line(tmp_path):
    out = _gen(tmp_path)
    for d in load_items(out):
        assert d["gold"], d["name"]
        for i, line in enumerate(d["lines"], start=1):
            if line.strip():
                assert d["gold"].get(i) in P.SECTIONS, (d["name"], i, line)
        assert sum(d["is_hdr"].values()) >= 1, d["name"]


def test_stress_headers_really_miss_the_lexicon(tmp_path):
    """Premise guard: paraphrase/combined/custom headers must NOT fire
    explicit_heading, otherwise those strata silently stop stress-testing
    rules-only. (Their conventional Experience header legitimately fires.)"""
    for h in ["Career Journey", "Where I've Worked", "Where I Studied",
              "Schooling", "Tech Stack", "What I Know", "Things I've Built",
              "Selected Work", "Career Snapshot",
              "Skills & Certifications", "Awards & Honors",
              "Open Source", "Patents", "Leadership", "Community"]:
        assert P.explicit_heading(h) is None, h
    out = _gen(tmp_path)
    man = json.loads((out / "suite.json").read_text())
    for d in load_items(out):
        s = man[d["name"]]
        hdrs = [l for i, l in enumerate(d["lines"], start=1) if d["is_hdr"].get(i)]
        if s == "unusual_headings":
            assert all(P.explicit_heading(h) is None for h in hdrs), d["name"]
        elif s in ("mixed_sections", "custom_sections"):
            assert any(P.explicit_heading(h) for h in hdrs), d["name"]  # Experience fires
            assert any(P.explicit_heading(h) is None for h in hdrs), d["name"]  # stress ones miss


def test_rules_predict_baseline():
    class B:
        def __init__(self, heading, text):
            self.heading = heading
            self.text = text

    assert rules_predict(B("Work Experience", "Work Experience"), False) == "experience"
    assert rules_predict(B(None, "Jane\nmail"), True) == "contact"
    assert rules_predict(B(None, "Учредитель"), False) == "other"


def test_eval_smoke_oracle_and_none(tmp_path, capsys):
    out = _gen(tmp_path)
    assert main(["eval", "--data", str(out), "--resolve", "none", "--folds", "2"]) == 0
    text = capsys.readouterr().out
    assert "ANSWER:" in text and "Per-stratum" in text


def test_makers_cover_all_strata():
    assert set(MAKERS) == {s for s, _ in STRATA}
    import random
    rnd = random.Random(0)
    for s, fn in MAKERS.items():
        lines, gold, heads = fn(rnd, 0)
        assert len(lines) == len(gold) == len(heads) and len(lines) > 10, s
