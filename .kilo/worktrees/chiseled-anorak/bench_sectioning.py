#!/usr/bin/env python3
"""
Controlled benchmark: does ML + deterministic + LLM beat simpler alternatives?

    python bench_sectioning.py gen  --out bench_data [--seed 1234]
    python bench_sectioning.py eval --data bench_data [--resolve oracle|llm|none] [--folds 5]

Design
------
* 24 resumes, 8 strata x 3 variants (seeded, reproducible). Strata stress the
  failure modes rules-only systems are known for: unusual headings, missing
  headings, two-column extraction artifacts, implicit experience, mixed
  sections, multiple jobs, custom sections.
* Inputs are .txt run through the REAL pipeline entry points
  (extract_markdown -> build_document -> segment_blocks from poc_sectioning).
  Scope note: this benchmarks segmentation -> decision, NOT PDF extraction
  fidelity. Two-column layouts are simulated as extraction artifacts
  ("left  |  right" row-wise joins), which is what MarkItDown/pdfminer emit.
* Ground truth is per-LINE (line_id, gold_section, is_header): every nonblank
  line carries its span's section, so predicted blocks never need to align 1:1
  with gold spans and missing-heading cases stay scorable. Primary metric:
  line accuracy on nonblank lines. Secondary: header-line accuracy, routing.
* Four systems, same ground truth, ML trained with GroupKFold (no leakage):
    rules : explicit heading -> first block contact -> experience shape -> other
    ml    : block classifier argmax (same Tfidf+LogReg recipe as the PoC)
    mldet : ml + deterministic validation (routed blocks fall back to ml)
    full  : mldet + routed blocks resolved by --resolve (oracle | llm | none)
* --resolve oracle is the free upper bound of the LLM stage; --resolve llm
  measures the real LLM (needs ANTHROPIC_API_KEY + anthropic package).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

import poc_sectioning as P

# --------------------------------------------------------------------------
# Suite definition
# --------------------------------------------------------------------------

BENCH_SEED = 1234
PER_STRATUM = 3
STRATA = [
    ("conventional", "standard headings, plain styles; the easy case"),
    ("unusual_headings", "paraphrased headers absent from the alias lexicon"),
    ("missing_headers", "1-2 header lines deleted post-hoc (implicit spans)"),
    ("two_column", "row-wise 'left | right' extraction artifacts on body lines"),
    ("implicit_experience", "jobs with dates/roles/bullets, never had a header"),
    ("mixed_sections", "combined + near-miss headers ('Skills & Certifications')"),
    ("multiple_jobs", "4 jobs with varied date formats"),
    ("custom_sections", "headers outside the lexicon (Open Source, Patents, ...)"),
]

PLAIN_HEADERS = {
    "summary": ["Summary", "Professional Summary"],
    "experience": ["Experience", "Work Experience", "Professional Experience"],
    "education": ["Education", "Academic Background"],
    "skills": ["Skills", "Technical Skills"],
    "projects": ["Projects", "Personal Projects"],
    "certifications": ["Certifications"],
    "awards": ["Awards"],
    "languages": ["Languages"],
}
# Paraphrases verified absent from poc SECTION_ALIASES (else rules would fire).
PARAPHRASE = {
    "summary": ["Career Snapshot", "Snapshot"],
    "experience": ["Career Journey", "Where I've Worked"],
    "education": ["Where I Studied", "Schooling"],
    "skills": ["Tech Stack", "What I Know"],
    "projects": ["Things I've Built", "Selected Work"],
}
CUSTOM_SECTIONS = [  # (header, gold_section, body lines)
    ("Open Source", "projects",
     ["maintainer.nvim: Neovim plugin, 2k stars", "- Merged PRs in pandas and requests"]),
    ("Patents", "publications",
     ["US1234567: fast resume parser, 2023"]),
    ("Leadership", "other",
     ["Led a team of 5 engineers", "Mentored 3 interns across 2 teams"]),
    ("Community", "volunteering",
     ["Weekend coding mentor, 2021 - Present"]),
]
DATE_FORMATS = ["Jan 2019 - Present", "2016 - 2018", "March 2020 to Dec 2021",
                "04/2022 - Present", "2015-2017", "Jun 2021 - Present"]
ROLES = ["Software Engineer", "Data Analyst", "ML Engineer", "Backend Developer", "QA Lead"]
COMPANIES = ["Infosys Ltd", "TCS", "Acme Corp", "Startup Labs", "Globex Inc"]


# --------------------------------------------------------------------------
# Resume builders: each returns (lines, gold, is_header) with gold=None on blanks
# --------------------------------------------------------------------------

def _contact(rnd):
    name = rnd.choice(["Jane Doe", "Aarav Sharma", "Maria Santos", "John Mathew"])
    return ([name, "jane.doe@mail.com | +91 98765 43210 | linkedin.com/in/janedoe", ""],
            ["contact", "contact", None], [False, False, False])


def _styled(header, rnd):
    s = rnd.choice(["plain", "md", "upper", "colon"])
    h = header.upper() if s == "upper" else header
    if s == "md":
        return [f"## {h}"]
    if s == "colon":
        return [f"{h}:"]
    return [h]


def _section(lines, gold, heads, header, gtype, body, rnd):
    """Header lines: gold=gtype + is_header. Body lines: gold=gtype (span label).

    Every nonblank line carries its span's section, so line accuracy measures
    true section assignment even when headings are missing or blocks split.
    """
    hl = _styled(header, rnd)
    for k, x in enumerate(hl):
        lines.append(x); gold.append(gtype); heads.append(k == 0)
    for x in body:
        lines.append(x); gold.append(gtype); heads.append(False)
    lines.append(""); gold.append(None); heads.append(False)


def _jobs(rnd, n, dates=None):
    out = []
    for _ in range(n):
        d = (dates or DATE_FORMATS)[rnd.randrange(len(dates or DATE_FORMATS))]
        out += [f"{rnd.choice(ROLES)} | {rnd.choice(COMPANIES)}", d,
                f"- Built pipelines using Python, cut cost by {rnd.randint(10, 60)}%",
                f"- Led standups and shipped {rnd.randint(2, 9)} releases"]
    return out


def _edu():
    return ["B.Tech Computer Science, VTU", "2014 - 2018 | CGPA 8.4"]


def _skills():
    return ["Python, SQL, Docker, Git, Airflow", "Communication, Leadership"]


def _projects():
    return ["Project Atlas: resume parser (Python)", "- Cut manual screening time by 30%"]


def _summary():
    return ["Results-driven engineer with 5 years of experience building data products."]


def _base_skeleton(rnd, headers: dict[str, list[str]], jobs_n=2):
    """Contact + summary + experience + education + skills with given header pool."""
    lines, gold, heads = _contact(rnd)
    _section(lines, gold, heads, rnd.choice(headers["summary"]), "summary", _summary(), rnd)
    _section(lines, gold, heads, rnd.choice(headers["experience"]), "experience",
             _jobs(rnd, jobs_n), rnd)
    _section(lines, gold, heads, rnd.choice(headers["education"]), "education", _edu(), rnd)
    _section(lines, gold, heads, rnd.choice(headers["skills"]), "skills", _skills(), rnd)
    return lines, gold, heads


def _mk_conventional(rnd, _v):
    return _base_skeleton(rnd, PLAIN_HEADERS)


def _mk_unusual(rnd, _v):
    pool = {k: PARAPHRASE[k] for k in PARAPHRASE}
    pool["certifications"] = ["Certifications"]
    lines, gold, heads = _contact(rnd)
    _section(lines, gold, heads, rnd.choice(pool["summary"]), "summary", _summary(), rnd)
    _section(lines, gold, heads, rnd.choice(pool["experience"]), "experience",
             _jobs(rnd, 2), rnd)
    _section(lines, gold, heads, rnd.choice(pool["education"]), "education", _edu(), rnd)
    _section(lines, gold, heads, rnd.choice(pool["skills"]), "skills", _skills(), rnd)
    _section(lines, gold, heads, rnd.choice(pool["projects"]), "projects", _projects(), rnd)
    return lines, gold, heads


def _drop_headers(lines, gold, heads, rnd, types, k):
    idx = [i for i, g in enumerate(gold) if g in types and heads[i]]
    drop = set(rnd.sample(idx, min(k, len(idx))))
    return ([l for i, l in enumerate(lines) if i not in drop],
            [g for i, g in enumerate(gold) if i not in drop],
            [h for i, h in enumerate(heads) if i not in drop])


def _mk_missing(rnd, _v):
    lines, gold, heads = _base_skeleton(rnd, PLAIN_HEADERS)
    return _drop_headers(lines, gold, heads, rnd, ["experience", "skills", "education"], k=2)


def _mk_twocol(rnd, _v):
    lines, gold, heads = _base_skeleton(rnd, PLAIN_HEADERS)
    out_l, out_g, out_h = [], [], []
    i = 0
    while i < len(lines):
        # Merge only same-span, non-header body pairs (never across sections).
        if (i + 1 < len(lines) and lines[i].strip() and lines[i + 1].strip()
                and not heads[i] and not heads[i + 1]
                and gold[i] is not None and gold[i] == gold[i + 1]
                and rnd.random() < 0.5):
            out_l.append(lines[i] + "  |  " + lines[i + 1])
            out_g.append(gold[i]); out_h.append(False); i += 2
        else:
            out_l.append(lines[i]); out_g.append(gold[i]); out_h.append(heads[i]); i += 1
    return out_l, out_g, out_h


def _mk_implicit(rnd, _v):
    lines, gold, heads = _contact(rnd)
    _section(lines, gold, heads, "Summary", "summary", _summary(), rnd)
    for x in _jobs(rnd, 2):  # jobs appended with NO experience header (gold stays experience)
        lines.append(x); gold.append("experience"); heads.append(False)
    lines.append(""); gold.append(None); heads.append(False)
    _section(lines, gold, heads, "Education", "education", _edu(), rnd)
    _section(lines, gold, heads, "Skills", "skills", _skills(), rnd)
    return lines, gold, heads


def _mk_mixed(rnd, _v):
    lines, gold, heads = _contact(rnd)
    _section(lines, gold, heads, "Experience", "experience", _jobs(rnd, 2), rnd)
    # Combined header (gold: primary type) with cert lines inside, no cert header.
    _section(lines, gold, heads, "Skills & Certifications", "skills",
             _skills() + ["AWS Certified Cloud Practitioner, 2021"], rnd)
    # Near-miss header: "awards and honors" != alias "honors and awards".
    _section(lines, gold, heads, "Awards & Honors", "awards",
             ["Employee of the Year, 2020"], rnd)
    return lines, gold, heads


def _mk_multiple(rnd, _v):
    lines, gold, heads = _contact(rnd)
    _section(lines, gold, heads, "Experience", "experience",
             _jobs(rnd, 4, dates=rnd.sample(DATE_FORMATS, 4)), rnd)
    _section(lines, gold, heads, "Skills", "skills", _skills(), rnd)
    return lines, gold, heads


def _mk_custom(rnd, v):
    lines, gold, heads = _contact(rnd)
    _section(lines, gold, heads, "Experience", "experience", _jobs(rnd, 2), rnd)
    picks = [CUSTOM_SECTIONS[(v + j) % len(CUSTOM_SECTIONS)] for j in range(2)]
    for header, gtype, body in picks:
        _section(lines, gold, heads, header, gtype, body, rnd)
    return lines, gold, heads


MAKERS = {"conventional": _mk_conventional, "unusual_headings": _mk_unusual,
          "missing_headers": _mk_missing, "two_column": _mk_twocol,
          "implicit_experience": _mk_implicit, "mixed_sections": _mk_mixed,
          "multiple_jobs": _mk_multiple, "custom_sections": _mk_custom}


def write_item(out_dir: Path, name, lines, gold, heads):
    (out_dir / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
    with open(out_dir / f"{name}.labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["line_id", "text", "gold_section", "is_header"])
        for i, l in enumerate(lines):
            w.writerow([i + 1, l, gold[i] or "", 1 if heads[i] else 0])


def cmd_gen(a):
    rnd = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    print(f"{'stratum':20}{'n':>4}  stresses")
    for stratum, blurb in STRATA:
        for v in range(PER_STRATUM):
            name = f"bench_{stratum}_{v:02d}"
            write_item(out, name, *MAKERS[stratum](rnd, v))
            manifest[name] = stratum
        print(f"{stratum:20}{PER_STRATUM:>4}  {blurb}")
    (out / "suite.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    n = len(manifest)
    print(f"\nwrote {n} resumes + labels + suite.json to {out} (seed={a.seed})")
    print(f"Then run: python bench_sectioning.py eval --data {out} --resolve oracle")


# --------------------------------------------------------------------------
# Four systems on the same pipeline
# --------------------------------------------------------------------------

def train_block_classifier(texts, labels) -> Pipeline:
    """Same recipe as poc train_ml_classifier (Tfidf 1-2 + balanced LogReg)."""
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1,
                                  sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    model.fit(texts, labels)
    return model


def rules_predict(block, is_first: bool) -> str:
    hit = P.explicit_heading(block.heading or "")
    if hit:
        return hit
    if is_first:
        return "contact"
    if P.block_has_experience_shape(block.text.splitlines()):
        return "experience"
    return "other"


def majority_gold(block, gold: dict[int, str], fallback: str) -> str:
    votes = [gold[i] for i in block.line_ids if gold.get(i)]
    return Counter(votes).most_common(1)[0][0] if votes else fallback


def load_items(data_dir: Path):
    items = []
    for txt in sorted(data_dir.glob("bench_*.txt")):
        lab = txt.with_suffix(".labels.csv")
        if not lab.exists():
            continue
        lines = txt.read_text(encoding="utf-8").split("\n")
        gold, is_hdr = {}, {}
        with open(lab, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                i = int(r["line_id"])
                gold[i] = r["gold_section"] or None
                is_hdr[i] = r["is_header"] == "1"
        items.append({"name": txt.stem, "lines": lines, "gold": gold, "is_hdr": is_hdr})
    return items


def llm_resolve_blocks(items_blocks, model_name: str) -> dict[str, dict[int, str]]:
    """One batched call per doc: {doc_name: {line_idx0: section}} for routed blocks."""
    import os
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("resolve=llm needs ANTHROPIC_API_KEY in the environment "
                 "(or use --resolve oracle for the free upper bound).")
    try:
        import anthropic
    except ImportError:
        sys.exit("resolve=llm needs the anthropic package: pip install anthropic")
    client = anthropic.Anthropic()
    out: dict[str, dict[int, str]] = {}
    for name, routed in items_blocks:
        if not routed:
            out[name] = {}
            continue
        payload = [{"id": b.block_id, "heading": b.heading, "text": b.text} for b in routed]
        prompt = (P.LLM_PROMPT.format(block="(see items)") + "\nItems:\n"
                  + json.dumps(payload)
                  + '\nReply with ONLY a JSON object mapping block id to {"section": ..., "confidence": ...}.')
        resp = client.messages.create(model=model_name, max_tokens=2000,
                                      messages=[{"role": "user", "content": prompt}])
        txt = re.sub(r"^```(json)?|```$", "", resp.content[0].text.strip(), flags=re.M).strip()
        try:
            res = json.loads(txt)
        except json.JSONDecodeError:
            print(f"  LLM unparseable for {name}; keeping ML fallback")
            out[name] = {}
            continue
        got: dict[int, str] = {}
        id2block = {b.block_id: b for b in routed}
        for bid, v in res.items():
            sec = v.get("section") if isinstance(v, dict) else v
            if bid in id2block and sec in P.SECTIONS:
                for lid in id2block[bid].line_ids:
                    got[lid] = sec
        out[name] = got
    return out


def score_items(items, pred: dict[str, dict[int, str]]):
    """Line accuracy (every nonblank line has a span-section gold) + header
    accuracy (gold-header lines labelled with the right section)."""
    correct = total = tph = htot = 0
    for d in items:
        for i, line in enumerate(d["lines"], start=1):
            if not line.strip():
                continue
            g = d["gold"].get(i)
            p = pred[d["name"]].get(i)
            total += 1
            correct += (p == g and bool(g))
            if d["is_hdr"].get(i):
                htot += 1
                tph += (p == g and bool(g))
    return {"lineAcc": correct / max(1, total),
            "headerAcc": tph / max(1, htot),
            "n_lines": total, "n_headers": htot}


def cmd_eval(a):
    data = Path(a.data)
    items = load_items(data)
    if len(items) < 5:
        sys.exit(f"Need >=5 bench resumes in {data} (run gen first).")
    man_path = data / "suite.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    print(f"{len(items)} resumes, "
          f"{sum(1 for d in items for i, l in enumerate(d['lines'], 1) if l.strip() and d['gold'].get(i))} "
          f"gold-labelled lines\n")

    # Segment once per doc through the REAL pipeline (MarkItDown included).
    seg, golds = {}, {}
    for d in items:
        raw = P.extract_markdown(str(data / f"{d['name']}.txt"))
        src = P.build_document(raw)
        d["blocks"] = P.segment_blocks(src)
        golds[d["name"]] = d["gold"]
        seg[d["name"]] = d["blocks"]
    print(f"segmented: {sum(len(b) for b in seg.values())} candidate blocks total")

    systems = ["rules", "ml", "mldet", "full"]
    pred: dict[str, dict[str, dict[int, str]]] = {s: {} for s in systems}
    routed_lines = routed_docs = nonblank = 0
    decisions: Counter = Counter()
    routed_flag: dict[str, dict[str, bool]] = {}  # exact per-block routing flags

    groups = list(range(len(items)))
    for tr, te in GroupKFold(n_splits=min(a.folds, len(items))).split(items, groups=groups):
        tr_texts, tr_labels = [], []
        for i in tr:
            for b in seg[items[i]["name"]]:
                tr_texts.append(b.text)
                tr_labels.append(majority_gold(b, golds[items[i]["name"]], "other"))
        model = train_block_classifier(tr_texts, tr_labels)
        for i in te:
            d = items[i]
            for s in systems:
                pred[s][d["name"]] = {}
            for bi, b in enumerate(d["blocks"]):
                ml_sec, ml_conf, alts = P.ml_predict(model, b.text)
                r = rules_predict(b, is_first=(bi == 0))
                c = copy.deepcopy(b)
                c.ml_section, c.ml_confidence = ml_sec, ml_conf
                P.validate_ml_prediction(c, alts)
                routed = bool(c.needs_llm)
                routed_flag.setdefault(d["name"], {})[b.block_id] = routed
                det_label = c.final_section if not routed else ml_sec
                if routed:
                    routed_lines += sum(1 for lid in b.line_ids
                                        if lid <= len(d["lines"]) and d["lines"][lid - 1].strip())
                for lid in b.line_ids:
                    pred["rules"][d["name"]][lid] = r
                    pred["ml"][d["name"]][lid] = ml_sec
                    pred["mldet"][d["name"]][lid] = det_label
                    pred["full"][d["name"]][lid] = det_label  # overwritten below if resolved
                decisions[c.decision] += 1
            nonblank += sum(1 for l in d["lines"] if l.strip())
    routed_docs = sum(1 for d in items if any(routed_flag[d["name"]].values()))

    def is_routed(doc_name: str, block) -> bool:
        return routed_flag[doc_name][block.block_id]

    if a.resolve == "oracle":
        # Upper bound: every block the deterministic layer routed gets its
        # majority-gold label. Confidently-wrong (unrouted) blocks stay wrong:
        # the LLM never sees them, exactly as in production.
        for d in items:
            for b in d["blocks"]:
                if is_routed(d["name"], b):
                    maj = majority_gold(b, d["gold"],
                                        pred["ml"][d["name"]][b.line_ids[0]])
                    for lid in b.line_ids:
                        pred["full"][d["name"]][lid] = maj
    elif a.resolve == "llm":
        llm_out = llm_resolve_blocks(
            [(d["name"], [b for b in d["blocks"] if is_routed(d["name"], b)])
             for d in items],
            a.model)
        for d in items:
            for lid, sec in llm_out.get(d["name"], {}).items():
                pred["full"][d["name"]][lid] = sec
    # resolve == none: full == mldet (ML + deterministic, no LLM).

    names = {"rules": "1) rules only", "ml": "2) ML only",
             "mldet": "3) ML + deterministic",
             "full": f"4) ML + det + {a.resolve}"}
    print(f"\n{'system':28}{'lineAcc':>9}{'headerAcc':>11}")
    M = {}
    for k, nm in names.items():
        M[k] = score_items(items, pred[k])
        m = M[k]
        print(f"{nm:28}{m['lineAcc']:9.3f}{m['headerAcc']:11.3f}")

    strata = sorted({manifest.get(d["name"], "unlabelled") for d in items})
    print(f"\nPer-stratum line accuracy (n docs):")
    print(f"{'stratum':20}{'rules':>8}{'ml':>8}{'mldet':>8}{'full':>8}{'n':>4}")
    stratum_win: dict[str, str] = {}
    for s in strata:
        sub = [d for d in items if manifest.get(d["name"], "unlabelled") == s]
        row = {}
        for k in systems:
            row[k] = score_items(sub, pred[k])["lineAcc"]
        best = max(row.values())
        winners = [k for k, v in row.items() if v == best]
        stratum_win[s] = "+".join(winners)
        print(f"{s:20}{row['rules']:8.3f}{row['ml']:8.3f}"
              f"{row['mldet']:8.3f}{row['full']:8.3f}{len(sub):>4}  best={stratum_win[s]}")

    rl = routed_lines / max(1, nonblank)
    rd = routed_docs / len(items)
    print(f"\nRouting: {rl:.1%} of lines in needs_llm blocks, "
          f"{rd:.0%} of resumes need >=1 LLM call")
    print("Decision mix (pre-LLM): "
          + ", ".join(f"{k}={v}" for k, v in decisions.most_common()))

    n_str = len(strata)
    full_wins = sum(1 for s in strata for w in [stratum_win[s]] if "full" in w.split("+"))
    chk = [
        ("full beats rules-only on lineAcc", M["full"]["lineAcc"] > M["rules"]["lineAcc"]),
        ("full >= ML-only on lineAcc", M["full"]["lineAcc"] >= M["ml"]["lineAcc"]),
        ("ML+det (no LLM) >= ML-only", M["mldet"]["lineAcc"] >= M["ml"]["lineAcc"]),
        ("full wins/ties majority of strata", full_wins * 2 >= n_str),
        ("no header-accuracy regression vs rules",
         M["full"]["headerAcc"] >= M["rules"]["headerAcc"]),
    ]
    print("\nCHECKS")
    for t, ok in chk:
        print(f"  [{'PASS' if ok else 'FAIL'}] {t}")
    key = [ok for _, ok in chk]
    ans = "YES" if all(key) else ("MIXED" if any(key) else "NO")
    print(f"\nANSWER: does ML + deterministic + {a.resolve} outperform simpler "
          f"alternatives? {ans}")
    if a.resolve == "oracle":
        print("(oracle = LLM upper bound: routed blocks assumed perfectly resolved. "
              "Re-run with --resolve llm for the measured number.)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Controlled sectioning benchmark")
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gen", help="Generate the stratified 24-resume suite")
    g.add_argument("--out", default="bench_data")
    g.add_argument("--seed", type=int, default=BENCH_SEED)
    e = sp.add_parser("eval", help="Score rules vs ml vs mldet vs full")
    e.add_argument("--data", default="bench_data")
    e.add_argument("--resolve", choices=["oracle", "llm", "none"], default="oracle")
    e.add_argument("--folds", type=int, default=5)
    e.add_argument("--model", default="claude-haiku-4-5-20251001",
                   help="LLM model id for --resolve llm (or set LLM_MODEL)")
    a = ap.parse_args(argv)
    if a.cmd == "gen":
        cmd_gen(a)
    else:
        import os
        if a.resolve == "llm" and os.getenv("LLM_MODEL"):
            a.model = os.getenv("LLM_MODEL")
        cmd_eval(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
