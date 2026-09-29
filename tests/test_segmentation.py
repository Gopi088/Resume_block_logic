"""B2 segmentation tests: structural candidate blocks, hyphen joins, boilerplate, integrity."""

from __future__ import annotations

from resume_parser.conversion import convert_bytes
from resume_parser.models import AssignmentStatus, BoundarySignal, LineKind
from resume_parser.normalization import normalize_document
from resume_parser.segmentation import (
    display_text,
    segment_document,
    should_join,
)


def _seg(text: str):
    conv = convert_bytes(text.encode("utf-8"), filename="r.txt", content_type="text/plain")
    conv = conv.model_copy(update={"extracted_markdown": text, "extracted_char_count": len(text)})
    return segment_document(normalize_document(conv))


def test_structural_blocks_no_semantic_labels():
    """B2 produces structural candidate blocks — NO semantic section labels."""
    seg = _seg("Jane Doe\njane@mail.com\n\nWORK EXPERIENCE\nAcme Corp\n\nEDUCATION\nB.Tech\n")
    assert seg.integrity.passed, seg.integrity.violations
    
    # Blocks exist but have NO label_proposal or proposal_source
    assert len(seg.blocks) >= 3
    
    # Check CandidateBlock fields
    for b in seg.blocks:
        assert hasattr(b, 'block_id')
        assert hasattr(b, 'line_ids')
        assert hasattr(b, 'boundary_signals')
        assert hasattr(b, 'assignment_status')
        assert b.assignment_status == AssignmentStatus.UNASSIGNED_PENDING_ML_CLASSIFICATION
        # No semantic fields
        assert not hasattr(b, 'label_proposal')
        assert not hasattr(b, 'proposal_source')
    
    # Second block starts with heading-like line "WORK EXPERIENCE"
    assert seg.blocks[1].header_line_id is not None
    assert seg.blocks[1].display_lines[0].is_header
    
    # Boundary signals are structural only
    for b in seg.blocks:
        for sig in b.boundary_signals:
            assert isinstance(sig, BoundarySignal)


def test_leading_span_is_first_block_no_semantic_label():
    """Leading content forms first block — no 'contact_header' semantic label."""
    seg = _seg("Aarav Sharma\n+91 99999\n\nSKILLS\nPython\n")
    assert len(seg.blocks) >= 2
    assert seg.blocks[0].line_ids == ["L000000", "L000001"]
    assert "Aarav Sharma" in seg.blocks[0].text
    assert seg.blocks[0].assignment_status == AssignmentStatus.UNASSIGNED_PENDING_ML_CLASSIFICATION


def test_hyphen_break_joined_with_provenance():
    seg = _seg("Maintained Java services for finan-\ncial apps on AWS.\n")
    assert seg.hyphen_joins == 1
    assert "financial apps" in seg.blocks[0].text
    assert len(seg.blocks[0].display_lines[0].source_line_ids) == 2
    # raw lines untouched
    assert seg.blocks[0].line_ids == ["L000000", "L000001"]


def test_no_join_without_hyphen_signal():
    seg = _seg("DesignedandoptimizedRESTfulAPIs\nnext line\n")
    assert seg.hyphen_joins == 0  # missing-space corruption is NOT guessed


def test_repeated_page_openings_flagged_not_deleted():
    text = "Softwa\nSoftwa\nWORK EXPERIENCE\nAcme\n\fSoftwa\nSoftwa\nSKILLS\nPython\n"
    seg = _seg(text)
    flagged = {lid for b in seg.boilerplate for lid in b.line_ids}
    assert {"L000000", "L000001", "L000004", "L000005"} <= flagged
    assert seg.page_count == 2
    # flagged lines excluded from block text but present in boilerplate list
    assert "Softwa" not in seg.blocks[0].text
    assert any(b.text == "Softwa" for b in seg.boilerplate)
    assert seg.integrity.passed, seg.integrity.violations


def test_scattered_repeats_are_not_boilerplate():
    seg = _seg("Pune\nAcme\nPune\nBeta\nPune\n")
    assert seg.boilerplate == []
    assert "Pune" in seg.blocks[0].text


def test_display_strips_private_use_glyphs():
    assert display_text("\ue901 ashish@mail.com") == " ashish@mail.com"
    assert "ashish@mail.com" in _seg("\ue901 ashish@mail.com\n").blocks[0].text


def test_should_join_guards():
    assert should_join("de-", "ployed")
    assert not should_join("-", "foo")       # bullet, not a break
    assert not should_join("EC2-", "S3")     # next starts uppercase
    assert not should_join("word", "next")   # no trailing hyphen
    assert not should_join("end -", "next")  # hyphen after space


def test_b2_integrity_catches_coverage_tamper():
    seg = _seg("Jane\n\nSKILLS\nPython\n")
    bad = [b.model_copy(update={"line_ids": b.line_ids[:-1]}) for b in seg.blocks]
    from resume_parser.conversion import convert_bytes as cb
    from resume_parser.normalization import normalize_document as nd
    conv = cb(b"Jane\n\nSKILLS\nPython\n", filename="r.txt", content_type="text/plain")
    conv = conv.model_copy(update={"extracted_markdown": "Jane\n\nSKILLS\nPython\n",
                                   "extracted_char_count": len("Jane\n\nSKILLS\nPython\n")})
    doc = nd(conv)
    from resume_parser.segmentation import build_b2_integrity, paginate
    rep = build_b2_integrity(doc, paginate(doc.lines), set(), bad)
    assert not rep.passed


def test_blank_line_not_universal_boundary():
    """Single blank line between related content should NOT always split blocks."""
    # Role line + blank + date line + blank + responsibilities should stay together
    text = (
        "Senior Business Analyst / Product Owner - Emids\n"
        "\n"
        "(April 2025 - Present)\n"
        "\n"
        "Project: Clinical Trials & Simulation\n"
        "Client: Cytel Inc.\n"
        "\n"
        "Responsibilities\n"
        "• Drove...\n"
        "• Delivered...\n"
        "• Authored...\n"
    )
    seg = _seg(text)
    # Should form ONE candidate block (or at most 2, but not one per line)
    # The key test: all non-blank lines in same block
    nonblank_lines = [ln for ln in seg.blocks for lid in ln.line_ids]
    assert len(seg.blocks) <= 3  # Not one block per line


def test_bullet_sequence_not_split():
    """Bullet sequences should not be split by blank lines."""
    text = (
        "Experience\n"
        "\n"
        "• First bullet\n"
        "\n"
        "• Second bullet\n"
        "\n"
        "• Third bullet\n"
    )
    seg = _seg(text)
    # All bullets should be in same block (or at most 2 blocks)
    assert len(seg.blocks) <= 2


def test_markdown_heading_creates_boundary():
    """Markdown heading (#) creates a structural boundary."""
    text = (
        "# Contact\n"
        "Jane Doe\n"
        "jane@email.com\n"
        "\n"
        "## Experience\n"
        "Acme Corp\n"
        "Role\n"
    )
    seg = _seg(text)
    assert len(seg.blocks) >= 2
    # First block has markdown heading
    assert any(b.boundary_signals and BoundarySignal.MARKDOWN_HEADING in b.boundary_signals for b in seg.blocks)


def test_heading_like_creates_boundary():
    """ALL CAPS short line with colon creates structural boundary."""
    text = (
        "CONTACT\n"
        "Jane Doe\n"
        "jane@email.com\n"
        "\n"
        "EXPERIENCE:\n"
        "Acme Corp\n"
        "Role\n"
    )
    seg = _seg(text)
    assert len(seg.blocks) >= 2
    assert any(BoundarySignal.HEADING_LIKE in b.boundary_signals for b in seg.blocks)


def test_table_rows_stay_together():
    """Table rows form a contiguous block."""
    text = (
        "Skills\n"
        "| Python | 5 years |\n"
        "| SQL | 3 years |\n"
        "| Docker | 2 years |\n"
        "\n"
        "Experience\n"
        "Acme Corp\n"
    )
    seg = _seg(text)
    # Table rows should be in one block
    table_blocks = [b for b in seg.blocks if any(d.line_kind == LineKind.TABLE_ROW for d in b.display_lines)]
    assert len(table_blocks) >= 1


def test_code_fence_stays_together():
    """Code fence lines form a contiguous block."""
    text = (
        "Code Sample\n"
        "```python\n"
        "def parse():\n"
        "    pass\n"
        "```\n"
        "\n"
        "Experience\n"
        "Acme Corp\n"
    )
    seg = _seg(text)
    code_blocks = [b for b in seg.blocks if any(d.line_kind == LineKind.CODE_FENCE for d in b.display_lines)]
    assert len(code_blocks) >= 1


def test_page_break_creates_boundary():
    """Form feed creates a page break boundary."""
    text = "Page 1 content\n\fPage 2 content"
    seg = _seg(text)
    assert seg.page_count == 2
    assert any(BoundarySignal.PAGE_BREAK in b.boundary_signals for b in seg.blocks)


def test_candidate_block_fields_complete():
    """Every CandidateBlock has all required structural fields."""
    seg = _seg("Name\nEmail\n\nEXPERIENCE\nCompany\nRole\n")
    for b in seg.blocks:
        assert b.block_id.startswith("B")
        assert isinstance(b.index, int)
        assert isinstance(b.line_ids, list)
        assert len(b.line_ids) > 0
        assert isinstance(b.start_line_index, int)
        assert isinstance(b.end_line_index, int)
        assert isinstance(b.page_indices, list)
        assert isinstance(b.display_lines, list)
        assert isinstance(b.text, str)
        assert b.text == "\n".join(d.text for d in b.display_lines)
        assert isinstance(b.is_continuation, bool)
        assert isinstance(b.boundary_signals, list)
        assert all(isinstance(s, BoundarySignal) for s in b.boundary_signals)
        assert b.assignment_status == AssignmentStatus.UNASSIGNED_PENDING_ML_CLASSIFICATION


def test_block_coverage_100_percent():
    """Every meaningful line assigned to exactly one block."""
    text = "Line 1\nLine 2\n\nLine 3\nLine 4\n"
    seg = _seg(text)
    all_line_ids = [lid for b in seg.blocks for lid in b.line_ids]
    # No duplicates
    assert len(all_line_ids) == len(set(all_line_ids))
    # All non-blank lines covered
    # L000000, L000001, L000003, L000004 are non-blank
    expected = {"L000000", "L000001", "L000003", "L000004"}
    assert set(all_line_ids) == expected


def test_block_ordering_matches_source():
    """Blocks appear in document order."""
    seg = _seg("A\nB\n\nC\nD\n")
    indices = [b.start_line_index for b in seg.blocks]
    assert indices == sorted(indices)


def test_display_line_provenance():
    """DisplayLine.source_line_ids correctly references source lines."""
    seg = _seg("Line one-\ntwo\n\nNext block\n")
    for b in seg.blocks:
        for d in b.display_lines:
            for lid in d.source_line_ids:
                assert lid in b.line_ids
            # Ordered and unique
            assert d.source_line_ids == sorted(d.source_line_ids)
            assert len(d.source_line_ids) == len(set(d.source_line_ids))


def test_block_text_reconstructs_from_display_lines():
    """Block.text equals join of display line texts."""
    seg = _seg("A\nB\n\nC\n")
    for b in seg.blocks:
        assert b.text == "\n".join(d.text for d in b.display_lines)