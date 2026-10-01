"""Backward-compatible final JSON section confidence and span projection."""

from print_resume_json import _section_meta_with_detection, semantic_entries


def test_section_meta_uses_observed_minimum_raw_ml_probability():
    meta = _section_meta_with_detection(
        [{"id": "workExperience", "key": "workExperience", "order": 2}],
        [
            {"section": "experience", "mlConfidence": 0.81,
             "verificationStatus": "VERIFIED_BY_B4", "blockId": "B1",
             "lineIds": ["L1"]},
            {"section": "experience", "mlConfidence": 0.42,
             "verificationStatus": "UNRESOLVED", "blockId": "B2",
             "lineIds": ["L2"]},
        ],
        [{"leftSection": "summary", "rightSection": "experience",
          "boundaryConfidence": None, "startLine": "L2", "endLine": "L1"}],
        {"sectionsSent": 0},
        {"total_meaningful_lines": 2, "assigned_lines": 2,
         "missing_lines": 0, "duplicate_assignments": 0,
         "unassigned_lines": 0, "coverage_percent": 100.0},
        {"spanCount": 2, "averageRawConfidence": 0.615,
         "highCount": 1, "mediumCount": 0, "lowCount": 1},
    )
    detection = meta[0]["detection"]
    assert detection["confidence"] == 0.42
    assert detection["confidenceMethod"].startswith("minimum_raw_ML")
    assert detection["calibrationStatus"] == "RAW_UNCALIBRATED_NO_GOLD_VALIDATION_SET"
    assert detection["spanDetections"][1]["lineIds"] == ["L2"]
    assert detection["boundaries"][0]["boundaryConfidence"] is None
    assert detection["lineCoverage"]["missing_lines"] == 0
    assert detection["rawConfidenceSummary"]["spanCount"] == 2


def test_semantic_entries_keep_mixed_spans_and_untrusted_content():
    out = {
        "classification": {"classifications": [{
            "block_id": "B000000", "predicted_section": "summary",
            "semantic_spans": [
                {"section": "summary", "text": "Profile text", "confidence": 0.8,
                 "source_line_ids": ["L000000"], "start_line_id": "L000000",
                 "end_line_id": "L000000"},
                {"section": "experience", "text": "Job text", "confidence": 0.6,
                 "source_line_ids": ["L000001"], "start_line_id": "L000001",
                 "end_line_id": "L000001"},
            ]}],
        },
        "final_sections": {"sections": [{"block_id": "B000000",
                            "final_section": "summary", "trusted": False,
                            "source": "ml_unresolved"}]},
    }
    entries = semantic_entries(out, [])
    assert [e["section"] for e in entries] == ["summary", "experience"]
    assert [e["text"] for e in entries] == ["Profile text", "Job text"]
    assert all(not e["trusted"] for e in entries)
    ids = [line_id for entry in entries for line_id in entry["line_ids"]]
    assert ids == ["L000000", "L000001"]


def test_table_job_items_pair_dates_and_roles_from_two_column_resume_rows():
    from print_resume_json import table_job_items

    jobs = table_job_items([{
        "text": "| **08/2024 – Present** **Company - Mantra Softech Pvt. Ltd.** **Technology** - * BI | Sr. Manager (Data Analysis) Project Description – Biometrics. * Led delivery. |"
    }])
    assert len(jobs) == 1
    assert jobs[0]["title"] == "Sr. Manager (Data Analysis)"
    assert jobs[0]["company"] == "Mantra Softech Pvt. Ltd."
    assert jobs[0]["years"] == "08/2024 – Present"
    assert "Led delivery." in jobs[0]["description"]


def test_date_line_rejects_years_embedded_in_skill_names():
    from print_resume_json import _is_date_line

    assert _is_date_line("02/2023 – 08/2024")
    assert _is_date_line("July 2020 to Present")
    assert not _is_date_line("MS Sql 2008 &")


def test_skill_items_split_markdown_table_bullets_and_keep_single_letter_skill():
    from print_resume_json import skill_items

    assert skill_items([{"text": "| * Python * R * Tableau | * Power BI |"}]) == [
        "Python", "R", "Tableau", "Power BI"]


def test_mapper_preserves_b3_label_instead_of_leaking_previous_heading():
    out = {
        "classification": {"classifications": [
            {"block_id": "B1", "predicted_section": "skills", "semantic_spans": [
                {"section": "skills", "text": "Languages", "source_line_ids": ["L1"],
                 "start_line_id": "L1", "end_line_id": "L1", "confidence": 0.4}]},
            {"block_id": "B2", "predicted_section": "skills", "semantic_spans": [
                {"section": "experience", "text": "Data Scientist\nJune 2022 - Present", "source_line_ids": ["L2"],
                 "start_line_id": "L2", "end_line_id": "L2", "confidence": 0.6}]},
        ]},
        "final_sections": {"sections": []},
    }
    entries = semantic_entries(out, [])
    assert [entry["section"] for entry in entries] == ["skills", "experience"]
    assert entries[1]["section_source"] == "model"


def test_technical_expertise_heading_projects_to_skills_not_summary():
    # General vocabulary alignment with B3 aliases: a "Technical Expertise"
    # span must not be absorbed by a preceding summary heading context.
    out = {
        "classification": {"classifications": [
            {"block_id": "B1", "predicted_section": "summary", "semantic_spans": [
                {"section": "summary", "text": "Professional Summary\nExperienced lead",
                 "source_line_ids": ["L1"], "start_line_id": "L1",
                 "end_line_id": "L1", "confidence": 0.4}]},
            {"block_id": "B2", "predicted_section": "skills", "semantic_spans": [
                {"section": "skills", "text": "Technical Expertise\nPrimary Skills: Python",
                 "source_line_ids": ["L2"], "start_line_id": "L2",
                 "end_line_id": "L2", "confidence": 0.49}]},
        ]},
        "final_sections": {"sections": []},
    }
    entries = semantic_entries(out, [])
    assert [entry["section"] for entry in entries] == ["summary", "skills"]
    assert entries[1]["section_source"] == "model"


def test_job_confidence_ignores_heading_override_spans_without_scores():
    # Detections projected via heading context carry mlConfidence=None; the
    # per-job weakest-confidence rollup must skip them instead of crashing
    # in min() (Ankur Jain.pdf regression: TypeError before the fix).
    detections = [
        {"section": "experience", "mlConfidence": None, "lineIds": ["L1"],
         "blockId": "B1"},
        {"section": "experience", "mlConfidence": 0.3, "lineIds": ["L1"],
         "blockId": "B1"},
    ]
    values = [d["mlConfidence"] for d in detections
              if d["section"] == "experience"
              and isinstance(d.get("mlConfidence"), (int, float))
              and set(["L1"]).intersection(d.get("lineIds", []))]
    assert values == [0.3]
    assert round(min(values), 4) == 0.3


def test_bold_lines_are_not_bullets_and_keep_their_markers_for_titles():
    from print_resume_json import BULLET_RE, clean_bullets

    assert not BULLET_RE.match("**Senior Business Analyst - Avaloq Wealth**")
    assert clean_bullets("**Senior Business Analyst - Avaloq Wealth**") == \
        "**Senior Business Analyst - Avaloq Wealth**"
    assert BULLET_RE.match("•Served as a key analyst")
    assert clean_bullets("•Served as a key analyst") == "Served as a key analyst"
    assert BULLET_RE.match("- Led delivery")


def test_group_jobs_splits_date_led_entries_and_keeps_title_with_date():
    from print_resume_json import group_jobs

    entries = [
        {"entry_id": "B:S0000", "text": "# WORK EXPERIENCE\n"
         "**Senior Business Analyst - Avaloq Wealth**\n"
         "02/2023 – 08/2025\nSydney, Australia"},
        {"entry_id": "B:S0001", "text": "01/2020 – 01/2023\nSydney, Australia"},
    ]
    jobs = group_jobs(entries, {})
    assert len(jobs) == 2
    first_texts = [t for t, _ in jobs[0]]
    assert first_texts[1] == "**Senior Business Analyst - Avaloq Wealth**"
    assert first_texts[2] == "02/2023 – 08/2025"


def test_job_years_finds_inline_month_name_ranges():
    from print_resume_json import job_years_from_lines

    assert job_years_from_lines(
        ["Technical Lead | Coforge Ltd. | Nov 2025 – Present"], [], {}) == \
        "Nov 2025 – Present"
    assert job_years_from_lines(
        [".Net Engineer | HCL Tech | Oct 2024 – Sep 2025"], [], {}) == \
        "Oct 2024 – Sep 2025"
    assert job_years_from_lines(["02/2023 – 08/2025"], [], {}) == \
        "02/2023 – 08/2025"


def test_job_item_strips_markdown_emphasis_from_title_and_company():
    from print_resume_json import job_item

    job = job_item(1, [("**Senior Business Analyst - Avaloq Wealth**", "E1"),
                       ("02/2023 – 08/2025", "E1")], {})
    assert job["title"] == "Senior Business Analyst"
    assert job["company"] == "Avaloq Wealth"
    assert job["years"] == "02/2023 – 08/2025"


def test_public_section_meta_omits_internal_source_ids_and_text():
    from print_resume_json import _public_section_meta

    result = _public_section_meta([{
        "id": "workExperience", "key": "workExperience",
        "detection": {
            "confidence": 0.4, "verificationStatus": "NOT_VERIFIED",
            "lineCoverage": {"coverage_percent": 100.0},
            "spanDetections": [{"section": "experience", "mlConfidence": 0.4,
                                "blockId": "B000001", "lineIds": ["L000001"],
                                "text": "private source text", "featureContext": ["cue"]}],
            "boundaries": [{"leftSection": "summary", "rightSection": "experience",
                             "startLine": "L1", "endLine": "L2", "boundaryConfidence": None}],
        },
    }])
    detection = result[0]["detection"]
    assert detection["confidence"] == 0.4
    assert detection["spanDetections"] == [{"section": "experience", "mlConfidence": 0.4}]
    assert detection["boundaries"] == [{"leftSection": "summary", "rightSection": "experience",
                                          "boundaryConfidence": None}]
