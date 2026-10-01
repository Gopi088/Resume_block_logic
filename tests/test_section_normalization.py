import json
import pytest
from resume_parser.section_normalization import normalize_heading, genuine_heading, normalize_resume
from resume_parser.source_evaluation import evaluate_extraction, concise_evaluation, verify_extraction, source_blocks
from print_resume_json import structured_resume


def document(*lines):
    return {'lines': [{'line_id': f'L{i}', 'display_text': text} for i, text in enumerate(lines)]}


@pytest.mark.parametrize('heading', ['SKILLS', 'TECHNICAL SKILLS', 'TECH STACK', 'TECHNOLOGIES', 'CORE COMPETENCIES', 'KEY SKILLS', 'TOOLS & TECHNOLOGIES', 'AREAS OF EXPERTISE', 'Professional Skills', 'Technical-Skill:', 'Tools/Technologies', '  TECHNICAL   SKILLS :'])
def test_skill_aliases(heading):
    assert normalize_heading(heading) == 'technicalSkills'


@pytest.mark.parametrize('heading,key', [('PROFESSIONAL EXPERIENCE', 'workExperience'), ('EMPLOYMENT HISTORY', 'workExperience'), ('CAREER HISTORY', 'workExperience'), ('ACADEMICS', 'education'), ('ACADEMIC BACKGROUND', 'education'), ('EDUCATIONAL QUALIFICATIONS', 'education'), ('EDUCATION & TRAINING', 'education'), ('SELECTED PROJECTS', 'projects'), ('PROFESSIONAL PROJECTS', 'projects'), ('CERTIFICATIONS & TRAINING', 'certifications'), ('LICENSES & CERTIFICATIONS', 'certifications'), ('COURSES', 'certifications'), ('RECOGNITION', 'awards'), ('LANGUAGE PROFICIENCY', 'languages')])
def test_other_aliases(heading, key):
    assert normalize_heading(heading) == key


def test_toolkit_fallback_and_unknown_narrative_sections_are_preserved():
    out = document('MY TOOLKIT', 'Python', 'SQL', 'AWS', 'Docker', 'WHAT I BRING', 'Led teams', 'Managed projects', 'Reduced deployment time', 'PUBLICATIONS', 'Published a paper', 'Research', 'Investigated methods')
    resume, detections = structured_resume(out)
    clean = normalize_resume(resume)
    assert clean['technicalSkills'] == ['Python', 'SQL', 'AWS', 'Docker']
    assert {entry['title'] for entry in clean['additionalSections']} == {'WHAT I BRING', 'PUBLICATIONS', 'Research'}
    assert normalize_heading('WHAT I BRING', ['Led teams', 'Managed projects']) is None
    result = evaluate_extraction(out, clean, detections)
    class Judge:
        prompts = []
        def complete(self, prompt):
            self.prompts.append(prompt)
            return json.dumps({'status': 'VERIFIED_CORRECT', 'confidence': .9, 'reason': 'Supported', 'missing': [], 'incorrect': [], 'hallucinated': []})
    judge = Judge()
    verify_extraction(result, judge)
    assert len(judge.prompts) == 4
    payloads = [json.loads(prompt.split('\n', 1)[1]) for prompt in judge.prompts]
    assert payloads[0]['block'] == 'technicalSkills'
    assert payloads[0]['original'] == ['Python', 'SQL', 'AWS', 'Docker']
    assert all('Led teams' not in p['original'] for p in payloads if p['block'] == 'technicalSkills')
    public = concise_evaluation(result)
    assert not public['missingBlocks']
    assert all(b['block'] in {'technicalSkills', 'additionalSections'} for b in public['blocks'])
    assert 'additionalSections::' not in json.dumps(public)


@pytest.mark.parametrize('text', ['• MS SQL', '• ETL', '• R', '• T-SQL', '• ELT', 'MS SQL', 'NETWORK LTD.', 'U.P. BOARD'])
def test_content_is_never_a_heading(text):
    assert not genuine_heading(text, ['Managed projects'], {'block': 'technicalSkills'})


def test_repeated_skill_sections_aggregate_and_preserve_short_skills():
    out = document('SKILLS', '• MS SQL', '• ETL', '• R', 'TECHNOLOGIES', '• T-SQL', '• ELT', 'TOOLS', 'Docker')
    resume, detections = structured_resume(out)
    clean = normalize_resume(resume)
    assert clean['technicalSkills'] == ['MS SQL', 'ETL', 'R', 'T-SQL', 'ELT', 'Docker']
    result = evaluate_extraction(out, clean, detections)
    assert len(result['blockComparison']) == 1
    assert result['blockComparison'][0]['status'] == 'correct'
    assert not result['missingBlocks']


def test_combined_contact_line_has_independent_field_results():
    out = document('ALICE SMITH', 'alice@example.com M:9557824150 | Bengaluru, Karnataka 560100', 'PERSONAL INFORMATION', 'Date of Birth: 05/07/93')
    resume, detections = structured_resume(out)
    result = evaluate_extraction(out, normalize_resume(resume), detections)
    row = concise_evaluation(result)['blocks'][0]
    assert row['extractionEvaluation']['status'] == 'CORRECT'
    assert set(row['extractionEvaluation']['fields']) == {'name', 'email', 'phone', 'location', 'dateOfBirth'}
    assert all(v['status'] == 'CORRECT' for v in row['extractionEvaluation']['fields'].values())
    resume['personalInfo']['email'] = 'wrong@example.com'
    bad = concise_evaluation(evaluate_extraction(out, resume, detections))['blocks'][0]
    assert bad['extractionEvaluation']['fields']['email']['status'] == 'INCORRECT'


def test_normalization_merges_aliases_without_losing_values_and_is_idempotent():
    raw = {'technicalSkills': ['MS SQL'], 'additional': {'technicalSkills': ['R', 'ETL'], 'certificationsTraining': ['Certificate']}, 'TECH STACK': ['AWS'], 'personalProjects': [{'title': 'App'}], 'customSections': {'PUBLICATIONS': {'text': 'Paper'}}, 'additionalSections': [{'title': 'PUBLICATIONS', 'content': ['Second paper']}]}
    clean = normalize_resume(raw)
    assert clean['technicalSkills'] == ['MS SQL', 'AWS', 'R', 'ETL']
    assert clean['projects'] == [{'title': 'App'}]
    assert clean['certifications'] == ['Certificate']
    assert clean['additionalSections'] == [{'title': 'PUBLICATIONS', 'content': ['Second paper', 'Paper']}]
    assert normalize_resume(clean) == clean


def test_headingless_spans_keep_existing_model_section_boundaries():
    out = document('Python SQL', 'Built systems at Acme')
    out['classification'] = {'classifications': [{'semantic_spans': [{'section': 'skills', 'source_line_ids': ['L0']}, {'section': 'experience', 'source_line_ids': ['L1']}]}]}
    assert [b['block'] for b in source_blocks(out)] == ['technicalSkills', 'workExperience']


def test_date_led_role_and_company_rows_remain_one_job():
    out = document('WORK EXPERIENCE', '12/2021 - 07/2022', 'Assistant Manager', 'Company- Example', 'Worked on data systems.', 'EDUCATION', 'BSc Mathematics')
    resume, _ = structured_resume(out)
    assert len(resume['workExperience']) == 1
    assert resume['workExperience'][0]['years'] == '12/2021 - 07/2022'
    assert 'Worked on data systems.' in resume['workExperience'][0]['description']


def test_phone_formatting_is_not_missing_content():
    out = document('ALICE SMITH', 'Email: alice@example.com | Phone: +1 (555) 123-4567', 'Location: Boston, MA')
    result = evaluate_extraction(out, {'personalInfo': {'name': 'ALICE SMITH', 'email': 'alice@example.com', 'phone': '15551234567', 'location': 'Boston, MA'}}, [])
    row = result['blockComparison'][0]
    assert row['status'] == 'correct'
    assert row['fieldComparison']['phone']['status'] == 'CORRECT'


def test_semantic_fallback_uses_model_context_but_never_content_fragments():
    assert genuine_heading('MY CAPABILITIES', ['Python', 'SQL', 'AWS'], model_labels=['skills'])
    assert normalize_heading('MY CAPABILITIES', ['Python', 'SQL', 'AWS'], model_labels=['skills']) == 'technicalSkills'
    assert not genuine_heading('Python', ['SQL', 'AWS'], model_labels=['skills'])


def test_unknown_sections_are_not_overridden_by_employment_content():
    out = document('RESEARCH', '2020 - 2022', 'Company: Example Ltd', 'Developed methods for industry')
    assert {b['block'] for b in source_blocks(out)} == {'additionalSections::RESEARCH'}


def test_duplicate_in_same_section_is_incorrect_content_not_wrong_block():
    out = document('SKILLS', 'Python')
    row = concise_evaluation(evaluate_extraction(out, {'technicalSkills': ['Python', 'Python']}, []))['blocks'][0]
    assert row['extractionEvaluation']['incorrectContent'] == ['Python']
    assert row['extractionEvaluation']['wrongBlockContent'] == []


def test_personal_field_assignment_errors_override_perfect_token_retention():
    out = document('ALICE SMITH', 'alice@example.com', 'Location: Boston, MA')
    resume = {'personalInfo': {'name': 'Boston, MA', 'email': 'alice@example.com', 'location': 'ALICE SMITH'}}
    row = concise_evaluation(evaluate_extraction(out, resume, []))['blocks'][0]
    assert row['extractionEvaluation']['status'] == 'INCORRECT'
    assert row['extractionEvaluation']['fields']['name']['status'] == 'INCORRECT'
    assert row['extractionEvaluation']['incorrectContent']
