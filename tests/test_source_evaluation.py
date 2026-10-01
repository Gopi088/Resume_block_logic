from resume_parser.source_evaluation import evaluate_extraction
from resume_parser.resolution import complete_json
import pytest


def document(*texts):
    return {'lines': [{'line_id': f'L{i}', 'display_text': t} for i, t in enumerate(texts)]}


def test_absent_sections_and_partial_source_content():
    out = document('SKILLS', 'Python', 'SQL', 'EDUCATION', 'BSc Mathematics')
    result = evaluate_extraction(out, {'additional': {'technicalSkills': ['Python']}}, [])
    assert [b['block'] for b in result['missingBlocks']] == ['education']
    assert result['partialBlocks'][0]['missingContent'] == ['SQL']
    assert result['sourceLineCoverage'] == pytest.approx(100/3, abs=.01)
    assert 0 < result['cosineSimilarity'] < 1


def test_extra_content_is_distinct_from_misplaced_source_content():
    out = document('SKILLS', 'Python', 'WORK EXPERIENCE', 'Engineer Acme')
    result = evaluate_extraction(out, {'additional': {'technicalSkills': ['Python', 'Engineer Acme', 'InventedCloud']}}, [])
    assert result['extraContent'][0]['extractedContent'] == ['InventedCloud']
    assert result['misplacedContent'][0]['extractedContent'] == ['Engineer Acme']


def test_json_retry_and_failure():
    class Client:
        def __init__(self, responses):
            self.responses = iter(responses)
            self.calls = 0
        def complete(self, prompt):
            self.calls += 1
            return next(self.responses)
    client = Client(['{"x": "unescaped"quote"}', '```json\n{"x": true}\n```'])
    assert complete_json(client, 'judge') == {'x': True}
    assert client.calls == 2
    client = Client(['{"x":', '{"x":'])
    with pytest.raises(ValueError):
        complete_json(client, 'judge')
    assert client.calls == 2


def test_date_only_job_is_not_emitted():
    from print_resume_json import job_item
    assert job_item(1, [('2020 - 2021', 'E1')], {}) is None


def test_failed_judge_never_verifies_content():
    from resume_parser.source_evaluation import verify_extraction
    result = evaluate_extraction(document('SKILLS', 'Python'), {'additional': {'technicalSkills': ['Python']}}, [])
    class Broken:
        def complete(self, prompt):
            return '{"status":'
    verify_extraction(result, Broken(), True)
    assert result['blockComparison'][0]['llmVerification']['status'] == 'LLM_ERROR'
    assert result['finalAccuracy']['llmVerifiedAccuracy'] is None


def test_skills_do_not_include_unrelated_source_blocks():
    from print_resume_json import skill_items
    entries = [{'text': 'Python\nAcme Engineer 2020-2021', 'line_ids': []}]
    assert skill_items(entries, allowed_source='Python SQL') == ['Python']


def test_explicit_boundaries_split_employers_and_individual_projects():
    from print_resume_json import structured_resume
    from resume_parser.structure_projection import public_resume
    out = document('ALICE SMITH', 'WORK EXPERIENCE', 'Sonata Software Ltd : Dec 2023- till now',
                   'Tiger Analytics : May 2022 – Nov 2023', 'Tech Mahindra : July 2018- May 2019',
                   'PROJECTS and POCs', 'POC: 1', 'Title: RAG Demo', 'Built a retriever.',
                   'Project: 1', 'Client Name: Acme', 'Role: Engineer', 'Duration: 2020-2021',
                   'Delivered software.', 'EDUCATION', 'BSc from Example College 2013-2017',
                   'CONTACT DETAILS', 'Email: alice@example.com')
    # Simulate the actual failure: the classifier calls the entire source experience.
    out['classification'] = {'classifications': [{'block_id': 'B1', 'semantic_spans': [
        {'section': 'experience', 'confidence': .99, 'source_line_ids': [l['line_id'] for l in out['lines']]}]}]}
    resume, detections = structured_resume(out)
    clean = public_resume(resume)
    assert [j['company'] for j in clean['workExperience']] == ['Sonata Software Ltd', 'Tiger Analytics', 'Tech Mahindra']
    assert len(clean['personalProjects']) == 2
    assert clean['personalProjects'][0]['title'] == 'RAG Demo'
    assert 'client' not in clean['personalProjects'][0]
    assert clean['personalProjects'][1]['client'] == 'Acme'
    assert clean['personalInfo']['name'] == 'ALICE SMITH'
    assert clean['personalInfo']['email'] == 'alice@example.com'
    assert all(not j['description'] for j in clean['workExperience'])
    result = evaluate_extraction(out, resume, detections)
    assert not result['missingBlocks']
    assert next(b for b in result['blockComparison'] if b['block'] == 'projects')['modelConfidence'] is None
    import json
    from resume_parser.source_evaluation import concise_evaluation
    output = json.dumps({'resume': clean, 'evaluation': concise_evaluation(result)})
    assert not any(k in output for k in ('sourceLineIds', 'line_ids', 'entry_id', 'originalResumeBlocks', 'B1', 'L0'))


def test_wrong_judge_approval_is_vetoed_and_only_relevant_blocks_are_sent():
    from resume_parser.source_evaluation import verify_extraction
    result = evaluate_extraction(document('SKILLS', 'Python', 'WORK EXPERIENCE', 'Engineer Acme'),
                                 {'additional': {'technicalSkills': ['Python', 'Engineer Acme']}},
                                 [{'section': 'skills', 'mlConfidence': .96, 'lineIds': ['L1']}])
    class Judge:
        prompts = []
        def complete(self, prompt):
            self.prompts.append(prompt)
            return '{"status":"CORRECT","confidence":0.99,"missing":[],"incorrect":[],"hallucinated":[],"reason":"looks good"}'
    judge = Judge()
    verify_extraction(result, judge)
    assert len(judge.prompts) == 1  # The missing experience block has no extraction to review.
    assert result['blockComparison'][0]['llmVerification']['status'] == 'VERIFIED_INCORRECT'
    assert result['finalAccuracy']['llmVerifiedAccuracy'] == 0


def test_timeout_never_becomes_verified():
    from resume_parser.source_evaluation import verify_extraction
    result = evaluate_extraction(document('SKILLS', 'Python'), {'additional': {'technicalSkills': ['Python']}}, [])
    class Timeout:
        def complete(self, prompt):
            raise TimeoutError('timeout')
    verify_extraction(result, Timeout(), True)
    assert result['blockComparison'][0]['llmVerification']['status'] == 'LLM_ERROR'
    assert result['blockComparison'][0]['llmVerification']['reason'] == 'timeout'
    assert result['finalAccuracy']['llmVerifiedAccuracy'] is None


def test_list_numbering_is_formatting_and_duplicate_values_are_errors():
    out = document('PROJECTS', 'Project: 1', '1. Built useful software.')
    result = evaluate_extraction(out, {'personalProjects': [{'title': 'Project: 1', 'description': ['Built useful software.']}]}, [])
    assert result['blockComparison'][0]['status'] == 'correct'
    result = evaluate_extraction(document('SKILLS', 'Python'), {'additional': {'technicalSkills': ['Python', 'Python']}}, [])
    assert result['blockComparison'][0]['status'] == 'incorrect'
    assert result['finalAccuracy']['contentAccuracy'] == 50


def test_configurable_transport_timeout_does_not_retry_large_prompt(monkeypatch):
    import urllib.request
    from resume_parser.models import LLMConfig
    from resume_parser.resolution import NvidiaNIMLLMClient
    calls = []
    monkeypatch.setenv('NVIDIA_API_KEY', 'test-key')
    def timed_out(request, timeout):
        calls.append(timeout)
        raise TimeoutError('read timed out')
    monkeypatch.setattr(urllib.request, 'urlopen', timed_out)
    client = NvidiaNIMLLMClient(LLMConfig(timeout_seconds=.1, transport_retries=0))
    with pytest.raises(RuntimeError, match='timed out after 1 attempt'):
        client.complete('Only one original/output block')
    assert calls == [.1]


def test_custom_heading_stops_previous_section_content():
    from print_resume_json import structured_resume
    out = document('ALICE', 'WORK EXPERIENCE', 'Acme Ltd : 2020-2021',
                   'RESEARCH IMPACT', 'Published an important result.', 'SKILLS', 'Python')
    resume, _ = structured_resume(out)
    assert resume['workExperience'][0]['company'] == 'Acme Ltd'
    assert resume['workExperience'][0]['description'] == []
    assert resume['customSections']['RESEARCH IMPACT']['text'] == 'Published an important result.'
    assert resume['additional']['technicalSkills'] == ['Python']


def test_extra_section_with_real_content_is_wrong_block_not_hallucination():
    from resume_parser.source_evaluation import concise_evaluation
    result = evaluate_extraction(document('SKILLS', 'Python'),
                                 {'additional': {'technicalSkills': ['Python'], 'languages': ['Python']}}, [])
    extra = next(b for b in concise_evaluation(result)['blocks'] if b['block'] == 'languages')
    assert extra['extractionEvaluation']['wrongBlockContent'] == ['Python']
    assert extra['extractionEvaluation']['hallucinatedContent'] == []
    assert extra['extractionEvaluation']['cosineSimilarity'] is None
    assert result['finalAccuracy']['blockAccuracy'] == 50


def test_unnumbered_project_titles_start_distinct_entries():
    from resume_parser.structure_projection import project_items
    from print_resume_json import clean_bullets
    entries = [{'text': 'Project: Inventory App\nBuilt an application.\nTitle: Search Tool\nIndexed documents.', 'line_ids': []}]
    result = project_items(entries, clean_bullets)
    assert [p['title'] for p in result] == ['Inventory App', 'Search Tool']
    assert result[0]['description'] == ['Built an application.']
    assert result[1]['description'] == ['Indexed documents.']


def test_raw_confidence_is_not_hidden_without_calibration():
    from resume_parser.source_evaluation import concise_evaluation
    result = evaluate_extraction(document('SUMMARY', 'Built software.'), {'summary': {'text': 'Built software.'}},
                                 [{'section': 'summary', 'mlConfidence': .7515, 'lineIds': ['L1']}])
    row = concise_evaluation(result)['blocks'][0]
    assert row['modelConfidence']['value'] == .7515
    assert row['modelConfidence']['type'] == 'raw_model_probability'
    assert 'modelProbability' not in row
    assert row['extractionEvaluation']['sourceMatch'] is True
    assert result['finalAccuracy']['sourceGroundedCorrectness'] == 100
    assert result['finalAccuracy']['averageModelConfidence'] == .7515


def test_all_extracted_blocks_are_reviewed_even_with_low_or_unavailable_confidence():
    import json
    from resume_parser.source_evaluation import verify_extraction, concise_evaluation
    result = evaluate_extraction(document('SUMMARY', 'Built software.', 'SKILLS', 'Python', 'WORK EXPERIENCE', 'Engineer Acme'),
                                 {'summary': {'text': 'Built software.'}, 'additional': {'technicalSkills': ['Python']},
                                  'workExperience': [{'title': 'Engineer Acme'}]},
                                 [{'section': 'summary', 'mlConfidence': .75, 'lineIds': ['L1']},
                                  {'section': 'experience', 'mlConfidence': .47, 'lineIds': ['L5']}])
    class Judge:
        payloads = []
        def complete(self, prompt):
            data = json.loads(prompt.split('\n', 1)[1])
            self.payloads.append(data)
            if data['block'] == 'technicalSkills':
                raise TimeoutError('timeout')
            status = 'CORRECT' if data['block'] == 'summary' else 'PARTIAL'
            return json.dumps({'status': status, 'confidence': .98, 'missing': [], 'incorrect': [], 'hallucinated': [], 'reason': 'Checked source.'})
    judge = Judge()
    verify_extraction(result, judge)
    assert {p['block'] for p in judge.payloads} == {'summary', 'technicalSkills', 'workExperience'}
    assert judge.payloads[0]['original'] == ['Built software.']
    assert judge.payloads[1]['original'] == ['Python']
    assert result['llmVerification']['submitted'] == 3
    assert result['llmVerification']['reviewed'] == 2
    assert result['llmVerification']['errors'] == 1
    assert result['finalAccuracy']['llmVerifiedAccuracy'] == 50
    summary = concise_evaluation(result)['evaluationSummary']
    assert summary['llmVerification']['reviewed'] == 2
    assert summary['llmVerification']['verifiedCorrect'] == 1
    assert summary['averageModelConfidence'] == .61


def test_missing_configuration_is_explicit_and_never_fakes_submissions():
    from resume_parser.source_evaluation import verify_extraction
    result = evaluate_extraction(document('SKILLS', 'Python'), {'additional': {'technicalSkills': ['Python']}}, [])
    verify_extraction(result, None, unavailable_reason='Set OPENROUTER_API_KEY in the environment.')
    assert result['llmVerification']['eligible'] == 1
    assert result['llmVerification']['submitted'] == 0
    assert result['llmVerification']['errors'] == 1
    assert result['blockComparison'][0]['llmVerification']['status'] == 'LLM_ERROR'
    assert result['finalAccuracy']['llmVerifiedAccuracy'] is None


def test_heading_correction_uses_actual_assigned_class_probability(monkeypatch):
    import numpy as np
    from resume_parser.source_evaluation import supplement_model_probabilities
    from resume_parser.classification import SectionClassifier
    out = document('PROJECTS', 'Built useful software.')
    span = {'section': 'experience', 'confidence': .8, 'text': 'Built useful software.',
            'feature_context': ['projects portfolio'], 'source_line_ids': ['L1']}
    out['classification'] = {'classifications': [{'semantic_spans': [span]}]}
    detection = {'section': 'experience', 'mlConfidence': .8, 'lineIds': ['L1']}
    class Pipeline:
        classes_ = ['experience', 'projects']
        def predict_proba(self, texts):
            assert texts == ['projects portfolio Built useful software.']
            return np.array([[.8, .2]])
    class Classifier:
        pipeline = Pipeline()
    monkeypatch.setattr(SectionClassifier, 'load', lambda path: Classifier())
    detections = supplement_model_probabilities(out, [detection], 'model.pkl')
    result = evaluate_extraction(out, {'personalProjects': [{'description': ['Built useful software.']}]}, detections)
    row = result['blockComparison'][0]
    assert row['modelConfidence'] == .2  # P(projects), not the winning P(experience).
    assert row['confidenceType'] == 'raw_model_probability'
    assert 'different section' in row['confidenceReason']


def test_main_dispatches_all_six_blocks_through_existing_nvidia_client(monkeypatch, capsys):
    import json
    import urllib.request
    import print_resume_json as cli
    from resume_parser.resolution import NvidiaNIMLLMClient
    out = document('ALICE', 'SUMMARY', 'Built software.', 'SKILLS', 'Python',
                   'WORK EXPERIENCE', 'Acme Ltd : 2020-2021', 'PROJECTS', 'POC: 1', 'Title: Demo', 'Built a tool.',
                   'EDUCATION', 'BSc from College 2013-2017')
    monkeypatch.setattr(cli, 'build_output', lambda *args, **kwargs: out)
    monkeypatch.setenv('NVIDIA_API_KEY', 'test-key')
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    payloads = []
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            answer = {'status': 'CORRECT', 'confidence': .98, 'missing': [], 'incorrect': [], 'hallucinated': [], 'reason': 'Source supported.'}
            return json.dumps({'choices': [{'message': {'content': json.dumps(answer)}}]}).encode()
    def transport(request, timeout):
        data = json.loads(request.data)
        assert data['model'] == NvidiaNIMLLMClient.DEFAULT_MODEL
        payloads.append(json.loads(data['messages'][0]['content'].split('\n', 1)[1]))
        return Response()
    monkeypatch.setattr(urllib.request, 'urlopen', transport)
    assert cli.main(['unused.pdf']) == 0
    result = json.loads(capsys.readouterr().out)
    assert len(payloads) == 6
    assert payloads[1]['original'] == ['Built software.']
    verification = result['evaluation']['evaluationSummary']['llmVerification']
    assert verification['submitted'] == verification['reviewed'] == 6
    assert verification['errors'] == 0
    assert result['evaluation']['evaluationSummary']['llmVerification']['llmVerifiedAccuracy'] == 100


def test_low_classifier_probability_does_not_lower_extraction_evidence_score():
    import json
    from resume_parser.source_evaluation import verify_extraction, concise_evaluation
    result = evaluate_extraction(document('SKILLS', 'Python'), {'additional': {'technicalSkills': ['Python']}},
                                 [{'section': 'skills', 'mlConfidence': .049, 'lineIds': ['L1']}])
    class Judge:
        def complete(self, prompt):
            return json.dumps({'status': 'VERIFIED_CORRECT', 'confidence': .99, 'missing': [], 'incorrect': [], 'hallucinated': [], 'reason': 'Supported.'})
    verify_extraction(result, Judge())
    row = concise_evaluation(result)['blocks'][0]
    assert row['modelConfidence']['value'] == .049
    assert row['extractionEvaluation']['extractionConfidence'] > .98
    assert row['extractionEvaluation']['status'] == 'CORRECT'


def test_real_project_eight_omission_is_partial_with_high_cosine():
    from resume_parser.source_evaluation import concise_evaluation, verify_extraction
    out = document('PROJECTS', 'Project: 7', 'Built useful software.', 'Project: 8', 'Built useful software.')
    result = evaluate_extraction(out, {'personalProjects': [{'title': 'Project: 7', 'description': ['Built useful software.', 'Built useful software.', '8 users']} ]}, [])
    row = result['blockComparison'][0]
    assert 'Project: 8' in row['missingContent']  # The number 8 elsewhere cannot cover an omitted entry.
    assert row['status'] in {'partial', 'incorrect'}
    # A purely source-supported omission is PARTIAL.
    result = evaluate_extraction(out, {'personalProjects': [{'title': 'Project: 7', 'description': ['Built useful software.']} ]}, [])
    assert result['blockComparison'][0]['status'] == 'partial'
    verify_extraction(result, None, unavailable_reason='No key')
    row = concise_evaluation(result)['blocks'][0]['extractionEvaluation']
    assert row['status'] == 'PARTIAL'
    assert row['sourceMatch'] is False
    assert row['extractionConfidence'] < 1


def test_partial_judge_without_grounded_omission_cannot_silently_be_correct():
    from resume_parser.source_evaluation import verify_extraction, concise_evaluation
    result = evaluate_extraction(document('PROJECTS', 'Project: 7', 'Built software.'),
                                 {'personalProjects': [{'title': 'Project: 7', 'description': ['Built software.']}]}, [])
    class Judge:
        def complete(self, prompt):
            return '{"status":"VERIFIED_PARTIAL","confidence":0.98,"missing":["Project: 8"],"incorrect":[],"hallucinated":[],"reason":"Project 8 omitted."}'
    verify_extraction(result, Judge())
    row = concise_evaluation(result)['blocks'][0]
    assert row['extractionEvaluation']['status'] == 'NOT_VERIFIED'
    assert row['extractionEvaluation']['missingContent'] == []  # Never invent source content.
    assert result['llmVerification']['verifiedPartial'] == 1
    assert result['llmVerification']['verificationCoverage'] == 100


def test_llm_coverage_reports_partial_review_and_timeouts_separately():
    import json
    from resume_parser.source_evaluation import verify_extraction, concise_evaluation
    out = document('SUMMARY', 'Built software.', 'SKILLS', 'Python', 'WORK EXPERIENCE', 'Engineer Acme',
                   'EDUCATION', 'BSc College', 'LANGUAGES', 'English', 'AWARDS', 'Prize')
    resume = {'summary': {'text': 'Built software.'}, 'additional': {'technicalSkills': ['Python'], 'languages': ['English'], 'awards': ['Prize']},
              'workExperience': [{'title': 'Engineer Acme'}], 'education': [{'degree': 'BSc College'}]}
    result = evaluate_extraction(out, resume, [])
    class Judge:
        calls = 0
        def complete(self, prompt):
            self.calls += 1
            if self.calls > 3:
                raise TimeoutError('timeout')
            status = 'VERIFIED_PARTIAL' if self.calls == 3 else 'VERIFIED_CORRECT'
            return json.dumps({'status': status, 'confidence': .97, 'missing': [], 'incorrect': [], 'hallucinated': [], 'reason': 'Review.'})
    judge = Judge()
    verify_extraction(result, judge)
    summary = concise_evaluation(result)['evaluationSummary']['llmVerification']
    assert summary['eligible'] == summary['submitted'] == 6
    assert summary['reviewed'] == 3
    assert summary['verifiedCorrect'] == 2
    assert summary['verifiedPartial'] == 1
    assert summary['verifiedIncorrect'] == 0
    assert summary['errors'] == 3
    assert summary['verificationCoverage'] == 50
    assert summary['llmVerifiedAccuracy'] == 66.67
    assert all(b['llmVerification']['status']=='LLM_ERROR' for b in result['blockComparison'][3:])


def test_swapped_project_clients_are_detected_even_when_all_tokens_match():
    from resume_parser.source_evaluation import concise_evaluation
    out = document('PROJECTS', 'Project: 1', 'Client Name: Alpha', 'Project: 2', 'Client Name: Beta')
    result = evaluate_extraction(out, {'personalProjects': [{'title': 'Project: 1', 'client': 'Beta'}, {'title': 'Project: 2', 'client': 'Alpha'}]}, [])
    row = concise_evaluation(result)['blocks'][0]['extractionEvaluation']
    assert row['status'] == 'INCORRECT'
    assert row['wrongBlockContent'] == ['Beta', 'Alpha']
    assert row['hallucinatedContent'] == []
    assert row['sourceMatch'] is False
    assert row['extractionConfidence'] < 1
    assert result['finalAccuracy']['sourceGroundedCorrectness'] < 100


def test_failed_required_semantic_review_keeps_ambiguous_array_not_verified():
    from resume_parser.source_evaluation import verify_extraction, concise_evaluation
    out = document('PROJECTS', 'Title: App', 'Client Name: Alpha', 'Title: Tool', 'Client Name: Beta')
    result = evaluate_extraction(out, {'personalProjects': [{'title': 'App', 'client': 'Beta'}, {'title': 'Tool', 'client': 'Alpha'}]}, [])
    verify_extraction(result, None, unavailable_reason='timeout')
    row = concise_evaluation(result)['blocks'][0]['extractionEvaluation']
    assert row['status'] == 'NOT_VERIFIED'
    assert 'unresolved' in row['evaluationReason']


def test_verified_partial_real_omission_updates_block_and_summary():
    from resume_parser.source_evaluation import verify_extraction, concise_evaluation
    out = document('PROJECTS', 'Project: 7', 'Built software.', 'Project: 8', 'Tested software.')
    result = evaluate_extraction(out, {'personalProjects': [{'title': 'Project: 7', 'description': ['Built software.']}]}, [])
    class Judge:
        def complete(self, prompt):
            return '{"status":"VERIFIED_PARTIAL","confidence":0.97,"missing":["Project: 8"],"incorrect":[],"hallucinated":[],"reason":"Source project 8 is omitted."}'
    verify_extraction(result, Judge())
    output = concise_evaluation(result)
    quality = output['blocks'][0]['extractionEvaluation']
    assert quality['status'] == 'PARTIAL'
    assert quality['sourceMatch'] is False
    assert 'Project: 8' in quality['missingContent']
    assert output['evaluationSummary']['correctBlocks'] == 0
    assert output['evaluationSummary']['partialBlocks'] == 1
    assert output['evaluationSummary']['llmVerification']['verifiedPartial'] == 1
    assert output['evaluationSummary']['llmVerification']['llmVerifiedAccuracy'] == 0
