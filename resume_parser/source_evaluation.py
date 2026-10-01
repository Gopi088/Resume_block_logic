"""Evaluate emitted resume values against source blocks, independently of confidence."""
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

METADATA = {'id', 'entry_id', 'line_ids', 'mlConfidence', 'modelProbability', 'llmVerification'}

def comparable(text, key=None):
    # Field labels are formatting, not content lost by parsing into named fields.
    text = re.sub(r"(?m)^\s*\d+[.)]\s+", "", text)
    text = re.sub(r"(?im)^\s*(?:title|project name|project(?=\s*:\s*[^\d\s])|client(?: name)?|role|duration|ph(?:one)?|email|date|location|address)\s*:\s*", "", text)
    if key == 'personalInfo':
        from .section_normalization import PHONE
        text = PHONE.sub(lambda match: re.sub(r'\D', '', match.group()), text)
        text = re.sub(r'(?i)\b(?:m|mobile|tel|phone|email|date\s+of\s+birth|d\.?o\.?b\.?|birth\s+date|name|location|address)\s*:\s*', '', text)
    if key == 'education':
        text = re.sub(r'\s+from\s+', ' ', text, flags=re.I)
    return text


def tokens(text):
    return re.findall(r'\w+', text.casefold())

def values(obj):
    if isinstance(obj, str):
        return [obj] if obj.strip() else []
    if isinstance(obj, list):
        return [v for item in obj for v in values(item)]
    if isinstance(obj, dict):
        return [v for k, item in obj.items() if k not in METADATA for v in values(item)]
    return []

def source_blocks(out):
    """Normalize genuine source headings; keep content fragments inside their section."""
    from .section_normalization import normalize_heading, genuine_heading, LABEL_TO_CANONICAL, ACTION, TECH, heading_key, ALIASES as HEADING_ALIASES, NONCANONICAL
    from .classification import _EMPLOYMENT_DATE_RE, _ROLE_LINE_RE, _employment_shape
    records = [line for line in out.get('lines', []) if tokens(line.get('display_text', line.get('normalized_text', '')))]
    labels = {lid: span['section'] for cls in out.get('classification', {}).get('classifications', [])
              for span in cls.get('semantic_spans', []) for lid in span.get('source_line_ids', [])}
    texts = [r.get('display_text', r.get('normalized_text', '')).strip() for r in records]
    has_headings = any(genuine_heading(text, texts[i+1:i+6]) for i, text in enumerate(texts))
    blocks, current = [], None
    active_job = False
    for index, (line, text) in enumerate(zip(records, texts)):
        following = texts[index+1:index+9]
        for next_index, candidate in enumerate(following):
            if heading_key(candidate) in HEADING_ALIASES or heading_key(candidate) in NONCANONICAL:
                following = following[:next_index]
                break
        heading = re.sub(r'^[#*\s]+|[*:\s]+$', '', text).strip()
        key = normalize_heading(heading, following, model_labels=[labels.get(r['line_id']) for r in records[index+1:index+9]])
        is_heading = genuine_heading(text, following, current, model_labels=[labels.get(r['line_id']) for r in records[index+1:index+9]])
        # A date/company sequence starts an implicit employment block even after explicit skills.
        date_job = _EMPLOYMENT_DATE_RE.search(text) and any(re.search(r'\bcompany\b|\bpvt\b|\bltd\b', t, re.I) for t in following[:5])
        role_job = active_job and current and current['block'] == 'technicalSkills' and not re.match(r'^[•*\-]', text) and _ROLE_LINE_RE.search(text) and len(text) < 150
        narrative = active_job and current and current['block'] == 'technicalSkills' and (ACTION.search(text) or text.endswith('.') or len(text.split()) > 12 or re.search(r'\b(?:has rolled|report creation|aims to|is based|brings together)\b', text, re.I))
        employment_context = current is None or current['block'] in {'personalInfo', 'summary', 'technicalSkills', 'workExperience'}
        implicit_job = employment_context and (date_job or role_job or narrative or _employment_shape(texts, index))
        skill_fragment = active_job and not is_heading and re.match(r'^\s*[•*\-]\s+', text) and len(text.split()) <= 8 and not ACTION.search(text) and bool(TECH.search(text))
        if skill_fragment:
            key = 'technicalSkills'
            is_heading = False
        elif implicit_job and not is_heading:
            key = 'workExperience'
            active_job = True
        elif is_heading:
            active_job = active_job if key == 'technicalSkills' else key == 'workExperience'
            key = key or 'additionalSections::' + heading
        else:
            key = LABEL_TO_CANONICAL.get(labels.get(line['line_id']), current['block'] if current else 'personalInfo') if not has_headings else current['block'] if current else 'personalInfo'
        if current is None or current['block'] != key or is_heading:
            current = {'block': key, 'heading': heading if is_heading else None, 'sourceContent': [], 'sourceLineIds': []}
            blocks.append(current)
        if is_heading:
            continue
        current['sourceContent'].append(text)
        current['sourceLineIds'].append(line['line_id'])
    return [block for block in blocks if block['sourceContent']]

def similarity(source, extracted):
    if not tokens(source) or not tokens(extracted):
        return 0.0
    matrix = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b').fit_transform([source, extracted])
    return round(float(cosine_similarity(matrix[0], matrix[1])[0, 0]), 4)

def evaluate_extraction(out, resume, detections):
    from .section_normalization import normalize_resume, LABEL_TO_CANONICAL
    resume = normalize_resume(resume)
    original = source_blocks(out)
    # Pool repeated headings only for scoring; retain each original boundary above.
    grouped = {}
    for b in original:
        if b['block'] not in grouped:
            grouped[b['block']] = {**b, 'sourceContent': [], 'sourceLineIds': []}
        grouped[b['block']]['sourceContent'].extend(b['sourceContent'])
        grouped[b['block']]['sourceLineIds'].extend(b['sourceLineIds'])
    emitted = {k: values(v) for k, v in resume.items() if k != 'additionalSections'}
    emitted.update({'additionalSections::' + entry['title']: values(entry.get('content', [])) for entry in resume.get('additionalSections', [])})
    comparisons = []
    covered = total = supported = generated = source_tokens = 0
    all_source = '\n'.join(comparable(t, b['block']) for b in original for t in b['sourceContent'])
    for block in grouped.values():
        key = block['block']
        extracted = emitted.get(key, [])
        source = '\n'.join(comparable(t, key) for t in block['sourceContent'])
        extracted_text = ' '.join(comparable(t, key) for t in extracted)
        sc, ec = Counter(tokens(source)), Counter(tokens(extracted_text))
        overlap = sum((sc & ec).values())
        source_tokens += sum(sc.values())
        block_supported = min(overlap, sum(len(tokens(comparable(t, key))) for t in extracted if phrase_supported(comparable(t, key), source)))
        supported += block_supported
        generated += sum(ec.values())
        project_markers = {entity_marker(v) for item in resume.get('projects', []) for name in ('title', 'name')
                           for v in [item.get(name, '')] if entity_marker(v)} if key == 'projects' else set()
        remaining = ec.copy()
        missing = []
        for text in block['sourceContent']:
            required = Counter(tokens(comparable(text, key)))
            if required - remaining or (key == 'projects' and entity_marker(text) and entity_marker(text) not in project_markers):
                missing.append(text)
            remaining -= required
        available = sc.copy()
        extra = []
        for text in extracted:
            required = Counter(tokens(comparable(text, key)))
            if required - available or not phrase_supported(comparable(text, key), source):
                extra.append(text)
            available -= required
        association_errors = []
        if key == 'projects':
            association_errors, association_missing = project_association_differences(block['sourceContent'], resume.get('projects', []))
            extra = list(dict.fromkeys(extra + association_errors))
            missing = list(dict.fromkeys(missing + association_missing))
            penalty = min(block_supported, sum(len(tokens(comparable(t, key))) for t in association_errors))
            supported -= penalty
            block_supported -= penalty
        covered += len(block['sourceContent']) - len(missing)
        total += len(block['sourceContent'])
        status = 'missing' if not extracted else 'incorrect' if extra else 'partial' if missing else 'correct'
        expected = next((label for label, field in LABEL_TO_CANONICAL.items() if field == key), key)
        contributing = [d for d in detections if set(block['sourceLineIds']).intersection(d.get('lineIds', []))]
        probabilities = [d.get('sectionProbabilities', {}).get(expected,
                         d.get('mlConfidence') if d.get('section') == expected else None)
                         for d in contributing]
        usable = [p for p in probabilities if not isinstance(p, bool) and isinstance(p, (int, float)) and 0 <= p <= 1]
        probability = round(min(usable), 4) if usable and len(usable) == len(contributing) else None
        metadata = out.get('classification', {}).get('model_metadata', {})
        calibrated = metadata.get('calibrationStatus') == 'CALIBRATED' or metadata.get('raw_confidence_summary', {}).get('calibrationStatus') == 'CALIBRATED'
        disagreement = any(d.get('section') != expected for d in contributing)
        reason = ('Model predictions disagree with the source section; no probability for the assigned section is available.' if contributing else 'No contributing model probability is available.') if probability is None else ('The model favored a different section for one or more spans; this is its actual probability for the assigned section.' if disagreement else 'Low model probability for one or more contributing spans.' if probability < .5 else '')
        extracted_structured = resume.get(key, next((e for e in resume.get('additionalSections', []) if 'additionalSections::' + e['title'] == key), []))
        from .structure_projection import public_resume
        line_coverage = (len(block['sourceContent']) - len(missing)) / len(block['sourceContent'])
        field_support = max(0, sum(phrase_supported(comparable(t, key), source) for t in extracted) - len(association_errors)) / len(extracted) if extracted else 0.0
        fields = {}
        if key == 'personalInfo':
            from .section_normalization import personal_fields
            expected_fields = personal_fields(block['sourceContent'])
            actual_fields = resume.get(key, {})
            for field in dict.fromkeys([*expected_fields, *actual_fields]):
                original_value, actual_value = expected_fields.get(field), actual_fields.get(field)
                canonical = lambda value: re.sub(r'\D', '', value) if field == 'phone' else tokens(value)
                field_status = 'missing' if not actual_value else 'incorrect' if not original_value or canonical(str(original_value)) != canonical(str(actual_value)) else 'correct'
                fields[field] = {'status': field_status.upper(), 'source': original_value, 'extracted': actual_value}
        field_errors = [field + ': ' + str(row['extracted']) for field, row in fields.items() if row['status'] == 'INCORRECT']
        if field_errors:
            status = 'incorrect'
        elif any(row['status'] == 'MISSING' for row in fields.values()) and status == 'correct':
            status = 'partial'
        if fields:
            field_support = min(field_support, sum(row['status'] == 'CORRECT' for row in fields.values()) / len(fields))
        comparisons.append({**block, 'fieldComparison': fields, 'status': status, 'deterministicStatus': status,
                            'extractedStructured': public_resume(extracted_structured),
                            'requiresSemanticVerification': key == 'projects' and len(resume.get('projects', [])) > 1 and any(not (entity_marker(item.get('name', '')) or entity_marker(item.get('title', ''))) for item in resume.get('projects', [])),
                            'evidence': {'contentPrecision': block_supported / sum(ec.values()) if ec else 0.0,
                                         'sourceCompleteness': overlap / sum(sc.values()) if sc else 0.0,
                                         'sourceLineCoverage': line_coverage, 'supportedFields': field_support}, 'extractedContent': extracted,
                            'missingContent': missing, 'extraContent': extra, 'associationErrors': association_errors,
                            'incorrectContent': field_errors + [t for t in extra if phrase_supported(comparable(t, key), source) and t not in association_errors],
                            'modelConfidence': probability,
                            'confidenceType': ('calibrated_model_probability' if calibrated else 'raw_model_probability') if probability is not None else 'unavailable',
                            'confidenceReason': reason,
                            'cosineSimilarity': similarity(source, extracted_text),
                            'llmVerification': {'status': 'NOT_VERIFIED', 'confidence': None, 'reason': 'LLM review has not run.'}})
    present = {b['block'] for b in original}
    misplaced = [{'block': b['block'], 'extractedContent': [t for t in b['extraContent'] if phrase_supported(comparable(t, b['block']), all_source) and (t in b['associationErrors'] or not phrase_supported(comparable(t, b['block']), '\n'.join(comparable(v, b['block']) for v in b['sourceContent'])))]} for b in comparisons]
    extra_blocks = [{'block': k, 'extractedContent': v,
                     'wrongBlockContent': [t for t in v if phrase_supported(comparable(t, k), all_source)],
                     'hallucinatedContent': [t for t in v if not phrase_supported(comparable(t, k), all_source)]}
                    for k, v in emitted.items() if v and k not in present]
    # Content retention uses multiset precision/recall F1, not model confidence.
    generated += sum(len(tokens(' '.join(v))) for k, v in emitted.items() if k not in present)
    precision = supported / generated if generated else 0.0
    recall = supported / source_tokens if source_tokens else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    coverage = round(100*covered/total, 2) if total else 0.0
    block_accuracy = round(100*sum(b['status']=='correct' for b in comparisons)/(len(comparisons) + len(extra_blocks)), 2) if comparisons else 0.0
    confs = [b['modelConfidence'] for b in comparisons if b['modelConfidence'] is not None]
    cosine = similarity(all_source, ' '.join(comparable(v, k) for k, vs in emitted.items() for v in vs))
    return {'detectedBlocks': original, 'blockComparison': comparisons,
            'correctBlocks': [b for b in comparisons if b['status']=='correct'],
            'incorrectBlocks': [b for b in comparisons if b['status']=='incorrect'],
            'partialBlocks': [b for b in comparisons if b['status']=='partial'],
            'missingBlocks': [b for b in comparisons if b['status']=='missing'],
            'missingContent': [{'block': b['block'], 'sourceContent': b['missingContent']} for b in comparisons if b['missingContent']],
            'extraBlocks': extra_blocks, 'misplacedContent': [b for b in misplaced if b['extractedContent']], 'extraContent': [{'block': b['block'], 'extractedContent': [t for t in b['extraContent'] if not phrase_supported(comparable(t, b['block']), all_source)]} for b in comparisons if any(not phrase_supported(comparable(t, b['block']), all_source) for t in b['extraContent'])],
            'sourceLineCoverage': coverage, 'blockCoverage': round(100*sum(bool(b['extractedContent']) for b in comparisons)/len(comparisons), 2) if comparisons else 0.0,
            'cosineSimilarity': cosine, 'averageConfidence': round(sum(confs)/len(confs), 4) if confs else None,
            'llmVerification': {'status': 'NOT_VERIFIED'},
            'finalAccuracy': {'sourceGroundedCorrectness': round(100*precision, 2), 'sourceContentRetentionF1': round(100*f1, 2), 'blockAccuracy': block_accuracy, 'contentAccuracy': round(100*precision, 2), 'sourceCoverage': coverage, 'averageCosineSimilarity': round(sum(b['cosineSimilarity'] for b in comparisons)/len(comparisons), 4) if comparisons else 0.0, 'averageModelConfidence': round(sum(confs)/len(confs), 4) if confs else None, 'llmVerifiedAccuracy': None},
            'metricDefinitions': {'sourceContentRetentionF1': 'Source token retention F1 (multiset precision and recall); not semantic or human-labelled accuracy.', 'contentAccuracy': 'Percentage of emitted tokens in source-supported fields within their corresponding block; duplicates consume source occurrences.', 'blockAccuracy': 'Correct source blocks divided by source blocks plus extra extracted sections.', 'sourceLineCoverage': 'Percentage of meaningful source lines whose tokens are all retained within their block.', 'groundTruthScope': 'Converted original resume text; PDF conversion errors are not measured.'}}



def entity_marker(text):
    match = re.match(r'^\s*(?:[#*•]\s*)*(project|poc)\s*[:#-]?\s*(\d+)\b', text, re.I)
    return (match.group(1).lower(), int(match.group(2))) if match else None


def project_association_differences(source_lines, items):
    """Check named project fields within their original numbered source span."""
    units = {}
    current = None
    for text in source_lines:
        marker = entity_marker(text)
        if marker:
            current = marker
            units.setdefault(current, [])
        if current is not None:
            units[current].append(text)
    wrong, missing = [], []
    for item in items:
        marker = entity_marker(item.get('name', '')) or entity_marker(item.get('title', ''))
        if marker not in units:
            continue
        source = '\n'.join(comparable(t, 'projects') for t in units[marker])
        emitted = values(item)
        wrong.extend(t for t in emitted if not phrase_supported(comparable(t, 'projects'), source))
        extracted = ' '.join(comparable(t, 'projects') for t in emitted)
        missing.extend(t for t in units[marker] if not phrase_supported(comparable(t, 'projects'), extracted))
    return wrong, missing


def phrase_supported(text, source):
    phrase = ' '.join(tokens(text))
    return not phrase or (' ' + phrase + ' ') in (' ' + ' '.join(tokens(source)) + ' ')


def verify_extraction(evaluation, client, verify_all=True, unavailable_reason=None):
    """Independently review every extracted source block; never invent submissions."""
    import json
    from .resolution import complete_json
    selected = [b for b in evaluation['blockComparison'] if b['extractedContent']]
    reviewed = []
    submitted = errors = declined = 0
    for block in selected:
        if client is None:
            block['llmVerification'] = {'status': 'LLM_ERROR', 'confidence': None,
                                        'reason': unavailable_reason or 'LLM provider is not configured.'}
            errors += 1
            continue
        prompt = ('Is the extracted content supported by the original content? Check for missing information, '
                  'incorrect information, hallucinated information and wrong-block content. Do not invent or correct content. '
                  'Return ONLY strict JSON {"status":"VERIFIED_CORRECT"|"VERIFIED_PARTIAL"|"VERIFIED_INCORRECT",'
                  '"confidence":number 0..1 or null,"missing":[],"incorrect":[],"hallucinated":[],"reason":"..."}.\n' +
                  json.dumps({'block': 'additionalSections' if block['block'].startswith('additionalSections::') else block['block'], 'original': block['sourceContent'], 'extracted': block['extractedStructured']}, ensure_ascii=False))
        try:
            # A submission counts one logical block request, including its JSON repair retry.
            submitted += 1
            answer = complete_json(client, prompt)
            status = answer.get('status')
            # Accept older strict JSON judge status values for compatibility.
            status = status.removeprefix('VERIFIED_') if isinstance(status, str) else status
            confidence = answer.get('confidence')
            if (status not in {'CORRECT', 'INCORRECT', 'PARTIAL', 'NOT_VERIFIED', 'LLM_ERROR'}
                    or (confidence is not None and (isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1))
                    or (status in {'CORRECT', 'INCORRECT', 'PARTIAL'} and confidence is None)
                    or any(not isinstance(answer.get(k), list) or any(not isinstance(v, str) for v in answer[k]) for k in ('missing', 'incorrect', 'hallucinated'))
                    or not isinstance(answer.get('reason'), str)):
                raise ValueError('Invalid LLM verification schema')
            if status in {'NOT_VERIFIED', 'LLM_ERROR'}:
                block['llmVerification'] = {'status': status, 'confidence': None, 'reason': answer['reason']}
                errors += status == 'LLM_ERROR'
                declined += status == 'NOT_VERIFIED'
                continue
            reason = answer['reason']
            if status == 'CORRECT' and (block['missingContent'] or block['extraContent'] or block['incorrectContent'] or any(answer[k] for k in ('missing', 'incorrect', 'hallucinated'))):
                status = 'INCORRECT' if block['extraContent'] or block['incorrectContent'] or answer['incorrect'] or answer['hallucinated'] else 'PARTIAL'
                reason = 'Deterministic source matching vetoed CORRECT: missing or unsupported content. ' + reason
            block['llmVerification'] = {'status': 'VERIFIED_' + status, 'confidence': confidence, 'reason': reason,
                                        'missing': answer['missing'], 'incorrect': answer['incorrect'], 'hallucinated': answer['hallucinated']}
            reviewed.append(block)
        except Exception as exc:
            errors += 1
            reason = 'timeout' if 'timed out' in str(exc).lower() or isinstance(exc, TimeoutError) else str(exc)[:300]
            block['llmVerification'] = {'status': 'LLM_ERROR', 'confidence': None, 'reason': reason}
    evaluation['llmVerification'] = {
        'status': 'NOT_VERIFIED' if not selected else 'LLM_ERROR' if errors else 'COMPLETE' if len(reviewed) == len(selected) else 'PARTIAL',
        'eligible': len(selected), 'submitted': submitted, 'reviewed': len(reviewed),
        'verifiedCorrect': sum(b['llmVerification']['status'] == 'VERIFIED_CORRECT' for b in reviewed),
        'verifiedPartial': sum(b['llmVerification']['status'] == 'VERIFIED_PARTIAL' for b in reviewed),
        'verifiedIncorrect': sum(b['llmVerification']['status'] == 'VERIFIED_INCORRECT' for b in reviewed),
        'errors': errors, 'notVerified': declined,
        'verificationCoverage': round(100 * len(reviewed) / len(selected), 2) if selected else 0.0}
    for block in evaluation['blockComparison']:
        reconcile_block(block)
    if unavailable_reason:
        evaluation['llmVerification']['reason'] = unavailable_reason
    evaluation['finalAccuracy']['llmVerifiedAccuracy'] = round(100*sum(b['llmVerification']['status']=='VERIFIED_CORRECT' for b in reviewed)/len(reviewed), 2) if reviewed else None


def reconcile_block(block):
    """Reconcile successful judge findings without changing extracted content."""
    judge = block['llmVerification']
    status = judge['status']
    base = block['deterministicStatus']
    if base in {'incorrect', 'missing'}:
        block['status'] = base
        return
    if status == 'VERIFIED_INCORRECT':
        block['status'] = 'incorrect'
    elif status == 'VERIFIED_PARTIAL':
        source = '\n'.join(block['sourceContent'])
        extracted = ' '.join(block['extractedContent'])
        grounded = []
        for claim in judge.get('missing', []):
            marker = entity_marker(claim)
            if marker:
                matches = [t for t in block['sourceContent'] if entity_marker(t) == marker]
                if matches and not any(entity_marker(v) == marker for v in block['extractedContent']):
                    grounded.extend(matches)
            elif phrase_supported(comparable(claim, block['block']), source) and not phrase_supported(comparable(claim, block['block']), extracted):
                grounded.append(claim)
        block['missingContent'] = list(dict.fromkeys(block['missingContent'] + grounded))
        # Unsubstantiated partial findings remain unresolved, never silently CORRECT.
        block['status'] = 'partial' if block['missingContent'] else 'not_verified'
    elif status in {'LLM_ERROR', 'NOT_VERIFIED'}:
        # Exhaustive exact source matching can still establish deterministic correctness.
        block['status'] = base if base != 'correct' or (not block['requiresSemanticVerification'] and all(v == 1 for v in block['evidence'].values())) else 'not_verified'
    else:
        block['status'] = base


def extraction_confidence(block):
    """An uncalibrated evidence score, not a probability or model accuracy."""
    import math
    if block['status'] == 'not_verified':
        return None
    factors = list(block['evidence'].values()) + [block['cosineSimilarity']]
    base = math.prod(max(0.0, min(1.0, v)) for v in factors) ** (1 / len(factors))
    judge = block['llmVerification']
    if judge['status'] in {'VERIFIED_CORRECT', 'VERIFIED_PARTIAL', 'VERIFIED_INCORRECT'}:
        confidence = judge['confidence']
        if judge['status'] == 'VERIFIED_CORRECT':
            # A successful independent review contributes its certainty as another factor.
            base = (base ** len(factors) * confidence) ** (1 / (len(factors) + 1))
        else:
            # A confident adverse review is evidence against complete correctness.
            base *= 1 - confidence
    return round(base, 4)


def concise_evaluation(evaluation):
    """One concise evaluation per real block, with separated diagnostic metrics."""
    blocks = []
    wrong = {b['block']: b['extractedContent'] for b in evaluation['misplacedContent']}
    hallucinated = {b['block']: b['extractedContent'] for b in evaluation['extraContent']}
    for row in evaluation['blockComparison']:
        reconcile_block(row)
        quality = {'status': row['status'].upper(), 'extractionConfidence': extraction_confidence(row),
                   'cosineSimilarity': row['cosineSimilarity'], 'sourceMatch': row['deterministicStatus'] == 'correct' and not row['missingContent'],
                   'sourceCoverage': round(100 * row['evidence']['sourceLineCoverage'], 2),
                   'missingContent': row['missingContent'], 'incorrectContent': list(dict.fromkeys(row['incorrectContent'] + row['llmVerification'].get('incorrect', []))),
                   'wrongBlockContent': wrong.get(row['block'], []), 'hallucinatedContent': hallucinated.get(row['block'], [])}
        block = {'block': 'additionalSections' if row['block'].startswith('additionalSections::') else row['block'], 'modelConfidence': {'value': row['modelConfidence'], 'type': row['confidenceType']}, 'extractionEvaluation': quality,
                 'llmVerification': {'missing': [], 'incorrect': [], 'hallucinated': [], **row['llmVerification']}}
        quality['confidenceType'] = 'evidence_based'
        if row['block'].startswith('additionalSections::'):
            block['title'] = row['block'].split('::', 1)[1]
        if row.get('fieldComparison'):
            quality['fields'] = row['fieldComparison']
        if row['status'] == 'not_verified':
            quality['evaluationReason'] = 'Independent verification is unresolved; deterministic source matching cannot establish full correctness for this block.'
        if row['confidenceReason']:
            block['confidenceReason'] = row['confidenceReason']
        blocks.append(block)
    for extra in evaluation['extraBlocks']:
        blocks.append({'block': 'additionalSections' if extra['block'].startswith('additionalSections::') else extra['block'], **({'title': extra['block'].split('::', 1)[1]} if extra['block'].startswith('additionalSections::') else {}), 'modelConfidence': {'value': None, 'type': 'unavailable'},
                       'extractionEvaluation': {'status': 'INCORRECT', 'extractionConfidence': 0.0, 'confidenceType': 'evidence_based', 'sourceCoverage': 0.0, 'incorrectContent': [], 'cosineSimilarity': None,
                                               'sourceMatch': False, 'missingContent': [], 'wrongBlockContent': extra['wrongBlockContent'],
                                               'hallucinatedContent': extra['hallucinatedContent']},
                       'llmVerification': {'status': 'NOT_VERIFIED', 'confidence': None, 'reason': 'No corresponding original block.', 'missing': [], 'incorrect': [], 'hallucinated': []}})
    real = blocks[:len(evaluation['blockComparison'])]
    extraction_scores = [b['extractionEvaluation']['extractionConfidence'] for b in real if b['extractionEvaluation']['extractionConfidence'] is not None]
    summary = {'totalBlocks': len(real),
               **{name: sum(b['extractionEvaluation']['status'] == status for b in blocks)
                  for name, status in [('correctBlocks', 'CORRECT'), ('partialBlocks', 'PARTIAL'), ('incorrectBlocks', 'INCORRECT'), ('missingBlocks', 'MISSING'), ('notVerifiedBlocks', 'NOT_VERIFIED')]},
               'averageModelConfidence': evaluation['finalAccuracy']['averageModelConfidence'],
               'averageExtractionConfidence': round(sum(extraction_scores) / len(extraction_scores), 4) if extraction_scores else None,
               'averageCosineSimilarity': evaluation['finalAccuracy']['averageCosineSimilarity'],
               'sourceGroundedCorrectness': evaluation['finalAccuracy']['sourceGroundedCorrectness'],
               'sourceContentRetentionF1': evaluation['finalAccuracy']['sourceContentRetentionF1'],
               'sourceCoverage': evaluation['finalAccuracy']['sourceCoverage'],
               'llmVerification': {**evaluation['llmVerification'], 'llmVerifiedAccuracy': evaluation['finalAccuracy']['llmVerifiedAccuracy']}}
    return {'blocks': blocks, 'missingBlocks': list(dict.fromkeys('additionalSections' if b['block'].startswith('additionalSections::') else b['block'] for b in evaluation['missingBlocks'])),
            'cosineSimilarity': evaluation['cosineSimilarity'], 'evaluationSummary': summary,
            'metricDefinitions': {
                **{k: v for k, v in evaluation['metricDefinitions'].items() if k in {'sourceContentRetentionF1', 'groundTruthScope'}},
                'sourceCoverage': 'Percentage of meaningful original source lines completely retained in the corresponding block.',
                'modelConfidence': 'Minimum actual classifier probability across contributing source spans for the assigned section. Not confidence in final JSON correctness.',
                'extractionConfidence': 'Uncalibrated evidence score: geometric mean of source-supported token precision, source token completeness, complete-line coverage, supported-field fraction and TF-IDF cosine. A successful correct LLM review adds its confidence as a sixth factor; partial/incorrect review multiplies the base score by (1 - LLM confidence). Errors add no positive evidence. Unresolved verification returns null; averages exclude unavailable scores. This is not a statistically calibrated probability or ML accuracy.',
                'sourceGroundedCorrectness': 'Percentage of extracted tokens supported by their corresponding original block; omissions are reported separately by completeness, source coverage and retention F1. Not ML model accuracy.',
                'sourceMatch': 'Deterministic comparison found no missing, duplicated, wrong-block or hallucinated content; independent judge findings are reported separately.',
                'cosineSimilarity': 'TF-IDF text similarity between corresponding source and extracted blocks; not extraction accuracy, semantic proof or LLM verification.',
                'llmVerifiedAccuracy': 'Successfully reviewed VERIFIED_CORRECT blocks / successfully reviewed blocks * 100; excludes errors and NOT_VERIFIED.',
                'verificationCoverage': 'Successfully reviewed blocks / eligible blocks * 100.'}}


def supplement_model_probabilities(out, detections, model_path):
    """Recover actual class probabilities for heading-corrected spans only.

    Reproduce the original B3 input, and require its recorded winning
    probability to agree before using any other class probability.
    This scores extraction without changing a label or resume field.
    """
    from .classification import SectionClassifier
    spans = [span for cls in out.get('classification', {}).get('classifications', [])
             for span in cls.get('semantic_spans', [])]
    if not spans:
        return detections
    classifier = SectionClassifier.load(model_path)
    pipeline = classifier.pipeline
    if not callable(getattr(pipeline, 'predict_proba', None)):
        return detections
    inputs = [(' '.join(span.get('feature_context', [])) + ' ' if span.get('feature_context') else '') + span.get('text', '')
              for span in spans]
    probabilities = pipeline.predict_proba(inputs)
    enriched = []
    for detection, span, row in zip(detections, spans, probabilities):
        result = dict(detection)
        recorded = span.get('confidence')
        if isinstance(recorded, (int, float)) and abs(float(max(row)) - recorded) < 1e-6:
            result['sectionProbabilities'] = {str(label): float(probability) for label, probability in zip(pipeline.classes_, row)}
        enriched.append(result)
    return enriched
