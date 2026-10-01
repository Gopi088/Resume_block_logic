"""Shared product section normalization over the existing parser's source spans."""
import re
from .classification import _SPAN_HEADING_ALIASES

LABEL_TO_CANONICAL = {'contact': 'personalInfo', 'summary': 'summary', 'experience': 'workExperience',
                     'education': 'education', 'skills': 'technicalSkills', 'projects': 'projects',
                     'certifications': 'certifications', 'languages': 'languages', 'awards': 'awards'}
LEGACY_KEYS = {'personalProjects': 'projects', 'certificationsTraining': 'certifications'}
EXTRA_ALIASES = {
    'technicalSkills': ['tech stack', 'technologies', 'tools', 'tools and technologies', 'professional skills', 'technology'],
    'workExperience': ['career history', 'work history', 'employment history'],
    'education': ['academics', 'educational qualifications', 'education and training'],
    'projects': ['selected projects', 'professional projects'],
    'certifications': ['certifications and training', 'professional certifications', 'courses'],
    'awards': ['recognition', 'accomplishments'], 'languages': ['language proficiency', 'language skills'],
}


def heading_key(text):
    text = re.sub(r'^[#*\s]+|[*:\s]+$', '', text).casefold()
    if re.fullmatch(r'(?:[a-z]\s+){3,}[a-z]', text):
        text = text.replace(' ', '')
    text = re.sub(r'[&/]', ' and ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    singular = {'skills': 'skill', 'technologies': 'technology', 'projects': 'project', 'certifications': 'certification',
                'languages': 'language', 'awards': 'award', 'licenses': 'license', 'competencies': 'competency', 'tools': 'tool'}
    return ' '.join(singular.get(word, word) for word in text.split() if word != 'and')


ALIASES = {heading_key(alias): LABEL_TO_CANONICAL[label.value]
           for label, aliases in _SPAN_HEADING_ALIASES.items() if label.value in LABEL_TO_CANONICAL for alias in aliases}
ALIASES.update({heading_key(alias): key for key, aliases in EXTRA_ALIASES.items() for alias in aliases})
NONCANONICAL = {'publications', 'research', 'patents', 'volunteering', 'memberships', 'professional affiliations', 'interests', 'conferences',
                'research impact', 'what i bring', 'references', 'hobbies', 'activities', 'community service'}
TECH = re.compile(r'\b(?:python|sql|t sql|ms sql|etl|elt|r|aws|docker|azure|java|javascript|tableau|powerbi|power bi|excel|airflow|ssis|ssrs|od[i]|linux|git|kubernetes|data modeling|data analysis|business intelligence)\b', re.I)
ACTION = re.compile(r'\b(?:led|managed|reduced|developed|built|delivered|responsible|worked|implemented|designed|published|analyzed|analysed|prepared|maintain|maintained|develop|prepare|use|coordinate|act|track|translated|collaborated|provided|performed|created|resolved|identified|utilized|utilised|generated|cleaned|involved|optimized|optimised|supported|ensured|investigated|configured|gathered|facilitate|defined|leveraged|monitor)\b', re.I)
PHONE = re.compile(r'(?<!\w)(?:\+\d{1,3}[- ]?)?(?:\d[- ()]*){10,13}(?!\d)')
ORG = re.compile(r'\b(?:ltd|limited|inc|incorporated|corporation|llc|pvt|board|college|university|school|institute|company|client)\b', re.I)


def technical_content(lines):
    parts = [re.sub(r'^\s*[•*\-]\s*', '', t).strip() for t in lines if t.strip()]
    if not parts or any(ACTION.search(t) or ORG.search(t) for t in parts):
        return False
    return sum(bool(TECH.search(t)) and len(t.split()) <= 15 for t in parts) / len(parts) >= .6


def normalize_heading(heading, content=(), context=None, model_labels=()):
    key = heading_key(heading)
    if key in ALIASES:
        return ALIASES[key]
    if key in NONCANONICAL:
        return None
    # Unknown titles need both structural evidence and content, never model probability alone.
    if technical_content(content) and (re.search(r'toolkit|toolbox|stack|expertise|proficienc', key) or 'skills' in model_labels):
        return 'technicalSkills'
    return None


def genuine_heading(text, content=(), current=None, model_labels=()):
    title = re.sub(r'^[#*\s]+|[*:\s]+$', '', text).strip()
    if not title or re.match(r'^\s*[•●▪◦\-+]\s+', text) or ORG.search(title) or re.search(r'\d|@|[|.]', title):
        return False
    if heading_key(title) in ALIASES or heading_key(title) in NONCANONICAL:
        return True
    if len(title.split()) <= 8 and not TECH.search(title) and normalize_heading(title, content, context=current, model_labels=model_labels):
        return True
    if text.startswith('#') or (text.startswith('**') and text.endswith('**')):
        return len(title.split()) <= 8 and not TECH.search(title)
    return current is not None and title.isupper() and 2 <= len(title.split()) <= 5 and not TECH.search(title) and bool(content) and any(ACTION.search(t) for t in content[:2])


def normalize_resume(resume):
    """Canonicalize the existing JSON projection without dropping any values."""
    result, extra = {}, []

    def add(key, value):
        canonical = LEGACY_KEYS.get(key, key)
        if canonical not in LABEL_TO_CANONICAL.values():
            canonical = normalize_heading(key, value if isinstance(value, list) else ())
        if canonical is None:
            extra.append({'title': key, 'content': value if isinstance(value, list) else [value]})
        elif canonical not in result:
            result[canonical] = value
        elif isinstance(result[canonical], list) and isinstance(value, list):
            result[canonical] = result[canonical] + value
        elif isinstance(result[canonical], dict) and isinstance(value, dict):
            merged = dict(result[canonical])
            for field, item in value.items():
                if field not in merged:
                    merged[field] = item
                elif merged[field] != item:
                    merged[field] = '\n'.join(map(str, (merged[field], item)))
            result[canonical] = merged
        else:
            result[canonical] = [result[canonical], value]

    for key, value in resume.items():
        if key not in {'additional', 'customSections', 'additionalSections'}:
            add(key, value)
    for key, value in resume.get('additional', {}).items():
        if key not in {'mlConfidence', 'llmVerification'}:
            add(key, value)
    extra.extend(resume.get('additionalSections', []))
    for title, value in resume.get('customSections', {}).items():
        content = [value['text']] if isinstance(value, dict) and value.get('text') else value if isinstance(value, list) else [value]
        canonical = normalize_heading(title, content)
        if canonical:
            add(canonical, content)
        else:
            extra.append({'title': title, 'content': content})
    if extra:
        # Repeated unknown headings also preserve their content in one logical block.
        grouped = {}
        for entry in extra:
            grouped.setdefault(entry['title'], []).extend(entry.get('content', []))
        result['additionalSections'] = [{'title': title, 'content': content} for title, content in grouped.items()]
    return result


def personal_fields(lines):
    """Recognize contact fields independently of PDF line ordering."""
    result = {}
    email = re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+')
    phone = PHONE
    for raw in lines:
        text = re.sub(r'^[#*•\s]+', '', raw).strip()
        found_email = email.search(text)
        if found_email:
            result['email'] = found_email.group()
        found_phone = phone.search(text)
        if found_phone:
            result['phone'] = found_phone.group().strip()
        dob = re.search(r'(?:date\s+of\s+birth|d\.?o\.?b\.?|birth\s+date)\s*[:\-]\s*([^|;]+)', text, re.I)
        if dob:
            result['dateOfBirth'] = dob.group(1).strip()
        explicit = re.search(r'\b(location|address|name|date)\s*:\s*([^|;]+)', text, re.I)
        if explicit and not dob:
            result[explicit.group(1).lower()] = explicit.group(2).strip()
        for part in text.split('|'):
            part = re.sub(r'^(?:location|address)\s*:\s*', '', part.strip(), flags=re.I)
            if ',' in part and not email.search(part) and not phone.search(part) and len(part.split()) <= 8 and not re.search(r'\b(?:experience|worked|with|from|for)\b', part, re.I):
                result['location'] = part
        if 'name' not in result and not re.search(r'\d|@|[:|]', text) and len(text.split()) <= 6 and not re.search(r'\b(summary|contact|personal information)\b', text, re.I):
            result['name'] = text
    return result
