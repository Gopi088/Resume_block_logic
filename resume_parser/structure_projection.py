"""Project classified source spans while enforcing explicit section boundaries."""
import re
from .source_evaluation import source_blocks


def bounded_entries(out):
    from .section_normalization import LABEL_TO_CANONICAL
    labels = {field: label for label, field in LABEL_TO_CANONICAL.items()}
    spans = [(c['block_id'], span) for c in out.get('classification', {}).get('classifications', [])
             for span in c.get('semantic_spans', [])]
    by_line = {lid: (bid, span) for bid, span in spans for lid in span.get('source_line_ids', [])}
    entries = []
    for block in source_blocks(out):
        for text, lid in zip(block['sourceContent'], block['sourceLineIds']):
            bid, span = by_line.get(lid, ('', {}))
            predicted = span.get('section', 'unknown')
            section = labels.get(block['block'], block['block'].split('::', 1)[-1])
            # Explicit headings govern their content. Headingless boundaries
            # are provided by the existing structural/ML span detector.
            probability = span.get('confidence') if predicted == section else None
            entries.append({'entry_id': f'{bid}:{lid}', 'block_id': bid, 'section': section,
                            'ml_section': predicted, 'section_source': 'source_heading' if block['heading'] else 'model',
                            'text': text, 'line_ids': [lid], 'ml_confidence': probability,
                            'start_line_id': lid, 'end_line_id': lid})
    return entries


def project_items(entries, clean):
    """Split numbered projects/POCs and preserve explicitly labelled fields."""
    result, current = [], None
    for entry in entries:
        for raw in entry['text'].splitlines():
            text = clean(raw).strip()
            if not text:
                continue
            marker = re.fullmatch(r'(?:project|poc)\s*[:#-]?\s*\d+(?:\s*[:.\-]\s*.*)?', text, re.I)
            heading = re.match(r'^#{1,6}\s+(.+)$', text)
            if marker or heading:
                current = {'title': heading.group(1) if heading else text, 'description': [], 'line_ids': []}
                result.append(current)
            if current is None:
                current = {'description': [], 'line_ids': []}
                result.append(current)
            current['line_ids'].extend(entry.get('line_ids', []))
            if marker or heading:
                continue
            field = re.match(r'^(title|project name|project|client(?: name)?|role|duration)\s*:\s*(.+)$', text, re.I)
            if field:
                key = {'title': 'title', 'project name': 'title', 'project': 'title', 'client': 'client', 'client name': 'client',
                       'role': 'role', 'duration': 'duration'}[field.group(1).lower()]
                if key == 'title' and current['description']:
                    current = {'description': [], 'line_ids': list(entry.get('line_ids', []))}
                    result.append(current)
                if key == 'title' and current.get('title'):
                    current['name'] = current['title']
                current[key] = field.group(2).strip()
            else:
                current['description'].append(text)
    return [item for item in result if any(item.get(key) for key in ('title', 'client', 'role', 'duration', 'description'))]


def public_resume(obj):
    """Single output boundary: exclude all internal provenance recursively."""
    internal = {'id', 'entry_id', 'line_ids', 'sourceLineIds', 'block_id', 'blockId', 'lineIds',
                'start_line_id', 'end_line_id', 'section_source', 'sectionType', 'items', 'strings'}
    if isinstance(obj, dict):
        return {k: public_resume(v) for k, v in obj.items()
                if k not in internal and v is not None and v != ''}
    if isinstance(obj, list):
        return [public_resume(v) for v in obj]
    return obj
