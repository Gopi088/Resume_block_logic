# Resume Timeline Review — Chrome side-panel extension (MV3)

Recruiter-first review UI over the existing B0–B9 parser. No parser logic was
changed: the panel consumes real backend output only.

## Run

```bash
# 1. Backend (wraps parse_resume.build_output, adds a `review` projection)
.venv/bin/python -m uvicorn backend.server:app --port 8000

# 2. Extension: chrome://extensions → Developer mode → Load unpacked → `extension/`
# 3. Click the toolbar icon (or open the side panel) on any tab.
```

Without the backend running, the panel boots from
`extension/sample_data.json` — real parser output for `my_resumes/A.Bhargava.docx`
(7 events, 1 gap, 11 undated), so every state is inspectable offline. Uploading
a file (`Parse resume`) POSTs it to `/api/parse` when the backend is up.

Regenerate the fixture after parser changes:

```bash
.venv/bin/python -c "
import json, sys; sys.path.insert(0, '.')
from backend.server import build_review_projection
from parse_resume import build_output
out = build_output('my_resumes/A.Bhargava.docx', include_eval=False,
                   include_text=False, blocks_only=False, model_path='model_real_v2.pkl')
out['review'] = build_review_projection(out)
keep = ['document_id','filename','sha256','timeline','entries','entry_dates','final_sections','lines','review','counts']
json.dump({'ok': True, **{k: out[k] for k in keep if k in out}}, open('extension/sample_data.json','w'))"
```

## Data flow (existing contracts preserved)

`parse_resume.build_output` → `backend/server.py:build_review_projection`
(pure re-projection: candidate identity, accuracy, events/gaps/undated with
exact evidence pointers) → `sidepanel.js` renders. Review notes persist in
`chrome.storage.local` under `review:{sha256}` (localStorage fallback outside
the extension). API contracts (`timeline`, `entries`, `entry_dates`,
`final_sections`, `lines`) are untouched.

## Design (per UI research)

- One viewport, ≤420px, single column; secondary info behind expandable items.
- Header → accuracy (compact `28% · Low — verify`) → timeline (most-recent-first,
  `DATE → ROLE → ORG`) → gaps → undated → evidence/review inside each item.
- Hick's law: 3 buttons max per item (expand, View in Resume, Mark Reviewed).
- Confidence shown as High/Medium/Low only — no ML/LLM internals.
- Gaps are dashed, framed as "needs your review", never a verdict on the candidate.
- Review gate: Mark Reviewed is blocked until a non-empty note exists; the note
  is stored with the review. Clearing the note re-opens the item.
- View in Resume: shows the exact source excerpt (date span highlighted, page +
  line IDs) and best-effort scroll/highlight in the open tab; honest fallback
  text when the tab doesn't contain the passage (e.g. PDF viewer).
- States: loading, empty, error/offline-backend, low-confidence hint,
  evidence-unavailable (gaps are derived absences — stated explicitly).
