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

- One viewport, ≤420px, single column; career gaps surface first, timeline and
  undated sections stay collapsed until the recruiter expands them.
- Header shows name + career span only (no role/org description); accuracy is a
  compact `28% · Low — verify`.
- Timeline rows are one line each (`dates + title`); org appears only on expand,
  and only when it differs from the title. No per-item statuses or buttons
  beyond Expand and View in Resume.
- Confidence shown as High/Medium/Low only — no ML/LLM internals.
- Gaps are dashed, framed as "needs your review", never a verdict on the candidate.
- One review note per resume ("Your review" section): empty notes are blocked;
  saving stamps the time automatically (anonymous device id, nothing to type).
  Notes POST to `/api/notes/{sha}` (item `resume`, kept in
  `backend/notes_store.json`) so every recruiter sees them; offline they stay
  on-device with an honest notice.
- View in Resume highlights the exact passage and nothing else: no panel
  excerpt, no backend page, no new tabs. Scriptable tabs get an in-place
  highlight (verbatim span, then dash/whitespace-normalized block match);
  anything else (e.g. Chrome's PDF viewer, which blocks all extensions) gets
  a same-tab jump via a `#:~:text=` fragment. Parse the file that's actually
  open — the button matches against the panel's current resume.
- States: loading, empty, error/offline-backend, low-confidence hint,
  evidence-unavailable (gaps are derived absences — stated explicitly).
