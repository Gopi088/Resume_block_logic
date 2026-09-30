# Career Timeline — Chrome side-panel extension (MV3)

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
`extension/sample_data.json` — real parser output for `my_resumes/A.Bhargava.docx`,
so every state is inspectable offline.

With the backend up, opening the side panel on a resume tab analyzes that tab
automatically via `POST /api/parse-url {url}` — no upload click needed
(`file://` URLs are read from local disk incl. Windows↔WSL drive mapping,
`http(s)` URLs are downloaded server-side). Uploading via the file picker
remains for DOCX and other files that can't render in a tab. Auto-analysis
only ever replaces the sample view — never a review in progress.

## Preview fixtures (edge-state test data)

Synthetic, review-shaped payloads for previewing states the sample resume
doesn't cover. Regenerate with `.venv/bin/python extension/fixtures/make_fixtures.py`.
Open the panel with `?fixture=N`, e.g. `sidepanel.html?fixture=2`:

| Fixture | State |
|---|---|
| `f1` | Clean resume: 5 high-confidence jobs, one real gap, promotion path |
| `f2` | Problem resume: bullet fragments, year-only dates, zero-length range, missing title |
| `f3` | Very low confidence (7%): manual-check banner, totals hidden |
| `f4` | Empty / unreadable parse with Re-scan link |
| `f5` | Human-checked note (preload storage key `review:fixf5`) |
| `f6` | Anchor edge cases: dangling ids, no anchor, page-only |

## Unit tests (no new dependencies)

```bash
node --test extension/timeline-lib.test.js   # totals, grouping, gaps, dates, anchors
.venv/bin/python -m pytest tests/ -q         # backend contract (unchanged)
```

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
  other-information sections stay collapsed until the recruiter expands them.
- Compact header: name, most recent role, `Extraction confidence N%`, and
  `⚠ N items need review` / `✓ All reviewed`. No blanket verify orders.
- Light chronological rows (date column, strong title, secondary org,
  chevron = expand only) with separators instead of cards. Experience rows
  read as roles; every other section carries a kind tag (Education,
  Certification, Skills, Summary, …) so dated blocks are never mislabeled.
- Expanded rows show org/location/dates-as-written/confidence and the
  View-in-Resume jump. One review note covers the whole resume ("Your review"
  section, mandatory before Mark Reviewed); the saved note stays visible with
  its timestamp and the header flips to `✓ Reviewed`.
- Review notes POST to `/api/notes/{sha}` under a single `resume` item, so
  every recruiter sees them; offline they stay on-device with an honest notice.
  The author identity is auto-stamped (anonymous device id, nothing to type).
- View in Resume highlights the exact passage and nothing else: no panel
  excerpt, no backend page, no new tabs. Scriptable tabs get an in-place
  highlight (verbatim span, then dash/whitespace-normalized block match);
  anything else (e.g. Chrome's PDF viewer, which blocks all extensions) gets
  a same-tab jump to `#page=N:~:text=<exact quote>`, where the quote is built
  from the evidence's own cleaned source-line words (multi-word, so it pins
  the exact portion rather than a loose word). Parse the file that's actually
  open — the button matches against the panel's current resume.
- States: loading, empty, error/offline-backend, low-confidence hint,
  evidence-unavailable (gaps are derived absences — stated explicitly).
