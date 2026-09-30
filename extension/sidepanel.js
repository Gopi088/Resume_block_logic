/* Recruiter review panel logic. Consumes the backend /api/parse contract
   (or the bundled real-parser fixture offline). No mock data is constructed:
   every rendered fact comes from parser fields; missing data renders as an
   explicit unavailable state. */
(function () {
  "use strict";

  var BACKEND = "http://localhost:8000";
  var store = storage();

  var state = {
    doc: null,      // full parser payload (fixture or backend response)
    review: null,   // review projection
    notes: {},      // itemId -> [{note, by, at}] (shared across recruiters)
    reviewer: "",
    backendUp: false,
  };

  var el = {
    loading: document.getElementById("stateLoading"),
    error: document.getElementById("stateError"),
    errorMsg: document.getElementById("errorMsg"),
    retry: document.getElementById("retryBtn"),
    review: document.getElementById("review"),
    empty: document.getElementById("stateEmpty"),
    file: document.getElementById("fileInput"),
    source: document.getElementById("sourceLabel"),
    name: document.getElementById("candName"),
    meta: document.getElementById("candMeta"),
    accPct: document.getElementById("accPct"),
    accLabel: document.getElementById("accLabel"),
    accHint: document.getElementById("accHint"),
    progressBar: document.getElementById("progressBar"),
    progressText: document.getElementById("progressText"),
    reviewerName: document.getElementById("reviewerName"),
    timelineToggle: document.getElementById("timelineToggle"),
    timelineChev: document.getElementById("timelineChev"),
    timeline: document.getElementById("timeline"),
    timelineCount: document.getElementById("timelineCount"),
    gapSection: document.getElementById("gapSection"),
    gaps: document.getElementById("gaps"),
    gapCount: document.getElementById("gapCount"),
    undatedSection: document.getElementById("undatedSection"),
    undatedToggle: document.getElementById("undatedToggle"),
    undatedChev: document.getElementById("undatedChev"),
    undated: document.getElementById("undated"),
    undatedCount: document.getElementById("undatedCount"),
    evidenceView: document.getElementById("evidenceView"),
    evidenceBack: document.getElementById("evidenceBack"),
    evidenceTitle: document.getElementById("evidenceTitle"),
    evidenceSub: document.getElementById("evidenceSub"),
    evidenceDoc: document.getElementById("evidenceDoc"),
  };

  el.retry.addEventListener("click", function () { loadFixture(); });
  el.file.addEventListener("change", onFile);
  el.reviewerName.addEventListener("input", function () {
    state.reviewer = el.reviewerName.value.trim();
    store.set("reviewerName", state.reviewer);
  });
  el.timelineToggle.addEventListener("click", function () {
    toggleSection(el.timeline, el.timelineToggle, el.timelineChev);
  });
  el.undatedToggle.addEventListener("click", function () {
    toggleSection(el.undated, el.undatedToggle, el.undatedChev);
  });
  el.evidenceBack.addEventListener("click", closeEvidenceView);

  store.get("reviewerName", function (v) {
    state.reviewer = v || "";
    el.reviewerName.value = state.reviewer;
  });
  loadFixture();
  pingBackend();

  function toggleSection(body, toggle, chev) {
    var open = body.hidden;
    body.hidden = !open;
    toggle.setAttribute("aria-expanded", open ? "true" : "false");
    chev.textContent = open ? "▾" : "▸";
  }

  /* ---------------- data loading ---------------- */

  function loadFixture() {
    showLoading(true);
    fetch("sample_data.json")
      .then(function (r) { if (!r.ok) throw new Error("fixture missing"); return r.json(); })
      .then(function (doc) {
        el.source.textContent = "Sample resume (real parser output)";
        setDocument(doc);
      })
      .catch(function (err) { showError("Sample data could not be loaded: " + err.message); });
  }

  function pingBackend() {
    fetch(BACKEND + "/health").then(function () { state.backendUp = true; }).catch(function () {});
  }

  function onFile(e) {
    var f = e.target.files && e.target.files[0];
    if (!f) return;
    showLoading(true);
    el.source.textContent = "Parsing " + f.name + "…";
    var fd = new FormData();
    fd.append("file", f, f.name);
    fetch(BACKEND + "/api/parse", { method: "POST", body: fd })
      .then(function (r) { if (!r.ok) throw new Error("backend " + r.status); return r.json(); })
      .then(function (doc) {
        state.backendUp = true;
        el.source.textContent = f.name;
        setDocument(doc);
      })
      .catch(function () {
        showError("The local review backend isn’t reachable at " + BACKEND +
          ". Start it with: .venv/bin/python -m uvicorn backend.server:app --port 8000");
      });
    e.target.value = "";
  }

  function setDocument(doc) {
    state.doc = doc;
    state.review = doc.review || null;
    if (!state.review) { showError("This payload has no review projection."); return; }
    loadNotes(function () {
      render();
      showLoading(false);
    });
  }

  /* ---------------- shared review notes ---------------- */

  function storageKey() {
    var sha = (state.doc && state.doc.sha256) || (state.review.candidate.sha256) || "unknown";
    return "review:" + sha;
  }

  function storage() {
    if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
      return {
        get: function (k, cb) { chrome.storage.local.get(k, function (o) { cb(o[k]); }); },
        set: function (k, v, cb) {
          var o = {}; o[k] = v; chrome.storage.local.set(o, function () { cb && cb(); });
        }
      };
    }
    return {
      get: function (k, cb) {
        try { cb(JSON.parse(localStorage.getItem(k) || "null")); }
        catch (e) { cb(null); }
      },
      set: function (k, v, cb) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) {} cb && cb(); }
    };
  }

  function normalizeNotes(raw) {
    // Migrate the old single-note shape {note, at} to the shared list shape.
    var out = {};
    Object.keys(raw || {}).forEach(function (id) {
      var v = raw[id];
      if (Array.isArray(v)) out[id] = v.filter(function (n) { return n && n.note; });
      else if (v && v.note) out[id] = [{ note: v.note, by: v.by || "Recruiter", at: v.at || null }];
    });
    return out;
  }

  function mergeNotes(a, b) {
    var out = {};
    Object.keys(a || {}).concat(Object.keys(b || {})).forEach(function (id) {
      var seen = {};
      out[id] = (a[id] || []).concat(b[id] || []).filter(function (n) {
        var k = (n.note || "") + "|" + (n.by || "") + "|" + (n.at || "");
        if (seen[k]) return false;
        seen[k] = true;
        return !!n.note;
      });
    });
    return out;
  }

  function loadNotes(cb) {
    store.get(storageKey(), function (local) {
      var merged = normalizeNotes(local);
      fetch(BACKEND + "/api/notes/" + docSha())
        .then(function (r) { if (!r.ok) throw new Error("no backend"); return r.json(); })
        .then(function (body) {
          state.backendUp = true;
          state.notes = mergeNotes(merged, normalizeNotes((body || {}).notes));
          cb();
        })
        .catch(function () { state.notes = merged; cb(); });
    });
  }

  function docSha() {
    return (state.doc && state.doc.sha256) || (state.review.candidate.sha256) || "unknown";
  }

  function saveNotes() { store.set(storageKey(), state.notes); }

  function pushNote(it, noteText, cb) {
    var entry = { note: noteText, by: state.reviewer, at: new Date().toISOString() };
    var id = itemId(it);
    state.notes[id] = (state.notes[id] || []).concat([entry]);
    saveNotes();
    // Share with other recruiters through the backend; local copy keeps
    // working offline.
    fetch(BACKEND + "/api/notes/" + docSha(), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ item_id: id, note: noteText, by: state.reviewer }),
    }).then(function () { state.backendUp = true; cb(true); })
      .catch(function () { cb(false); });
  }

  function itemId(it) {
    return it.kind === "gap" ? it.gap_id : it.entry_id;
  }
  function itemNotes(it) {
    return state.notes[itemId(it)] || [];
  }
  function isReviewed(it) {
    return itemNotes(it).length > 0;
  }

  /* ---------------- rendering ---------------- */

  function render() {
    var r = state.review;
    el.error.hidden = true;
    el.review.hidden = false;
    el.evidenceView.hidden = true;

    // Candidate header: name + career span only. No role/org description.
    el.name.textContent = r.candidate.name || "Unnamed candidate";
    el.meta.textContent = r.candidate.years_experience != null
      ? r.candidate.years_experience + " yrs span" : "";

    el.accPct.textContent = r.accuracy.percent + "%";
    el.accLabel.textContent = r.accuracy.label;
    if (r.accuracy.percent < 45) {
      el.accHint.hidden = false;
      el.accHint.textContent = "Low extraction confidence — verify each item against the resume.";
    } else { el.accHint.hidden = true; }

    // Gaps first (the alert), shown only when found.
    el.gaps.innerHTML = "";
    var gaps = r.gaps || [];
    el.gapSection.hidden = gaps.length === 0;
    el.gapCount.textContent = gaps.length ? "· " + gaps.length + " to check" : "";
    gaps.forEach(function (g) { el.gaps.appendChild(renderGap(g)); });

    // Timeline: most-recent-first, collapsed until wanted.
    var events = (r.events || []).slice().sort(function (a, b) {
      if (a.start_date === b.start_date) return 0;
      return (a.start_date || "") < (b.start_date || "") ? 1 : -1;
    });
    el.timeline.innerHTML = "";
    el.timelineCount.textContent = events.length ? "· " + events.length + " roles" : "";
    events.forEach(function (ev) { el.timeline.appendChild(renderEvent(ev)); });

    // Undated: collapsed until wanted.
    el.undated.innerHTML = "";
    var und = r.undated || [];
    el.undatedSection.hidden = und.length === 0;
    el.undatedCount.textContent = und.length ? "· " + und.length : "";
    und.forEach(function (u) { el.undated.appendChild(renderUndated(u)); });

    el.empty.hidden = events.length > 0 || gaps.length > 0;
    updateProgress();
  }

  function reviewableCount() {
    var r = state.review;
    return (r.events || []).length + (r.gaps || []).length + (r.undated || []).length;
  }
  function reviewedCount() {
    var r = state.review;
    var all = (r.events || []).concat(r.gaps || [], r.undated || []);
    return all.filter(isReviewed).length;
  }
  function updateProgress() {
    var total = reviewableCount(), done = reviewedCount();
    el.progressBar.style.width = total ? Math.round(100 * done / total) + "%" : "0";
    el.progressText.textContent = total ? done + " of " + total + " reviewed" : "Nothing to review";
  }

  /* ---------------- compact rows: dates + title + status icon ---------------- */

  function rowHead(it, datesLabel, titleText) {
    var head = document.createElement("button");
    head.className = "item-head";
    head.setAttribute("aria-expanded", "false");
    head.title = titleText || "";
    var st = document.createElement("span");
    st.className = "item-status " + (isReviewed(it) ? "status-done" : "status-review");
    st.textContent = isReviewed(it) ? "✓" : "!";
    st.title = isReviewed(it) ? "Reviewed" : "Needs review";
    head.innerHTML =
      '<span class="item-dates">' + esc(datesLabel) + "</span>" +
      '<span class="item-title">' + esc(titleText || "—") + "</span>";
    head.appendChild(st);
    var chev = document.createElement("span");
    chev.className = "chev";
    chev.textContent = "▸";
    head.appendChild(chev);
    return { head: head, status: st, chev: chev };
  }

  function bindExpand(container, head, chev) {
    head.addEventListener("click", function () {
      var open = container.classList.toggle("open");
      head.setAttribute("aria-expanded", open ? "true" : "false");
      chev.textContent = open ? "▾" : "▸";
    });
  }

  function renderEvent(ev) {
    var li = document.createElement("li");
    li.className = "item" + (isReviewed(ev) ? " reviewed" : " needs-review");
    var parts = rowHead(ev, fmtRange(ev.start_date, ev.end_date, ev.is_ongoing), ev.title);
    bindExpand(li, parts.head, parts.chev);

    var body = document.createElement("div");
    body.className = "item-body";
    if (ev.organization && !sameText(ev.organization, ev.title)) {
      body.appendChild(detail("Organization", ev.organization));
    }
    if (ev.location) body.appendChild(detail("Location", ev.location));
    body.appendChild(detail("Detected from", sectionName(ev.section)));
    body.appendChild(detail("Confidence", ev.confidence_band + (ev.trusted ? "" : " — verify against resume")));
    if (ev.raw_range) body.appendChild(detail("Dates as written", ev.raw_range));
    body.appendChild(viewResumeButton(ev));
    body.appendChild(reviewBlock(ev, li));

    li.appendChild(parts.head);
    li.appendChild(body);
    return li;
  }

  function renderGap(g) {
    var div = document.createElement("div");
    div.className = "item gap" + (isReviewed(g) ? " reviewed" : " needs-review");
    var months = g.gap_months_approx != null ? " (" + g.gap_months_approx + " months)" : "";
    var parts = rowHead(g, fmtRange(g.start_date, g.end_date, false), "Potential career gap" + months);
    parts.head.title = "No employment stated here — your call.";
    bindExpand(div, parts.head, parts.chev);

    var body = document.createElement("div");
    body.className = "item-body";
    body.appendChild(detail("Why flagged", "No employment covers this stretch (breaks under 90 days are ignored)."));
    var ev = document.createElement("div");
    ev.className = "evidence";
    ev.innerHTML = '<span class="muted">A gap is an absence — there is no resume passage to show. ' +
      "Compare the roles around it in the timeline.</span>";
    body.appendChild(ev);
    body.appendChild(reviewBlock(g, div));

    div.appendChild(parts.head);
    div.appendChild(body);
    return div;
  }

  function renderUndated(u) {
    var div = document.createElement("div");
    div.className = "item" + (isReviewed(u) ? " reviewed" : " needs-review");
    var parts = rowHead(u, "No dates", u.title || ("Entry " + (u.entry_id || "")));
    bindExpand(div, parts.head, parts.chev);

    var body = document.createElement("div");
    body.className = "item-body";
    if (u.organization && !sameText(u.organization, u.title)) {
      body.appendChild(detail("Organization", u.organization));
    }
    body.appendChild(detail("Detected from", sectionName(u.section)));
    body.appendChild(detail("Confidence", u.confidence_band + " — verify against resume"));
    body.appendChild(viewResumeButton(u));
    body.appendChild(reviewBlock(u, div));

    div.appendChild(parts.head);
    div.appendChild(body);
    return div;
  }

  function sameText(a, b) {
    function norm(s) { return (s || "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim(); }
    var na = norm(a), nb = norm(b);
    return !!na && !!nb && (na === nb || na.indexOf(nb) === 0 || nb.indexOf(na) === 0);
  }

  function detail(k, v) {
    var p = document.createElement("p");
    p.className = "detail-row";
    p.innerHTML = '<span class="k">' + esc(k) + ":</span> " + esc(v);
    return p;
  }

  /* ---------------- View in Resume: straight to the highlighted text ---------------- */

  function viewResumeButton(it) {
    var btn = document.createElement("button");
    btn.className = "btn";
    btn.textContent = "View in Resume";
    btn.title = "Open the exact resume passage, highlighted";
    if (!it.evidence || !it.evidence.excerpt) btn.disabled = true;
    btn.addEventListener("click", function () { openEvidenceView(it); });
    var wrap = document.createElement("div");
    wrap.className = "actions evidence-go";
    wrap.appendChild(btn);
    return wrap;
  }

  function openEvidenceView(it) {
    var ev = it.evidence || {};
    el.evidenceTitle.textContent = it.title || (it.kind === "gap" ? "Potential career gap" : "Entry");
    el.evidenceSub.textContent = it.kind === "event"
      ? fmtRange(it.start_date, it.end_date, it.is_ongoing) + " · " + sectionName(it.section)
      : (it.kind === "gap" ? fmtRange(it.start_date, it.end_date, false) : sectionName(it.section));

    // Rebuild the resume from its exact source lines; highlight this item's lines.
    el.evidenceDoc.innerHTML = "";
    var hits = {};
    (ev.line_ids || []).forEach(function (id) { hits[id] = true; });
    var lines = ((state.doc && state.doc.lines) || []).slice().sort(function (a, b) {
      return (a.index || 0) - (b.index || 0);
    });
    var lastPage = null, firstHit = null;
    lines.forEach(function (ln) {
      if (ln.page_index !== lastPage) {
        lastPage = ln.page_index;
        var pg = document.createElement("div");
        pg.className = "epage";
        pg.textContent = "Page " + (lastPage + 1);
        el.evidenceDoc.appendChild(pg);
      }
      var text = ln.display_text || "";
      if (!text.trim()) return;
      var div = document.createElement("div");
      div.className = "eline" + (hits[ln.line_id] ? " hit" : "");
      div.id = "ev-" + ln.line_id;
      if (ev.raw_range && hits[ln.line_id]) {
        div.innerHTML = esc(text).replace(esc(ev.raw_range), "<mark>" + esc(ev.raw_range) + "</mark>");
      } else {
        div.textContent = text;
      }
      el.evidenceDoc.appendChild(div);
      if (hits[ln.line_id] && !firstHit) firstHit = div;
    });

    el.review.hidden = true;
    el.evidenceView.hidden = false;
    if (firstHit && firstHit.scrollIntoView) firstHit.scrollIntoView({ block: "center" });

    // Best effort: also highlight in the open resume tab.
    var texts = [];
    if (ev.raw_range) texts.push(ev.raw_range);
    var firstLine = (ev.excerpt || "").split("\n").filter(function (l) { return l.trim(); })[0];
    if (firstLine) texts.push(firstLine.trim().slice(0, 120));
    if (texts.length) sendHighlight(texts, function () {});
  }

  function closeEvidenceView() {
    el.evidenceView.hidden = true;
    el.review.hidden = false;
  }

  function sendHighlight(texts, cb) {
    try {
      if (typeof chrome === "undefined" || !chrome.tabs) return cb(false);
      chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
        if (!tabs || !tabs[0] || !tabs[0].id) return cb(false);
        chrome.tabs.sendMessage(tabs[0].id, { type: "HIGHLIGHT_RESUME_EVIDENCE", texts: texts }, function (resp) {
          if (chrome.runtime.lastError) return cb(false);
          cb(!!(resp && resp.found));
        });
      });
    } catch (e) { cb(false); }
  }

  /* ---------------- review: mandatory note + name, shared with all recruiters ---------------- */

  function reviewBlock(it, container) {
    var wrap = document.createElement("div");

    var list = document.createElement("ul");
    list.className = "saved-notes";
    renderSavedNotes(list, it);

    var label = document.createElement("label");
    label.className = "note-label";
    label.textContent = "Review note";
    label.htmlFor = "note-" + itemId(it);

    var ta = document.createElement("textarea");
    ta.className = "note";
    ta.id = "note-" + itemId(it);
    ta.placeholder = "e.g. Verified against resume p.1 — dates match.";

    var err = document.createElement("p");
    err.className = "note-error";
    err.hidden = true;

    var actions = document.createElement("div");
    actions.className = "actions";
    var btn = document.createElement("button");
    syncReviewBtn(btn, it);
    btn.addEventListener("click", function () {
      var note = ta.value.trim();
      // Hard gates: reviewer name (so others know who reviewed) + note.
      if (!state.reviewer) {
        err.hidden = false;
        err.textContent = "Add your name under “Reviewing as” so other recruiters know who reviewed.";
        el.reviewerName.focus();
        return;
      }
      if (!note) {
        err.hidden = false;
        err.textContent = "Add a review note before marking this item as reviewed.";
        ta.focus();
        return;
      }
      err.hidden = true;
      btn.disabled = true;
      pushNote(it, note, function (shared) {
        btn.disabled = false;
        ta.value = "";
        renderSavedNotes(list, it);
        syncReviewBtn(btn, it);
        refreshStatus(container, it);
        updateProgress();
        syncHint(syncEl, shared);
      });
    });

    ta.addEventListener("input", function () {
      if (ta.value.trim()) err.hidden = true;
    });

    var syncEl = document.createElement("p");
    syncEl.className = "sync-hint";
    syncHint(syncEl, state.backendUp);

    actions.appendChild(btn);
    wrap.appendChild(list);
    wrap.appendChild(label);
    wrap.appendChild(ta);
    wrap.appendChild(err);
    wrap.appendChild(actions);
    wrap.appendChild(syncEl);
    return wrap;
  }

  function renderSavedNotes(list, it) {
    list.innerHTML = "";
    itemNotes(it).forEach(function (n) {
      var li = document.createElement("li");
      var when = n.at ? fmtTime(n.at) : "";
      li.innerHTML = esc(n.note) +
        '<div class="by">' + esc(n.by || "Recruiter") + (when ? " · " + esc(when) : "") + "</div>";
      list.appendChild(li);
    });
  }

  function syncHint(p, shared) {
    p.textContent = shared
      ? "Saved — visible to other recruiters."
      : "Stored on this device only (backend offline) — other recruiters can’t see it yet.";
  }

  function syncReviewBtn(btn, it) {
    if (isReviewed(it)) {
      btn.className = "btn done";
      btn.textContent = "✓ Add note";
    } else {
      btn.className = "btn primary";
      btn.textContent = "Mark Reviewed";
    }
  }

  function refreshStatus(container, it) {
    container.classList.toggle("reviewed", isReviewed(it));
    container.classList.toggle("needs-review", !isReviewed(it));
    var st = container.querySelector(".item-status");
    if (st) {
      st.className = "item-status " + (isReviewed(it) ? "status-done" : "status-review");
      st.textContent = isReviewed(it) ? "✓" : "!";
      st.title = isReviewed(it) ? "Reviewed" : "Needs review";
    }
  }

  /* ---------------- helpers ---------------- */

  function fmtRange(start, end, ongoing) {
    function fmt(d) {
      if (!d) return "?";
      var months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
      var parts = d.split("-");
      if (parts.length < 2) return parts[0];
      return months[parseInt(parts[1], 10) - 1] + " " + parts[0];
    }
    if (!start && !end) return "No dates";
    return fmt(start) + " – " + (ongoing || !end ? "Present" : fmt(end));
  }

  function fmtTime(iso) {
    try {
      return new Date(iso).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
    } catch (e) { return iso; }
  }

  function sectionName(s) {
    if (!s) return "Resume";
    return s.charAt(0).toUpperCase() + s.slice(1) + " section";
  }

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  function showLoading(on) {
    el.loading.hidden = !on;
    if (on) { el.error.hidden = true; el.review.hidden = true; }
  }
  function showError(msg) {
    showLoading(false);
    el.review.hidden = true;
    el.error.hidden = false;
    el.errorMsg.textContent = msg;
  }
})();
