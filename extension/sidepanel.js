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
    notes: {},      // itemId -> {note, at}
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
    role: document.getElementById("candRole"),
    meta: document.getElementById("candMeta"),
    accPct: document.getElementById("accPct"),
    accLabel: document.getElementById("accLabel"),
    accHint: document.getElementById("accHint"),
    progressBar: document.getElementById("progressBar"),
    progressText: document.getElementById("progressText"),
    timeline: document.getElementById("timeline"),
    timelineCount: document.getElementById("timelineCount"),
    gapSection: document.getElementById("gapSection"),
    gaps: document.getElementById("gaps"),
    gapCount: document.getElementById("gapCount"),
    undatedSection: document.getElementById("undatedSection"),
    undated: document.getElementById("undated"),
    undatedCount: document.getElementById("undatedCount"),
  };

  el.retry.addEventListener("click", function () { loadFixture(); });
  el.file.addEventListener("change", onFile);

  loadFixture();
  pingBackend();

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
    fetch(BACKEND + "/health").catch(function () { /* offline: fixture stays */ });
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

  /* ---------------- review state (notes persist) ---------------- */

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

  function loadNotes(cb) {
    store.get(storageKey(), function (v) { state.notes = v || {}; cb(); });
  }
  function saveNotes() { store.set(storageKey(), state.notes); }

  function itemId(it) {
    return it.kind === "gap" ? it.gap_id : it.entry_id;
  }
  function isReviewed(it) {
    var n = state.notes[itemId(it)];
    return !!(n && n.note && n.note.trim());
  }

  /* ---------------- rendering ---------------- */

  function render() {
    var r = state.review;
    el.error.hidden = true;
    el.review.hidden = false;

    // Candidate header: name, most recent role, years.
    el.name.textContent = r.candidate.name || "Unnamed candidate";
    var roleBits = [r.candidate.current_role, r.candidate.current_org].filter(Boolean);
    el.role.textContent = roleBits.join(" · ") || "Role not extracted";
    var meta = [];
    if (r.candidate.years_experience != null) meta.push(r.candidate.years_experience + " yrs span");
    meta.push(reviewableCount() + " items to review");
    el.meta.textContent = meta.join(" · ");

    // Accuracy: compact, never a dashboard metric.
    el.accPct.textContent = r.accuracy.percent + "%";
    el.accLabel.textContent = r.accuracy.label;
    if (r.accuracy.percent < 45) {
      el.accHint.hidden = false;
      el.accHint.textContent = "Low extraction confidence — verify each item against the resume.";
    } else { el.accHint.hidden = true; }

    // Timeline: most-recent-first (recruiter scanning order).
    var events = (r.events || []).slice().sort(function (a, b) {
      if (a.start_date === b.start_date) return 0;
      return (a.start_date || "") < (b.start_date || "") ? 1 : -1; // most-recent-first
    });
    el.timeline.innerHTML = "";
    el.timelineCount.textContent = events.length ? "· " + events.length + " roles" : "";
    el.empty.hidden = events.length > 0 || (r.gaps || []).length > 0;
    events.forEach(function (ev) { el.timeline.appendChild(renderEvent(ev)); });

    // Gaps: visually distinct, framed as needing attention — not a verdict.
    el.gaps.innerHTML = "";
    var gaps = r.gaps || [];
    el.gapSection.hidden = gaps.length === 0;
    el.gapCount.textContent = gaps.length ? "· " + gaps.length + " to check" : "";
    gaps.forEach(function (g) { el.gaps.appendChild(renderGap(g)); });

    // Undated entries.
    el.undated.innerHTML = "";
    var und = r.undated || [];
    el.undatedSection.hidden = und.length === 0;
    el.undatedCount.textContent = und.length ? "· " + und.length : "";
    und.forEach(function (u) { el.undated.appendChild(renderUndated(u)); });

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

  /* ---------------- items ---------------- */

  function renderEvent(ev) {
    var li = document.createElement("li");
    li.className = "item" + (isReviewed(ev) ? " reviewed" : (ev.needs_review ? " needs-review" : ""));
    li.dataset.item = itemId(ev);

    var head = document.createElement("button");
    head.className = "item-head";
    head.setAttribute("aria-expanded", "false");
    head.innerHTML =
      '<span class="item-dates">' + esc(fmtRange(ev.start_date, ev.end_date, ev.is_ongoing)) + "</span>" +
      '<span class="item-main"><span class="item-title">' + esc(ev.title || "Role not extracted") + "</span><br/>" +
      '<span class="item-sub">' + esc([ev.organization, ev.location].filter(Boolean).join(" · ") || sectionName(ev.section)) + "</span></span>" +
      statusHtml(ev) +
      '<span class="chev">▾</span>';
    head.addEventListener("click", function () {
      li.classList.toggle("open");
      head.setAttribute("aria-expanded", li.classList.contains("open") ? "true" : "false");
    });

    var body = document.createElement("div");
    body.className = "item-body";
    body.appendChild(detail("Detected from", sectionName(ev.section)));
    body.appendChild(detail("Confidence", ev.confidence_band + (ev.trusted ? "" : " — verify against resume")));
    if (ev.raw_range) body.appendChild(detail("Dates as written", ev.raw_range));
    body.appendChild(evidenceBlock(ev));
    body.appendChild(reviewBlock(ev, li));

    li.appendChild(head);
    li.appendChild(body);
    return li;
  }

  function renderGap(g) {
    var div = document.createElement("div");
    div.className = "item gap" + (isReviewed(g) ? " reviewed" : " needs-review");
    var months = g.gap_months_approx != null ? " (" + g.gap_months_approx + " months)" : "";
    var head = document.createElement("button");
    head.className = "item-head";
    head.innerHTML =
      '<span class="item-dates">' + esc(fmtRange(g.start_date, g.end_date, false)) + "</span>" +
      '<span class="item-main"><span class="item-title">Potential career gap' + esc(months) + "</span><br/>" +
      '<span class="item-sub">No employment stated here — your call.</span></span>' +
      statusHtml(g) +
      '<span class="chev">▾</span>';
    head.addEventListener("click", function () { div.classList.toggle("open"); });

    var body = document.createElement("div");
    body.className = "item-body";
    body.appendChild(detail("Why flagged", "No employment covers this stretch (gaps under 90 days are ignored)."));
    var ev = document.createElement("div");
    ev.className = "evidence";
    ev.innerHTML = '<span class="muted">A gap is an absence — there is no resume passage to show. ' +
      "Compare the roles around it in the timeline above.</span>";
    body.appendChild(ev);
    body.appendChild(reviewBlock(g, div));

    div.appendChild(head);
    div.appendChild(body);
    return div;
  }

  function renderUndated(u) {
    var div = document.createElement("div");
    div.className = "item" + (isReviewed(u) ? " reviewed" : " needs-review");
    var head = document.createElement("button");
    head.className = "item-head";
    head.innerHTML =
      '<span class="item-dates">No dates</span>' +
      '<span class="item-main"><span class="item-title">' + esc(u.title || "Entry") + "</span><br/>" +
      '<span class="item-sub">' + esc([u.organization, sectionName(u.section)].filter(Boolean).join(" · ")) + "</span></span>" +
      statusHtml(u) +
      '<span class="chev">▾</span>';
    head.addEventListener("click", function () { div.classList.toggle("open"); });

    var body = document.createElement("div");
    body.className = "item-body";
    body.appendChild(detail("Detected from", sectionName(u.section)));
    body.appendChild(detail("Confidence", u.confidence_band + " — verify against resume"));
    body.appendChild(evidenceBlock(u));
    body.appendChild(reviewBlock(u, div));

    div.appendChild(head);
    div.appendChild(body);
    return div;
  }

  function detail(k, v) {
    var p = document.createElement("p");
    p.className = "detail-row";
    p.innerHTML = '<span class="k">' + esc(k) + ":</span> " + esc(v);
    return p;
  }

  function statusHtml(it) {
    if (isReviewed(it)) return '<span class="item-status status-done">✓ Reviewed</span>';
    if (it.kind === "gap" || it.needs_review) return '<span class="item-status status-review">! Needs review</span>';
    return '<span class="item-status muted">—</span>';
  }

  /* ---------------- evidence: exact source, never generic ---------------- */

  function evidenceBlock(it) {
    var wrap = document.createElement("div");
    var ev = it.evidence;
    if (!ev || ev.unavailable || !ev.excerpt) {
      wrap.innerHTML = '<div class="evidence"><span class="muted">Exact source text isn’t available for this item.</span></div>';
      return wrap;
    }
    var html = esc(ev.excerpt);
    if (ev.raw_range) {
      // Highlight the exact matched date span inside the excerpt.
      html = html.replace(esc(ev.raw_range), "<mark>" + esc(ev.raw_range) + "</mark>");
    }
    var meta = [];
    if (ev.pages && ev.pages.length) meta.push("page " + ev.pages.map(function (p) { return p + 1; }).join(", "));
    if (ev.start_line) meta.push(ev.start_line + (ev.end_line && ev.end_line !== ev.start_line ? "–" + ev.end_line : ""));
    var div = document.createElement("div");
    div.className = "evidence";
    div.innerHTML = html + '<div class="evidence-meta">Exact source · ' + esc(meta.join(" · ") || "line provenance") + "</div>";

    var btn = document.createElement("button");
    btn.className = "btn";
    btn.textContent = "View in Resume";
    btn.title = "Scroll the open resume tab to this exact passage";
    btn.addEventListener("click", function () { viewInResume(it, btn); });

    var status = document.createElement("p");
    status.className = "muted small";
    status.style.margin = "4px 0 0";
    status.hidden = true;
    wrap.appendChild(div);
    wrap.appendChild(btn);
    wrap.appendChild(status);
    wrap._status = status;
    wrap._btn = btn;
    return wrap;
  }

  function viewInResume(it, btn) {
    var ev = it.evidence || {};
    var texts = [];
    if (ev.raw_range) texts.push(ev.raw_range);
    var firstLine = (ev.excerpt || "").split("\n").filter(function (l) { return l.trim(); })[0];
    if (firstLine) texts.push(firstLine.trim().slice(0, 120));
    if (!texts.length) return;
    btn.disabled = true;
    btn.textContent = "Finding…";
    var wrap = btn.parentElement;
    sendHighlight(texts, function (found) {
      btn.disabled = false;
      btn.textContent = "View in Resume";
      var s = wrap._status;
      s.hidden = false;
      s.textContent = found
        ? "Highlighted in the open resume tab."
        : "Not found in the open tab — the excerpt above is the exact source (open the parsed resume file to compare).";
    });
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

  /* ---------------- mandatory-note review gate ---------------- */

  function reviewBlock(it, container) {
    var id = itemId(it);
    var wrap = document.createElement("div");
    var saved = state.notes[id];

    var label = document.createElement("label");
    label.className = "note-label";
    label.textContent = "Review note";
    label.htmlFor = "note-" + id;

    var ta = document.createElement("textarea");
    ta.className = "note";
    ta.id = "note-" + id;
    ta.placeholder = "e.g. Verified against resume p.1 — dates match.";
    ta.value = (saved && saved.note) || "";

    var err = document.createElement("p");
    err.className = "note-error";
    err.hidden = true;
    err.textContent = "Add a review note before marking this item as reviewed.";

    var actions = document.createElement("div");
    actions.className = "actions";
    var btn = document.createElement("button");
    syncReviewBtn(btn, it);
    btn.addEventListener("click", function () {
      var note = ta.value.trim();
      if (!note) {
        // Hard gate: no note, no review. Explain + focus.
        err.hidden = false;
        ta.focus();
        return;
      }
      err.hidden = true;
      state.notes[id] = { note: note, at: new Date().toISOString() };
      saveNotes();
      syncReviewBtn(btn, it);
      refreshStatus(container, it);
      updateProgress();
    });

    ta.addEventListener("input", function () {
      if (ta.value.trim()) err.hidden = true;
      // Editing the note after review keeps the review (note is the record);
      // clearing it re-opens the item.
      if (isReviewed(it) && !ta.value.trim()) {
        delete state.notes[id];
        saveNotes();
        syncReviewBtn(btn, it);
        refreshStatus(container, it);
        updateProgress();
      } else if (ta.value.trim() && state.notes[id]) {
        state.notes[id] = { note: ta.value.trim(), at: new Date().toISOString() };
        saveNotes();
      }
    });

    actions.appendChild(btn);
    wrap.appendChild(label);
    wrap.appendChild(ta);
    wrap.appendChild(err);
    wrap.appendChild(actions);
    return wrap;
  }

  function syncReviewBtn(btn, it) {
    if (isReviewed(it)) {
      btn.className = "btn done";
      btn.textContent = "✓ Reviewed — update note to revise";
    } else {
      btn.className = "btn primary";
      btn.textContent = "Mark Reviewed";
    }
  }

  function refreshStatus(container, it) {
    container.classList.toggle("reviewed", isReviewed(it));
    container.classList.toggle("needs-review", !isReviewed(it) && (it.kind === "gap" || it.needs_review));
    var st = container.querySelector(".item-status");
    if (st) {
      var tmp = document.createElement("span");
      tmp.innerHTML = statusHtml(it);
      st.className = tmp.firstChild.className;
      st.textContent = tmp.firstChild.textContent;
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
