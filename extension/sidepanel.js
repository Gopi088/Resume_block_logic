/* Recruiter review panel logic. Consumes the backend /api/parse contract
   (or the bundled real-parser fixture offline). No mock data is constructed:
   every rendered fact comes from parser fields; missing data renders as an
   explicit unavailable state. */
(function () {
  "use strict";

  var BACKEND = "http://localhost:8000";
  var RESUME_NOTE_ID = "resume"; // the single review note for the whole resume
  var store = storage();

  var state = {
    doc: null,      // full parser payload (fixture or backend response)
    review: null,   // review projection
    note: null,     // {note, by, at} — the one resume-level review, shared
    reviewerId: "",
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
    reviewState: document.getElementById("reviewState"),
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
    savedNotes: document.getElementById("savedNotes"),
    resumeNote: document.getElementById("resumeNote"),
    resumeNoteError: document.getElementById("resumeNoteError"),
    markReviewedBtn: document.getElementById("markReviewedBtn"),
    syncHint: document.getElementById("syncHint"),
  };

  el.retry.addEventListener("click", function () { loadFixture(); });
  el.file.addEventListener("change", onFile);
  el.timelineToggle.addEventListener("click", function () {
    toggleSection(el.timeline, el.timelineToggle, el.timelineChev);
  });
  el.undatedToggle.addEventListener("click", function () {
    toggleSection(el.undated, el.undatedToggle, el.undatedChev);
  });
  el.markReviewedBtn.addEventListener("click", onMarkReviewed);
  el.resumeNote.addEventListener("input", function () {
    if (el.resumeNote.value.trim()) el.resumeNoteError.hidden = true;
  });

  // Stable anonymous identity, generated once — the recruiter never types a name.
  store.get("reviewerId", function (v) {
    state.reviewerId = v || ("R-" + Math.random().toString(36).slice(2, 6));
    store.set("reviewerId", state.reviewerId);
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
    loadNote(function () {
      render();
      showLoading(false);
    });
  }

  /* ---------------- the single shared resume note ---------------- */

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

  function docSha() {
    return (state.doc && state.doc.sha256) || (state.review.candidate.sha256) || "unknown";
  }

  function pickNote(entries) {
    // One note per resume: the latest shared entry wins.
    if (!entries) return null;
    if (Array.isArray(entries)) {
      var valid = entries.filter(function (n) { return n && n.note; });
      return valid.length ? valid[valid.length - 1] : null;
    }
    return entries.note ? entries : null;
  }

  function loadNote(cb) {
    store.get(storageKey(), function (local) {
      var localNote = pickNote(local && local[RESUME_NOTE_ID]);
      fetch(BACKEND + "/api/notes/" + docSha())
        .then(function (r) { if (!r.ok) throw new Error("no backend"); return r.json(); })
        .then(function (body) {
          state.backendUp = true;
          var shared = pickNote(body && body.notes && body.notes[RESUME_NOTE_ID]);
          // Latest timestamp wins so no recruiter's review is silently lost.
          state.note = latest(localNote, shared);
          cb();
        })
        .catch(function () { state.note = localNote; cb(); });
    });
  }

  function latest(a, b) {
    if (!a) return b;
    if (!b) return a;
    return (b.at || "") >= (a.at || "") ? b : a;
  }

  function onMarkReviewed() {
    var text = el.resumeNote.value.trim();
    if (!text) {
      el.resumeNoteError.hidden = false;
      el.resumeNote.focus();
      return;
    }
    el.resumeNoteError.hidden = true;
    el.markReviewedBtn.disabled = true;
    var entry = { note: text, by: state.reviewerId, at: new Date().toISOString() };
    state.note = entry;
    store.set(storageKey(), { resume: entry });
    fetch(BACKEND + "/api/notes/" + docSha(), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ item_id: RESUME_NOTE_ID, note: text, by: state.reviewerId }),
    }).then(function () { state.backendUp = true; finishSave(true); })
      .catch(function () { finishSave(false); });
  }

  function finishSave(shared) {
    el.markReviewedBtn.disabled = false;
    el.resumeNote.value = "";
    renderReviewSection(shared);
  }

  /* ---------------- rendering ---------------- */

  function render() {
    var r = state.review;
    el.error.hidden = true;
    el.review.hidden = false;

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
    renderReviewSection(state.backendUp);
  }

  function renderReviewSection(shared) {
    el.savedNotes.innerHTML = "";
    if (state.note) {
      var li = document.createElement("li");
      li.innerHTML = esc(state.note.note) +
        '<div class="by">Reviewed · ' + esc(fmtTime(state.note.at)) + "</div>";
      el.savedNotes.appendChild(li);
      el.reviewState.textContent = "✓ Reviewed · " + fmtTime(state.note.at);
      el.reviewState.className = "review-state small done";
      el.markReviewedBtn.className = "btn done";
      el.markReviewedBtn.textContent = "✓ Update review";
    } else {
      el.reviewState.textContent = "Not reviewed yet";
      el.reviewState.className = "review-state small todo";
      el.markReviewedBtn.className = "btn primary";
      el.markReviewedBtn.textContent = "Mark Reviewed";
    }
    el.syncHint.textContent = shared
      ? "Saved — visible to other recruiters."
      : "Stored on this device only (backend offline) — other recruiters can’t see it yet.";
  }

  /* ---------------- compact rows: dates + title only ---------------- */

  function rowHead(datesLabel, titleText) {
    var head = document.createElement("button");
    head.className = "item-head";
    head.setAttribute("aria-expanded", "false");
    head.title = titleText || "";
    head.innerHTML =
      '<span class="item-dates">' + esc(datesLabel) + "</span>" +
      '<span class="item-title">' + esc(titleText || "—") + "</span>" +
      '<span class="chev">▸</span>';
    return head;
  }

  function bindExpand(container, head) {
    var chev = head.querySelector(".chev");
    head.addEventListener("click", function () {
      var open = container.classList.toggle("open");
      head.setAttribute("aria-expanded", open ? "true" : "false");
      chev.textContent = open ? "▾" : "▸";
    });
  }

  function renderEvent(ev) {
    var li = document.createElement("li");
    li.className = "item";
    var head = rowHead(fmtRange(ev.start_date, ev.end_date, ev.is_ongoing), ev.title);
    bindExpand(li, head);

    // Expanded: only what doesn't fit in the row + the evidence action.
    var body = document.createElement("div");
    body.className = "item-body";
    if (ev.organization && !sameText(ev.organization, ev.title)) {
      body.appendChild(detail("Organization", ev.organization));
    }
    body.appendChild(detail("Confidence", ev.confidence_band + (ev.trusted ? "" : " — verify against resume")));
    body.appendChild(viewResumeBlock(ev));

    li.appendChild(head);
    li.appendChild(body);
    return li;
  }

  function renderGap(g) {
    var div = document.createElement("div");
    div.className = "item gap";
    var months = g.gap_months_approx != null ? " (" + g.gap_months_approx + " months)" : "";
    var head = rowHead(fmtRange(g.start_date, g.end_date, false), "Potential career gap" + months);
    head.title = "No employment stated here — your call.";
    bindExpand(div, head);

    var body = document.createElement("div");
    body.className = "item-body";
    body.appendChild(detail("Why flagged", "No employment covers this stretch (breaks under 90 days are ignored)."));
    div.appendChild(head);
    div.appendChild(body);
    return div;
  }

  function renderUndated(u) {
    var div = document.createElement("div");
    div.className = "item";
    var head = rowHead("No dates", u.title || "Entry");
    bindExpand(div, head);

    var body = document.createElement("div");
    body.className = "item-body";
    if (u.organization && !sameText(u.organization, u.title)) {
      body.appendChild(detail("Organization", u.organization));
    }
    body.appendChild(detail("Confidence", u.confidence_band + " — verify against resume"));
    body.appendChild(viewResumeBlock(u));

    div.appendChild(head);
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

  /* ---------------- View in Resume: highlight the open resume tab ---------------- */

  function viewResumeBlock(it) {
    var wrap = document.createElement("div");
    var ev = it.evidence;
    if (!ev || !ev.excerpt) {
      wrap.innerHTML = '<p class="muted small">Exact source text isn’t available for this item.</p>';
      return wrap;
    }
    var btn = document.createElement("button");
    btn.className = "btn";
    btn.textContent = "View in Resume";
    btn.title = "Highlight this exact passage in the open resume tab";

    var status = document.createElement("p");
    status.className = "muted small";
    status.hidden = true;

    btn.addEventListener("click", function () {
      btn.disabled = true;
      btn.textContent = "Finding…";
      highlightInOpenTab(it, function (res) {
        btn.disabled = false;
        btn.textContent = "View in Resume";
        status.hidden = false;
        // The button only ever highlights the open resume tab. Nothing from
        // the resume is rendered in the panel and no backend page is opened.
        if (res.ok) {
          status.textContent = "Highlighted in the open resume tab.";
        } else if (res.reason === "pdf-viewer") {
          status.textContent = "Chrome’s PDF viewer can’t be highlighted by any extension — " +
            "open the resume as text or HTML to use this.";
        } else {
          status.textContent = "Couldn’t find that passage in the active tab — " +
            "open the resume file in a tab, then try again. " +
            "(For file:// URLs, reload the tab after enabling “Allow access to file URLs”.)";
        }
      });
    });

    var actions = document.createElement("div");
    actions.className = "actions";
    actions.appendChild(btn);
    wrap.appendChild(actions);
    wrap.appendChild(status);
    return wrap;
  }

  function sendToTab(tabId, texts, cb) {
    function send(afterInject) {
      try {
        chrome.tabs.sendMessage(tabId, { type: "HIGHLIGHT_RESUME_EVIDENCE", texts: texts }, function (resp) {
          if (chrome.runtime.lastError) return afterInject ? cb(false) : inject();
          cb(!!(resp && resp.found));
        });
      } catch (e) { cb(false); }
    }
    function inject() {
      if (!chrome.scripting) return cb(false);
      try {
        chrome.scripting.executeScript({ target: { tabId: tabId }, files: ["content.js"] }, function () {
          if (chrome.runtime.lastError) return cb(false); // e.g. PDF viewer: not scriptable
          send(true);
        });
      } catch (e) { cb(false); }
    }
    send(false);
  }

  function highlightInOpenTab(it, cb) {
    var ev = it.evidence || {};
    var texts = [];
    if (ev.raw_range) texts.push(ev.raw_range);
    var firstLine = (ev.excerpt || "").split("\n").filter(function (l) { return l.trim(); })[0];
    if (firstLine) texts.push(firstLine.trim().slice(0, 120));
    if (!texts.length) return cb({ ok: false, reason: "no-text" });
    try {
      if (typeof chrome === "undefined" || !chrome.tabs) return cb({ ok: false, reason: "no-tabs" });
      chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
        if (!tabs || !tabs[0] || tabs[0].id == null) return cb({ ok: false, reason: "no-tab" });
        var tab = tabs[0];
        sendToTab(tab.id, texts, function (ok) {
          if (ok) return cb({ ok: true });
          cb({ ok: false, reason: looksLikePdf(tab.url) ? "pdf-viewer" : "not-found" });
        });
      });
    } catch (e) { cb({ ok: false, reason: "error" }); }
  }

  function looksLikePdf(url) {
    return !!url && /\.pdf([?#]|$)/i.test(url);
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
