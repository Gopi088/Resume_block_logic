/* Recruiter review panel logic. Consumes the backend /api/parse contract
   (or the bundled real-parser fixture offline). No mock data is constructed:
   every rendered fact comes from parser fields; missing data renders as an
   explicit unavailable state. */
(function () {
  "use strict";

  var BACKEND = "http://localhost:8000";
  var RESUME_URL_RE = /\.(pdf|txt|html?|md)([#?]|$)/i;
  var store = storage();

  // Backend `section` values, mapped to recruiter-facing kind labels.
  // Experience entries are roles (no label needed); everything else is named
  // for what it is so dated blocks are never mislabeled as career roles.
  var KIND_LABELS = {
    education: "Education",
    certifications: "Certification",
    awards: "Award",
    projects: "Project",
    publications: "Publication",
    languages: "Languages",
    volunteering: "Volunteering",
    interests: "Interests",
    references: "References",
    skills: "Skills",
    summary: "Summary",
    other: "Other",
    unknown: "Other",
    contact: "Contact",
    boilerplate: "Other"
  };

  var state = {
    doc: null,      // full parser payload (fixture or backend response)
    review: null,   // review projection
    notes: {},      // itemId -> [{note, by, at}] shared across recruiters
    reviewerId: "",
    backendUp: false,
    isFixture: true,
    lastAutoUrl: "",
    activeTab: null,
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
    accPct: document.getElementById("accPct"),
    accNote: document.getElementById("accNote"),
    actionNote: document.getElementById("actionNote"),
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
  };

  el.retry.addEventListener("click", function () { loadFixture(); });
  el.file.addEventListener("change", onFile);
  el.timelineToggle.addEventListener("click", function () {
    toggleSection(el.timeline, el.timelineToggle, el.timelineChev);
  });
  el.undatedToggle.addEventListener("click", function () {
    toggleSection(el.undated, el.undatedToggle, el.undatedChev);
  });

  store.get("reviewerId", function (v) {
    state.reviewerId = v || ("R-" + Math.random().toString(36).slice(2, 6));
    store.set("reviewerId", state.reviewerId);
  });
  if (typeof chrome !== "undefined" && chrome.runtime && chrome.runtime.onMessage) {
    chrome.runtime.onMessage.addListener(function (msg) {
      if (msg && msg.type === "OPEN_RESUME_TAB" && msg.url) {
        state.activeTab = { id: msg.id != null ? msg.id : null, url: msg.url };
        onOpenTabUrl(msg.url);
      }
    });
  }
  loadFixture();
  maybeAutoAnalyze();
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
        state.isFixture = true;
        setDocument(doc);
      })
      .catch(function (err) { showError("Sample data could not be loaded: " + err.message); });
  }

  function pingBackend() {
    fetch(BACKEND + "/health").then(function () { state.backendUp = true; }).catch(function () {});
  }

  function onOpenTabUrl(url) {
    if (!state.isFixture || url === state.lastAutoUrl) return;
    if (RESUME_URL_RE.test(url)) analyzeTabUrl(url);
  }

  function maybeAutoAnalyze() {
    try {
      if (typeof chrome === "undefined" || !chrome.tabs) return;
      chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
        var url = tabs && tabs[0] && tabs[0].url;
        if (!url) return;
        state.activeTab = { id: tabs[0].id != null ? tabs[0].id : null, url: url };
        if (RESUME_URL_RE.test(url)) analyzeTabUrl(url);
      });
    } catch (e) { /* stay on fixture */ }
  }

  function analyzeTabUrl(url) {
    state.lastAutoUrl = url;
    showLoading(true);
    el.source.textContent = "Analyzing open resume…";
    fetch(BACKEND + "/api/parse-url", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: url }),
    })
      .then(function (r) { return r.json(); })
      .then(function (doc) {
        if (!doc || doc.ok !== true) throw new Error((doc && doc.error) || "parse failed");
        state.backendUp = true;
        state.isFixture = false;
        el.source.textContent = (doc.filename || "Resume") + " (open tab)";
        setDocument(doc);
      })
      .catch(function () { loadFixture(); });
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
        state.isFixture = false;
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

  /* ---------------- shared per-item notes ---------------- */

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
    var out = {};
    Object.keys(raw || {}).forEach(function (id) {
      var v = raw[id];
      if (Array.isArray(v)) out[id] = v.filter(function (n) { return n && n.note; });
      else if (v && v.note) out[id] = [{ note: v.note, by: v.by || "auto", at: v.at || null }];
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
    var entry = { note: noteText, by: state.reviewerId, at: new Date().toISOString() };
    var id = itemId(it);
    state.notes[id] = (state.notes[id] || []).concat([entry]);
    saveNotes();
    fetch(BACKEND + "/api/notes/" + docSha(), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ item_id: id, note: noteText, by: state.reviewerId }),
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

    // Compact header: name + most recent role + confidence + action.
    el.name.textContent = r.candidate.name || "Unnamed candidate";
    var recentRole = mostRecentRole(r);
    el.role.textContent = recentRole || "";
    el.role.hidden = !recentRole;

    el.accPct.textContent = r.accuracy.percent + "%";
    if (r.accuracy.percent < 45) {
      el.accNote.textContent = "⚠ Low — some items need review";
    } else {
      el.accNote.textContent = "";
    }
    updateActionNote();

    // Gaps first (the alert), shown only when found.
    el.gaps.innerHTML = "";
    var gaps = r.gaps || [];
    el.gapSection.hidden = gaps.length === 0;
    el.gapCount.textContent = gaps.length ? "· " + gaps.length : "";
    gaps.forEach(function (g) { el.gaps.appendChild(renderGap(g)); });

    // Timeline: most-recent-first, collapsed until wanted.
    var events = (r.events || []).slice().sort(function (a, b) {
      if (a.start_date === b.start_date) return 0;
      return (a.start_date || "") < (b.start_date || "") ? 1 : -1;
    });
    el.timeline.innerHTML = "";
    el.timelineCount.textContent = events.length ? "· " + events.length : "";
    events.forEach(function (ev) { el.timeline.appendChild(renderEvent(ev)); });

    // Other information: recruiter wording, collapsed until wanted.
    el.undated.innerHTML = "";
    var und = r.undated || [];
    el.undatedSection.hidden = und.length === 0;
    el.undatedCount.textContent = und.length ? "· " + und.length : "";
    und.forEach(function (u) { el.undated.appendChild(renderUndated(u)); });

    el.empty.hidden = events.length > 0 || gaps.length > 0;
  }

  function mostRecentRole(r) {
    var exp = (r.events || []).filter(function (e) { return e.section === "experience"; });
    if (!exp.length) return null;
    exp.sort(function (a, b) {
      if (a.start_date === b.start_date) return 0;
      return (a.start_date || "") < (b.start_date || "") ? 1 : -1;
    });
    return exp[0].title || null;
  }

  function reviewableItems() {
    var r = state.review;
    return (r.events || []).concat(r.gaps || [], r.undated || []);
  }

  function updateActionNote() {
    var all = reviewableItems();
    var done = all.filter(isReviewed).length;
    var left = all.length - done;
    if (!all.length) {
      el.actionNote.textContent = "";
      el.actionNote.className = "action-note";
    } else if (left === 0) {
      el.actionNote.textContent = "✓ All reviewed";
      el.actionNote.className = "action-note done";
    } else {
      el.actionNote.textContent = "⚠ " + left + " item" + (left === 1 ? "" : "s") + " need review";
      el.actionNote.className = "action-note todo";
    }
  }

  function kindLabel(section) {
    if (!section || section === "experience") return null;
    return KIND_LABELS[section] || (section.charAt(0).toUpperCase() + section.slice(1));
  }

  /* ---------------- compact rows: date / title / org / status / expand ---------------- */

  function rowHead(datesLabel, titleHtml, subText, it) {
    var head = document.createElement("button");
    head.className = "trow";
    head.setAttribute("aria-expanded", "false");
    var main = document.createElement("span");
    main.className = "tmain";
    main.innerHTML = '<span class="ttitle">' + titleHtml + "</span>" +
      (subText ? '<br/><span class="tsub">' + esc(subText) + "</span>" : "") +
      '<br/><span class="tstatus ' + (isReviewed(it) ? "done" : "review") + '">' +
      (isReviewed(it) ? "✓ Reviewed" : "⚠ Needs review") + "</span>";
    head.innerHTML = '<span class="tdates">' + esc(datesLabel) + "</span>";
    head.appendChild(main);
    var chev = document.createElement("span");
    chev.className = "chev";
    chev.textContent = "▸";
    head.appendChild(chev);
    return { head: head, chev: chev };
  }

  function bindExpand(item, head, chev) {
    head.addEventListener("click", function () {
      var open = item.classList.toggle("open");
      head.setAttribute("aria-expanded", open ? "true" : "false");
      chev.textContent = open ? "▾" : "▸";
    });
  }

  function titleFor(ev) {
    var label = kindLabel(ev.section);
    var title = ev.title || "Untitled";
    if (ev.section === "experience") {
      return '<span class="dot muted">●</span> ' + esc(title);
    }
    if (label) {
      return '<span class="tkind">' + esc(label) + "</span>" + esc(title);
    }
    return esc(title);
  }

  function renderEvent(ev) {
    var li = document.createElement("li");
    li.className = "titem";
    var sub = (ev.organization && !sameText(ev.organization, ev.title)) ? ev.organization : "";
    if (ev.location) sub = sub ? sub + " · " + ev.location : ev.location;
    var parts = rowHead(fmtRange(ev.start_date, ev.end_date, ev.is_ongoing), titleFor(ev), sub, ev);
    bindExpand(li, parts.head, parts.chev);

    var body = document.createElement("div");
    body.className = "tbody";
    body.appendChild(detail("Organization", ev.organization));
    if (ev.location) body.appendChild(detail("Location", ev.location));
    if (ev.raw_range) body.appendChild(detail("Dates as written", ev.raw_range));
    body.appendChild(detail("Confidence", ev.confidence_band + (ev.trusted ? "" : " — check against resume")));
    body.appendChild(viewResumeBlock(ev));
    body.appendChild(reviewBlock(ev, li, parts.head));

    li.appendChild(parts.head);
    li.appendChild(body);
    return li;
  }

  function renderGap(g) {
    var div = document.createElement("div");
    div.className = "titem gap";
    var months = g.gap_months_approx != null ? " (" + g.gap_months_approx + " months)" : "";
    var parts = rowHead(fmtRange(g.start_date, g.end_date, false),
      esc("Potential gap"), "No employment stated" + months, g);
    parts.head.title = "Expand for details";
    bindExpand(div, parts.head, parts.chev);

    var body = document.createElement("div");
    body.className = "tbody";
    body.appendChild(detail("Why flagged", "No employment covers this stretch (breaks under 90 days are ignored). Only you can judge what it means."));
    body.appendChild(reviewBlock(g, div, parts.head));

    div.appendChild(parts.head);
    div.appendChild(body);
    return div;
  }

  function renderUndated(u) {
    var div = document.createElement("div");
    div.className = "titem";
    var label = kindLabel(u.section);
    var title = (u.title || "Entry");
    var titleHtml = label
      ? '<span class="tkind">' + esc(label) + "</span>" + esc(title)
      : esc(title);
    var sub = (u.organization && !sameText(u.organization, u.title)) ? u.organization : "";
    var parts = rowHead("Undated", titleHtml, sub, u);
    bindExpand(div, parts.head, parts.chev);

    var body = document.createElement("div");
    body.className = "tbody";
    body.appendChild(detail("Note", "This could not be placed on the timeline — no dates were found for it."));
    if (u.organization && !sameText(u.organization, u.title)) {
      body.appendChild(detail("Organization", u.organization));
    }
    body.appendChild(detail("Confidence", u.confidence_band + " — check against resume"));
    body.appendChild(viewResumeBlock(u));
    body.appendChild(reviewBlock(u, div, parts.head));

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
    if (v == null || v === "") return document.createTextNode("");
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
      var phrases = fragmentPhrases(it);
      try {
        if (phrases.length && navigator.clipboard) {
          navigator.clipboard.writeText(phrases[0]).catch(function () {});
        }
      } catch (e) { /* clipboard unavailable */ }
      var cached = state.activeTab;
      if (cached && cached.url && /^https?:|^file:/i.test(cached.url) && cached.id != null) {
        btn.disabled = true;
        btn.textContent = "Finding…";
        syncJump(cached, it, phrases, function (res) {
          btn.disabled = false;
          btn.textContent = "View in Resume";
          jumpStatus(status, res);
        });
        return;
      }
      btn.disabled = true;
      btn.textContent = "Finding…";
      highlightInOpenTab(it, function (res) {
        btn.disabled = false;
        btn.textContent = "View in Resume";
        jumpStatus(status, res);
      });
    });

    var actions = document.createElement("div");
    actions.className = "actions";
    actions.appendChild(btn);
    wrap.appendChild(actions);
    wrap.appendChild(status);
    return wrap;
  }

  function jumpStatus(status, res) {
    status.hidden = false;
    if (res.ok && res.via === "fragment") {
      status.textContent = res.page
        ? "Jumped to page " + res.page + " in the resume tab."
        : "Jumped to the passage in the resume tab.";
    } else if (res.ok) {
      status.textContent = "Highlighted in the open resume tab.";
    } else {
      status.textContent = "That passage isn’t in the active tab. " +
        "Click Parse resume and upload the file that’s open, then try again.";
    }
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
          if (chrome.runtime.lastError) return cb(false);
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
          fragmentNavigate(tab, it, cb);
        });
      });
    } catch (e) { cb({ ok: false, reason: "error" }); }
  }

  function wordsOf(s, n) {
    return (s || "").split(/[^A-Za-z0-9]+/)
      .filter(function (w) { return /[A-Za-z0-9]/.test(w); }).slice(0, n);
  }

  function uniqueWords(list, n) {
    var seen = {}, out = [];
    list.forEach(function (w) {
      var k = w.toLowerCase();
      if (!seen[k]) { seen[k] = true; out.push(w); }
    });
    return out.slice(0, n);
  }

  function fragmentPhrases(it) {
    var ev = it.evidence || {};
    var out = [];
    function take(words) {
      if (words.length >= 2 && /[a-zA-Z]{3,}/.test(words.join(" ")) && out.length < 3) {
        var p = words.join(" ");
        if (out.indexOf(p) === -1) out.push(p);
      }
    }
    take(uniqueWords(wordsOf(it.title || "", 6).concat(wordsOf(it.organization || "", 6)), 6));
    take(wordsOf(ev.raw_range || "", 8));
    var firstLine = ((ev.excerpt || "").split("\n").filter(function (l) { return l.trim(); })[0] || "");
    take(wordsOf(firstLine, 8));
    return out;
  }

  function fragmentPhrase(it) {
    return fragmentPhrases(it)[0] || null;
  }

  function evidenceUrl(tabUrl, page, phrases) {
    var base = tabUrl.split("#")[0];
    var url = base + (base.indexOf("?") === -1 ? "?" : "&") + "evjump=" + Date.now() +
      (page ? "#page=" + page : "#");
    (phrases || []).forEach(function (p, i) {
      url += (i === 0 ? ":~:text=" : "&text=") + encodeURIComponent(p);
    });
    return url;
  }

  function evidencePage(it) {
    var ev = it.evidence || {};
    return (ev.pages && ev.pages.length) ? (ev.pages[0] + 1) : null;
  }

  function syncJump(cached, it, phrases, cb) {
    try {
      chrome.tabs.update(cached.id,
        { url: evidenceUrl(cached.url, evidencePage(it), phrases) }, function () {
          if (chrome.runtime.lastError) return cb({ ok: false, reason: "not-found" });
          cb({ ok: true, via: "fragment", page: evidencePage(it) });
        });
    } catch (e) { cb({ ok: false, reason: "not-found" }); }
  }

  function fragmentNavigate(tab, it, cb) {
    var phrases = fragmentPhrases(it);
    if (!tab.url || !/^https?:|^file:/i.test(tab.url) || !phrases.length) {
      return cb({ ok: false, reason: "not-found" });
    }
    var page = evidencePage(it);
    try {
      chrome.tabs.update(tab.id, { url: evidenceUrl(tab.url, page, phrases) }, function () {
        if (chrome.runtime.lastError) return cb({ ok: false, reason: "not-found" });
        cb({ ok: true, via: "fragment", page: page });
      });
    } catch (e) { cb({ ok: false, reason: "not-found" }); }
  }

  /* ---------------- per-item review: mandatory note, stays with the item ---------------- */

  function reviewBlock(it, item, head) {
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

    var err = document.createElement("p");
    err.className = "note-error";
    err.hidden = true;

    var actions = document.createElement("div");
    actions.className = "actions";
    var btn = document.createElement("button");
    syncReviewBtn(btn, it);
    btn.addEventListener("click", function () {
      var note = ta.value.trim();
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
        refreshStatus(item, head, it);
        updateActionNote();
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
        '<div class="by">Reviewed' + (when ? " · " + esc(when) : "") + "</div>";
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

  function refreshStatus(item, head, it) {
    var st = head.querySelector(".tstatus");
    if (st) {
      st.className = "tstatus " + (isReviewed(it) ? "done" : "review");
      st.textContent = isReviewed(it) ? "✓ Reviewed" : "⚠ Needs review";
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
