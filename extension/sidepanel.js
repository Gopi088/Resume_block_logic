/* Career Timeline panel. Consumes the backend /api/parse contract
   (or bundled fixtures offline). Facts come only from parser fields;
   missing data renders as explicit unavailable states, never invented.
   The "Your review" section behaviour is unchanged. */
(function () {
  "use strict";

  var T = (typeof window !== "undefined" && window.CareerTimeline) || null;

  var BACKEND = "http://localhost:8000";
  var RESUME_NOTE_ID = "resume";
  var RESUME_URL_RE = /\.(pdf|txt|html?|md)([#?]|$)/i;
  var store = storage();

  var state = {
    doc: null,
    review: null,
    note: null,
    reviewerId: "",
    backendUp: false,
    isFixture: true,
    lastAutoUrl: "",
    activeTab: null,
    linesById: {},
    tlByEntry: {},
    referenceDate: null,
    firstFlaggedId: null,
    reduceMotion: false,
  };

  var el = {
    loading: document.getElementById("stateLoading"),
    error: document.getElementById("stateError"),
    errorMsg: document.getElementById("errorMsg"),
    retry: document.getElementById("retryBtn"),
    review: document.getElementById("review"),
    empty: document.getElementById("stateEmpty"),
    file: document.getElementById("fileInput"),
    source: document.getElementById("sourceStatus"),
    rescan: document.getElementById("rescanLink"),
    snapshot: document.getElementById("snapshot"),
    name: document.getElementById("candName"),
    snapRole: document.getElementById("snapRole"),
    snapTotals: document.getElementById("snapTotals"),
    tenureBar: document.getElementById("tenureBar"),
    timelineToggle: document.getElementById("timelineToggle"),
    timelineToggleLabel: document.getElementById("timelineToggleLabel"),
    timelineChev: document.getElementById("timelineChev"),
    timelineWrap: document.getElementById("timelineWrap"),
    timeline: document.getElementById("timeline"),
    eduToggle: document.getElementById("eduToggle"),
    eduCount: document.getElementById("eduCount"),
    eduList: document.getElementById("eduList"),
    otherToggle: document.getElementById("otherToggle"),
    otherCount: document.getElementById("otherCount"),
    otherList: document.getElementById("otherList"),
    savedNotes: document.getElementById("savedNotes"),
    resumeNote: document.getElementById("resumeNote"),
    resumeNoteError: document.getElementById("resumeNoteError"),
    markReviewedBtn: document.getElementById("markReviewedBtn"),
    syncHint: document.getElementById("syncHint"),
    live: document.getElementById("liveStatus"),
  };

  el.retry.addEventListener("click", function () { loadFixture(); });
  el.file.addEventListener("change", onFile);
  el.rescan.addEventListener("click", onRescan);
  el.timelineToggle.addEventListener("click", function () {
    var open = el.timelineWrap.hidden;
    el.timelineWrap.hidden = !open;
    el.timelineToggle.setAttribute("aria-expanded", open ? "true" : "false");
    el.timelineToggleLabel.textContent = open ? "Hide full timeline" : "View full timeline";
    el.timelineChev.textContent = open ? "▴" : "▾";
  });
  el.eduToggle.addEventListener("click", function () {
    toggleSublist(el.eduList, el.eduToggle);
  });
  el.otherToggle.addEventListener("click", function () {
    toggleSublist(el.otherList, el.otherToggle);
  });
  el.markReviewedBtn.addEventListener("click", onMarkReviewed);
  el.resumeNote.addEventListener("input", function () {
    if (el.resumeNote.value.trim()) el.resumeNoteError.hidden = true;
  });

  store.get("reviewerId", function (v) {
    state.reviewerId = v || ("R-" + Math.random().toString(36).slice(2, 6));
    store.set("reviewerId", state.reviewerId);
  });
  try {
    state.reduceMotion = window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  } catch (e) { state.reduceMotion = false; }
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

  function toggleSublist(list, toggle) {
    var open = list.hidden;
    list.hidden = !open;
    toggle.setAttribute("aria-expanded", open ? "true" : "false");
    toggle.querySelector(".chev").textContent = open ? "▾" : "▸";
  }

  function fixtureParam() {
    try {
      var m = /[?&]fixture=([1-7])\b/.exec(window.location.search || "");
      return m ? m[1] : null;
    } catch (e) { return null; }
  }

  /* ---------------- data loading ---------------- */

  function loadFixture() {
    showLoading(true);
    var url = fixtureParam() ? ("fixtures/f" + fixtureParam() + ".json") : "sample_data.json";
    fetch(url)
      .then(function (r) { if (!r.ok) throw new Error("fixture missing"); return r.json(); })
      .then(function (doc) {
        el.source.textContent = fixtureParam()
          ? ("Preview fixture " + fixtureParam())
          : "Sample resume (real parser output)";
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
    announce("Reading resume…");
    el.source.textContent = "Reading resume…";
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
        el.source.textContent = doc.filename || "Resume";
        setDocument(doc);
      })
      .catch(function () {
        chrome.runtime.sendMessage({ type: "PARSER_RESULT", success: false, confidence: null, summary: null }).catch(function () {});
        loadFixture();
      });
  }

  function onRescan() {
    if (state.lastAutoUrl) {
      analyzeTabUrl(state.lastAutoUrl);
    } else {
      el.file.click();
    }
  }

  function onFile(e) {
    var f = e.target.files && e.target.files[0];
    if (!f) return;
    showLoading(true);
    el.source.textContent = "Reading resume…";
    announce("Reading resume…");
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
        chrome.runtime.sendMessage({ type: "PARSER_RESULT", success: false, confidence: null, summary: null }).catch(function () {});
        showError("The local review backend isn’t reachable at " + BACKEND +
          ". Start it with: .venv/bin/python -m uvicorn backend.server:app --port 8000");
      });
    e.target.value = "";
  }

  function setDocument(doc) {
    state.doc = doc;
    state.review = doc.review || null;
    if (!state.review) { showError("This payload has no review projection."); return; }
    state.linesById = {};
    (doc.lines || []).forEach(function (ln) { state.linesById[ln.line_id] = ln; });
    state.tlByEntry = {};
    ((doc.timeline && doc.timeline.events) || []).forEach(function (e) { state.tlByEntry[e.entry_id] = e; });
    state.referenceDate = (doc.timeline && doc.timeline.reference_date) || new Date().toISOString().slice(0, 10);
    loadNote(function () {
      render();
      showLoading(false);
    });
    var confidence = null, summary = null;
    if (state.review && state.review.accuracy) {
      confidence = state.review.accuracy.percent != null ? state.review.accuracy.percent / 100 : null;
    }
    if (state.review && state.review.events) {
      var expEvents = state.review.events.filter(function (e) { return e.section === "experience"; });
      if (expEvents.length) {
        var companies = {};
        expEvents.forEach(function (e) { if (e.organization) companies[e.organization] = 1; });
        var n = Object.keys(companies).length;
        summary = expEvents.length + " yr" + (expEvents.length !== 1 ? "s" : "") + " · " + n + " compan" + (n === 1 ? "y" : "ies");
      }
    }
    chrome.runtime.sendMessage({
      type: "PARSER_RESULT", success: true, confidence: confidence, summary: summary
    }).catch(function () {});
  }

  /* ---------------- the single shared resume note (unchanged) ---------------- */

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
          state.note = latest(localNote, shared);
          if (!state.note && local) {
            Object.keys(local).forEach(function (id) {
              if (id === RESUME_NOTE_ID) return;
              state.note = latest(state.note, pickNote(local[id]));
            });
          }
          cb();
        })
        .catch(function () {
          state.note = localNote;
          cb();
        });
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
      if (state.note) {
        el.resumeNote.value = state.note.note;
        el.resumeNoteError.hidden = true;
        el.resumeNote.focus();
        try {
          el.resumeNote.setSelectionRange(el.resumeNote.value.length, el.resumeNote.value.length);
        } catch (e) {}
        return;
      }
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
    renderReviewSection(shared);
  }

  function renderReviewSection(shared) {
    el.savedNotes.innerHTML = "";
    if (state.note) {
      var li = document.createElement("li");
      li.innerHTML = esc(state.note.note) +
        '<div class="by">Checked by ' + esc(state.note.by || "a recruiter") +
        " · " + esc(fmtDay(state.note.at)) + "</div>";
      var edit = document.createElement("button");
      edit.className = "link-btn";
      edit.textContent = "Edit";
      edit.title = "Load this note into the box to revise it";
      edit.addEventListener("click", function () {
        el.resumeNote.value = state.note ? state.note.note : "";
        el.resumeNoteError.hidden = true;
        el.resumeNote.focus();
      });
      li.appendChild(edit);
      el.savedNotes.appendChild(li);
      el.markReviewedBtn.className = "btn done";
      el.markReviewedBtn.textContent = "✓ Update review";
      el.resumeNote.value = "";
    } else {
      el.markReviewedBtn.className = "btn primary";
      el.markReviewedBtn.textContent = "Mark Reviewed";
      el.resumeNote.value = "";
    }
    el.syncHint.textContent = shared
      ? "Saved — visible to other recruiters."
      : "Stored on this device only (backend offline) — other recruiters can’t see it yet.";
  }

  function onMarkReviewed() {
    var text = el.resumeNote.value.trim();
    if (!text) {
      if (state.note) {
        el.resumeNote.value = state.note.note;
        el.resumeNoteError.hidden = true;
        el.resumeNote.focus();
        try {
          el.resumeNote.setSelectionRange(el.resumeNote.value.length, el.resumeNote.value.length);
        } catch (e) {}
        return;
      }
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
    renderReviewSection(shared);
  }

  /* ---------------- snapshot + trust ---------------- */

  function qualifyingJobs(events) {
    return (events || []).filter(function (e) {
      return T.entryType(e.section) === "job" && Number(e.confidence) >= T.CONF_MEDIUM && e.start_date;
    });
  }

  function endOf(ev) {
    if (ev.is_ongoing) return null;
    return ev.end_date || ev.start_date;
  }

  function roleMonths(ev) {
    return T.monthsBetween(ev.start_date.slice(0, 7) + "-01",
      ((endOf(ev) || state.referenceDate).slice(0, 7)) + "-01");
  }

  function render() {
    var r = state.review;
    el.error.hidden = true;
    el.review.hidden = false;

    var events = r.events || [];
    var jobs = events.filter(function (e) { return T.entryType(e.section) === "job"; });
    var lowCount = events.filter(function (e) { return Number(e.confidence) < T.CONF_MEDIUM; }).length;
    var overall = (r.accuracy && r.accuracy.percent != null) ? r.accuracy.percent / 100 : null;
    var trust = T.trustLevel(overall, lowCount);
    var manual = trust.level === "manual";

    renderTimeline(r, jobs);
    renderSnapshot(r, jobs, trust, manual);
    renderReviewSection(state.backendUp);

    var flagged = (state.flaggedIds || []).length;
    chrome.runtime.sendMessage({ type: "LAUNCHER_BADGE", count: flagged })
      .catch(function () {});
    announce("Career timeline loaded for " + (r.candidate.name || "candidate") + ". " + trust.label + ".");
  }

  function renderSnapshot(r, jobs, trust, manual) {
    el.snapshot.hidden = false;
    el.name.textContent = (r.candidate && r.candidate.name) || "Unnamed candidate";

    // Current / latest role: most recent qualifying job.
    var q = qualifyingJobs(jobs).sort(function (a, b) {
      return (a.start_date || "") < (b.start_date || "") ? 1 : -1;
    });
    el.snapRole.innerHTML = "";
    el.snapRole.hidden = true;
    var current = q.length ? q[0] : null;
    if (current) {
      var rtxt = current.title || "Title not found";
      var bits = [];
      bits.push(esc(rtxt));
      if (current.organization) bits.push(esc(current.organization));
      var prec = T.precisionOf(state.tlByEntry[current.entry_id], current.raw_range);
      var when = current.is_ongoing
        ? ("since " + fmtMonthYear(current.start_date))
        : T.fmtRange(current, prec).text;
      bits.push(esc(when));
      el.snapRole.innerHTML = bits.join(" · ");
      el.snapRole.hidden = false;
      var ak = anchorInfo(current);
      if (ak.kind !== "none") {
        el.snapRole.appendChild(document.createTextNode(" "));
        el.snapRole.appendChild(showButton(current, ak, true));
      }
      if (!current.title || !current.organization) {
        var miss = document.createElement("span");
        miss.className = "reason";
        miss.textContent = !current.title ? "Title not found" : "Employer not found";
        el.snapRole.appendChild(document.createTextNode(" "));
        el.snapRole.appendChild(miss);
      }
    }

    // Totals only from qualifying jobs; hidden entirely when manual.
    el.snapTotals.hidden = true;
    el.tenureBar.innerHTML = "";
    el.tenureBar.hidden = true;
    if (!manual) {
      var totals = T.snapshotTotals(jobs);
      if (totals.computable && totals.totalMonths > 0) {
        el.snapTotals.hidden = false;
        el.snapTotals.textContent = T.durationWords(totals.totalMonths) + " experience · " +
          totals.employerCount + " employer" + (totals.employerCount === 1 ? "" : "s");
        if (totals.employers.length) {
          el.tenureBar.hidden = false;
          var max = totals.totalMonths;
          totals.employers.forEach(function (emp) {
            var seg = document.createElement("button");
            seg.type = "button";
            seg.style.width = Math.max(2, Math.round(100 * emp.months / max)) + "%";
            seg.setAttribute("aria-label", emp.name + ", " + T.durationWords(emp.months));
            seg.addEventListener("mouseenter", function () { showBarTip(seg, emp); });
            seg.addEventListener("focus", function () { showBarTip(seg, emp); });
            seg.addEventListener("mouseleave", hideBarTip);
            seg.addEventListener("blur", hideBarTip);
            seg.addEventListener("click", function () { openTimeline(); });
            el.tenureBar.appendChild(seg);
          });
        }
      } else {
        el.snapTotals.hidden = false;
        el.snapTotals.textContent = "Experience couldn't be calculated";
      }
    }
  }

  var barTipEl = null;
  function showBarTip(seg, emp) {
    hideBarTip();
    barTipEl = document.createElement("span");
    barTipEl.className = "bartip";
    barTipEl.textContent = emp.name + " · " + T.durationWords(emp.months);
    el.tenureBar.style.position = "relative";
    el.tenureBar.appendChild(barTipEl);
    var br = el.tenureBar.getBoundingClientRect(), sr = seg.getBoundingClientRect();
    barTipEl.style.left = Math.max(0, Math.min(sr.left - br.left, br.width - 40)) + "px";
    barTipEl.style.top = "-24px";
  }
  function hideBarTip() {
    if (barTipEl && barTipEl.parentNode) barTipEl.parentNode.removeChild(barTipEl);
    barTipEl = null;
  }

  /* ---------------- timeline ---------------- */

  function precFor(ev) {
    return T.precisionOf(state.tlByEntry[ev.entry_id], ev.raw_range);
  }

  function openTimeline() {
    if (el.timelineWrap.hidden) el.timelineToggle.click();
  }

  function renderTimeline(r, jobs) {
    el.timeline.innerHTML = "";
    el.eduList.innerHTML = "";
    el.otherList.innerHTML = "";
    state.flaggedIds = [];

    // Partition dated jobs through the sanity gate. Double-invalid rows
    // and dated non-job blocks move to "Other text found".
    var kept = [], other = [];
    (jobs || []).forEach(function (ev) {
      var g = T.sanitizeRow(ev);
      if (!g) { other.push({ ev: ev }); return; }
      kept.push({ ev: ev, title: g.title, employer: g.employer,
                  uncertain: g.uncertain, reasons: g.reasons });
    });
    (r.events || []).forEach(function (ev) {
      if (T.entryType(ev.section) === "unclassified") other.push({ ev: ev });
    });
    kept.sort(function (a, b) {
      return (a.ev.start_date || "") < (b.ev.start_date || "") ? 1 : -1;
    });

    // Dated education rows (invalid titles move to Other as well).
    var eduItems = [];
    (r.events || []).forEach(function (e) {
      if (T.entryType(e.section) !== "education" || !e.start_date) return;
      if (T.isValidTitle(e.title)) eduItems.push(e);
      else other.push({ ev: e });
    });
    eduItems.sort(function (a, b) {
      return (a.start_date || "") < (b.start_date || "") ? 1 : -1;
    });

    // Uncertainty: a small amber dot on uncertain rows.
    var showDots = kept.some(function (k) { return k.uncertain; }) ||
      eduItems.some(function (e) {
        return Number(e.confidence) < T.CONF_MEDIUM || precFor(e).inferred;
      });
    kept.forEach(function (k) {
      if (k.uncertain) state.flaggedIds.push("row-" + k.ev.entry_id);
    });
    eduItems.forEach(function (e) {
      if (Number(e.confidence) < T.CONF_MEDIUM || precFor(e).inferred) {
        state.flaggedIds.push("row-" + e.entry_id);
      }
    });

    // Inline gaps between consecutive trusted rows in display order.
    var trustedIds = {};
    kept.forEach(function (k) {
      if (Number(k.ev.confidence) >= T.CONF_MEDIUM) trustedIds[k.ev.entry_id] = true;
    });
    var withTl = kept.filter(function (k) { return trustedIds[k.ev.entry_id]; }).map(function (k) {
      var c = {};
      for (var key in k.ev) c[key] = k.ev[key];
      c._timeline = state.tlByEntry[k.ev.entry_id];
      return c;
    }).sort(function (a, b) { return (a.start_date || "") < (b.start_date || "") ? -1 : 1; });
    var gaps = T.computeGaps(withTl);
    function gapAfter(newerId, olderId) {
      if (!trustedIds[newerId] || !trustedIds[olderId]) return null;
      for (var i = 0; i < gaps.length; i++) {
        if (gaps[i].beforeEntryId === newerId && gaps[i].afterEntryId === olderId) return gaps[i];
      }
      return null;
    }

    // Group by employer only with valid employers and a real multi-role employer.
    var empCount = {};
    kept.forEach(function (k) {
      if (k.employer) empCount[k.employer.toLowerCase()] = (empCount[k.employer.toLowerCase()] || 0) + 1;
    });
    var grouped = Object.keys(empCount).some(function (k) { return empCount[k] >= 2; });
    var groups = {}, groupOrder = [];
    if (grouped) {
      kept.forEach(function (k) {
        if (!k.employer) return;
        var key = k.employer.toLowerCase();
        if (!groups[key]) { groups[key] = { name: k.employer, roles: [] }; groupOrder.push(key); }
        groups[key].roles.push(k);
      });
    }
    var prevEv = null;
    function emitGap(container, newerEv, olderEv) {
      var gp = newerEv && olderEv ? gapAfter(newerEv.entry_id, olderEv.entry_id) : null;
      if (gp) container.appendChild(renderGapRow(gp));
    }
    groupOrder.forEach(function (key) {
      var g = groups[key];
      var gEl = document.createElement("li");
      gEl.className = "emp";
      var head = document.createElement("div");
      head.className = "emp-head";
      var nameSpan = document.createElement("span");
      nameSpan.className = "emp-name";
      nameSpan.textContent = g.name;
      head.appendChild(nameSpan);
      var empMonths = T.unionMonths(g.roles
        .filter(function (x) { return Number(x.ev.confidence) >= T.CONF_MEDIUM && x.ev.start_date; })
        .map(function (x) {
          return { start: x.ev.start_date,
                   end: x.ev.is_ongoing ? state.referenceDate : (x.ev.end_date || x.ev.start_date) };
        }));
      if (empMonths > 0) {
        var t = document.createElement("span");
        t.className = "emp-tenure";
        t.textContent = " · " + T.durationWords(empMonths);
        head.appendChild(t);
      }
      gEl.appendChild(head);
      var list = document.createElement("ol");
      list.className = "roles";
      g.roles.forEach(function (k) {
        emitGap(list, prevEv, k.ev);
        list.appendChild(renderJobRow(k, showDots));
        prevEv = k.ev;
      });
      gEl.appendChild(list);
      el.timeline.appendChild(gEl);
    });
    // Ungrouped rows (flat mode, or valid singletons/invalid employers in grouped mode).
    var groupedIds = {};
    groupOrder.forEach(function (key) {
      groups[key].roles.forEach(function (k) { groupedIds[k.ev.entry_id] = true; });
    });
    kept.forEach(function (k) {
      if (groupedIds[k.ev.entry_id]) return;
      emitGap(el.timeline, prevEv, k.ev);
      var li = renderJobRow(k, showDots);
      el.timeline.appendChild(li);
      prevEv = k.ev;
    });

    // Education group (dated only).
    el.eduToggle.hidden = !eduItems.length;
    el.eduCount.textContent = eduItems.length ? "· " + eduItems.length : "";
    eduItems.forEach(function (it) {
      if (Number(it.confidence) < T.CONF_MEDIUM || precFor(it).inferred) {
        state.flaggedIds.push("row-" + it.entry_id);
      }
      el.eduList.appendChild(renderEduRow(it, showDots));
    });

    // Other text found (collapsed).
    el.otherToggle.hidden = !other.length;
    el.otherCount.textContent = other.length ? "· " + other.length : "";
    other.forEach(function (o) { el.otherList.appendChild(renderOtherRow(o)); });

    var hasContent = kept.length || eduItems.length || other.length;
    el.empty.hidden = !!hasContent;
    if (!hasContent) {
      el.snapshot.hidden = true;
      el.timelineWrap.hidden = true;
    }
  }

  function rowIdFor(it) {
    return "row-" + (it.entry_id || it.gap_id || Math.random().toString(36).slice(2));
  }

  function anchorInfo(it) {
    return T.anchorKind(it.evidence, state.linesById);
  }

  function showButton(it, ak, always) {
    var b = document.createElement("button");
    b.type = "button";
    b.className = "showlink" + (always ? " alwaysshow" : "");
    b.setAttribute("aria-label", "Show in resume");
    b.title = "Show in resume";
    b.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">' +
      '<circle cx="11" cy="11" r="7"/><line x1="16.5" y1="16.5" x2="21" y2="21"/></svg>' +
      '<span aria-hidden="true">Show in resume</span>';
    b.addEventListener("click", function (ev) {
      ev.stopPropagation();
      jumpNow(it, ak);
    });
    return b;
  }

  function magIcon() {
    var s = document.createElement("span");
    s.className = "mag";
    s.setAttribute("aria-hidden", "true");
    s.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">' +
      '<circle cx="11" cy="11" r="7"/><line x1="16.5" y1="16.5" x2="21" y2="21"/></svg>';
    return s;
  }

  function inlineMessage(li, text) {
    var old = li.querySelector(".inline-msg");
    if (old) old.parentNode.removeChild(old);
    var p = document.createElement("p");
    p.className = "inline-msg";
    p.textContent = text;
    li.appendChild(p);
  }

  function markShowing(li) {
    var prev = document.querySelector("#timeline li.showing, #eduList li.showing, #otherList li.showing");
    if (prev) prev.classList.remove("showing");
    if (li) li.classList.add("showing");
  }

  function navigateToAnchor(it, ak) {
    ak = ak || anchorInfo(it);
    var li = document.getElementById(rowIdFor(it));
    if (ak.kind === "none") {
      if (li) inlineMessage(li, "Couldn't locate this in the resume");
      announce("Couldn't locate this in the resume.");
      return;
    }
    var texts = [];
    if (ak.kind === "text") {
      var phrases = fragmentPhrases(it);
      phrases.forEach(function (p) { texts.push(p); });
      if (it.evidence && it.evidence.raw_range) texts.unshift(it.evidence.raw_range);
    }
    var done = function (ok, viaPage) {
      if (ok) {
        if (li) markShowing(li);
        announce(viaPage ? "Resume opened at the matching passage." : "Showing in resume.");
      } else if (ak.kind === "text" && ak.pages && ak.pages.length) {
        pageJump(ak.pages[0], function (ok2) {
          if (ok2) {
            if (li) markShowing(li);
            announce("Resume opened at the matching passage.");
          } else {
            if (li) inlineMessage(li, "Couldn't locate this in the resume");
            announce("Couldn't locate this in the resume.");
          }
        }, pagePhrases(it));
      } else {
        if (li) inlineMessage(li, "Couldn't locate this in the resume");
        announce("Couldn't locate this in the resume.");
      }
    };
    if (ak.kind === "text" && texts.length) {
      sendToTab(texts, done);
    } else if (ak.pages && ak.pages.length) {
      pageJump(ak.pages[0], function (ok) { done(ok, true); }, pagePhrases(it));
    } else {
      done(false);
    }
  }

  /* All fragment phrases for a page jump: exact source-line quote first,
     then header/date/excerpt anchors. The viewer tries each in order. */
  function pagePhrases(it) {
    var out = [];
    var q = anchorQuote(it);
    if (q) out.push(q);
    fragmentPhrases(it).forEach(function (p) {
      if (out.indexOf(p) === -1) out.push(p);
    });
    return out;
  }

  /* Gesture-preserving jump: runs synchronously inside the click handler so
     the browser keeps the text fragment. Falls back to the async path. */
  function jumpNow(it, ak) {
    ak = ak || anchorInfo(it);
    var cached = state.activeTab;
    if (cached && cached.url && /^https?:|^file:/i.test(cached.url) && cached.id != null &&
        ak.kind !== "none") {
      var page = (ak.pages && ak.pages.length) ? ak.pages[0] : null;
      var base = cached.url.split("#")[0];
      var frag = page != null ? "#page=" + (page + 1) : "#";
      pagePhrases(it).forEach(function (p, i) {
        frag += (i === 0 ? ":~:text=" : "&text=") + encodeURIComponent(p);
      });
      try {
        chrome.tabs.update(cached.id, { url: base + frag }, function () {
          var li = document.getElementById(rowIdFor(it));
          if (chrome.runtime.lastError) { navigateToAnchor(it, ak); return; }
          if (li) markShowing(li);
          announce("Resume opened at the matching passage.");
        });
        return;
      } catch (e) { /* fall through to async path */ }
    }
    navigateToAnchor(it, ak);
  }

  /* Best exact-portion quote for a text fragment: cleaned words from the
     evidence's own first substantial source line (closest to the viewer's
     text layer — markdown/table artifacts stripped). Falls back to the date
     range as written. */
  function anchorQuote(it) {
    var ev = it.evidence || {};
    var ids = ev.line_ids || [];
    for (var i = 0; i < ids.length; i++) {
      var ln = state.linesById[ids[i]];
      var t = ln && ln.display_text ? String(ln.display_text) : "";
      var words = [];
      if (typeof wordsOf === "function") {
        words = wordsOf(t, 12);
      } else {
        words = t.replace(/\s+/g, " ").trim().split(" ").slice(0, 12);
      }
      if (words.length >= 3 && /[a-zA-Z]{3,}/.test(words.join(" "))) {
        return words.join(" ");
      }
    }
    if (ev.raw_range) return String(ev.raw_range).trim();
    return null;
  }

  function sendToTab(texts, cb) {
    function send(afterInject) {
      try {
        if (typeof chrome === "undefined" || !chrome.tabs) return cb(false);
        chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
          if (!tabs || !tabs[0] || tabs[0].id == null) return cb(false);
          chrome.tabs.sendMessage(tabs[0].id,
            { type: "HIGHLIGHT_RESUME_EVIDENCE", texts: texts }, function (resp) {
              if (chrome.runtime.lastError) return afterInject ? cb(false) : inject(tabs[0].id);
              cb(!!(resp && resp.found));
            });
        });
      } catch (e) { cb(false); }
    }
    function inject(tabId) {
      if (!chrome.scripting) return cb(false);
      try {
        chrome.scripting.executeScript({ target: { tabId: tabId }, files: ["content.js"] }, function () {
          if (chrome.runtime.lastError) return cb(false);
          send(true);
        });
      } catch (e) { cb(false); }
    }
    // Prefer the cached tab when available (keeps the click gesture).
    try {
      if (state.activeTab && state.activeTab.id != null && chrome.tabs && chrome.tabs.sendMessage) {
        chrome.tabs.sendMessage(state.activeTab.id,
          { type: "HIGHLIGHT_RESUME_EVIDENCE", texts: texts }, function (resp) {
            if (chrome.runtime.lastError || !(resp && resp.found)) { send(false); return; }
            cb(true);
          });
        return;
      }
    } catch (e) {}
    send(false);
  }

  function pageJump(pageIndex, cb, quotes) {
    try {
      if (typeof chrome === "undefined" || !chrome.tabs) return cb(false);
      var tabId = (state.activeTab && state.activeTab.id != null) ? state.activeTab.id : null;
      var go = function (id, url) {
        if (url == null || !/^https?:|^file:/i.test(url)) return cb(false);
        var base = url.split("#")[0];
        var frag = "#page=" + (pageIndex + 1);
        (quotes || []).forEach(function (q, i) {
          frag += (i === 0 ? ":~:text=" : "&text=") + encodeURIComponent(q);
        });
        chrome.tabs.update(id, { url: base + frag }, function () {
          cb(!(chrome.runtime.lastError));
        });
      };
      if (tabId != null) {
        chrome.tabs.get(tabId, function (tab) {
          if (chrome.runtime.lastError || !tab) return cb(false);
          go(tabId, tab.url);
        });
      } else {
        chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
          if (!tabs || !tabs[0] || tabs[0].id == null) return cb(false);
          go(tabs[0].id, tabs[0].url);
        });
      }
    } catch (e) { cb(false); }
  }

  function normEq(a, b) {
    function norm(s) { return String(s || "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim(); }
    var na = norm(a), nb = norm(b);
    return !!na && !!nb && (na === nb || na.indexOf(nb) === 0 || nb.indexOf(na) === 0);
  }

  function datesDuration(ev) {
    var prec = precFor(ev);
    var fr = T.fmtRange(ev, prec);
    var endIso = ev.is_ongoing ? state.referenceDate : (ev.end_date || ev.start_date);
    var dur = ev.start_date ? T.durationWords(T.monthsBetween(
      ev.start_date.slice(0, 7) + "-01", (endIso || ev.start_date).slice(0, 7) + "-01")) : "";
    return fr.text + (dur ? " · " + dur : "");
  }

  /* Two-line job row. Line 1: employer if valid, else title. Line 2: title
     (only if valid and different) · city · dates · duration. Whole row
     clickable; magnifier icon on hover/focus only; static when no anchor. */
  function renderJobRow(k, showDots) {
    var ev = k.ev;
    var li = document.createElement("li");
    li.id = rowIdFor(ev);
    li.className = "titem";
    var line1 = k.employer || k.title;
    var line2 = [];
    if (k.title && k.title !== line1) line2.push(k.title);
    var city = ev.location ? String(ev.location).trim() : "";
    if (city && !normEq(city, line1) && !normEq(city, k.title)) line2.push(city);
    line2.push(datesDuration(ev));
    var ak = anchorInfo(ev);
    var clickable = ak.kind !== "none";
    var row = clickable ? document.createElement("button") : document.createElement("div");
    if (clickable) row.type = "button";
    row.className = "trowmain" + (clickable ? "" : " trowstatic");
    var text = document.createElement("span");
    text.className = "trowtext";
    var l1 = document.createElement("span");
    l1.className = "ttitle";
    l1.textContent = line1;
    text.appendChild(l1);
    var l2 = document.createElement("span");
    l2.className = "tsub";
    l2.textContent = line2.join(" · ");
    text.appendChild(l2);
    row.appendChild(text);
    if (k.uncertain && showDots) {
      var dot = document.createElement("span");
      dot.className = "amberdot";
      dot.title = k.reasons.join("; ");
      dot.setAttribute("aria-label", k.reasons.join("; "));
      row.appendChild(dot);
    }
    if (clickable) {
      row.appendChild(magIcon());
      row.setAttribute("aria-label", line1 + ", " + line2.join(" · ") + ". Activate to show in resume.");
      row.addEventListener("click", function () { jumpNow(ev, ak); });
    }
    li.appendChild(row);
    return li;
  }

  /* Dated education rows use the same two-line pattern. */
  function renderEduRow(it, showDots) {
    var li = document.createElement("li");
    li.id = rowIdFor(it);
    li.className = "titem";
    var reasons = [];
    if (Number(it.confidence) < T.CONF_MEDIUM) reasons.push("Low confidence");
    if (precFor(it).inferred) reasons.push("Dates estimated");
    var org = (it.organization && T.isValidEmployer(it.organization, it.location))
      ? String(it.organization).trim() : null;
    var line2 = [];
    if (org) line2.push(org);
    line2.push(datesDuration(it));
    var ak = anchorInfo(it);
    var clickable = ak.kind !== "none";
    var row = clickable ? document.createElement("button") : document.createElement("div");
    if (clickable) row.type = "button";
    row.className = "trowmain" + (clickable ? "" : " trowstatic");
    var text = document.createElement("span");
    text.className = "trowtext";
    var l1 = document.createElement("span");
    l1.className = "ttitle";
    l1.textContent = it.title;
    text.appendChild(l1);
    var l2 = document.createElement("span");
    l2.className = "tsub";
    l2.textContent = line2.join(" · ");
    text.appendChild(l2);
    row.appendChild(text);
    if (reasons.length && showDots) {
      var dot = document.createElement("span");
      dot.className = "amberdot";
      dot.title = reasons.join("; ");
      dot.setAttribute("aria-label", reasons.join("; "));
      row.appendChild(dot);
    }
    if (reasons.length) state.flaggedIds.push(li.id);
    if (clickable) {
      row.appendChild(magIcon());
      row.setAttribute("aria-label", it.title + ". Activate to show in resume.");
      row.addEventListener("click", function () { jumpNow(it, ak); });
    }
    li.appendChild(row);
    return li;
  }

  /* "Other text found" rows: dates plus a muted label only. Invalid fields
     are never displayed. Clickable when an anchor exists. */
  function renderOtherRow(o) {
    var ev = o.ev;
    var li = document.createElement("li");
    li.id = rowIdFor(ev);
    li.className = "titem";
    var label = ev.raw_range || T.fmtRange(ev, precFor(ev)).text;
    var ak = anchorInfo(ev);
    var clickable = ak.kind !== "none";
    var row = clickable ? document.createElement("button") : document.createElement("div");
    if (clickable) row.type = "button";
    row.className = "trowmain" + (clickable ? "" : " trowstatic");
    var text = document.createElement("span");
    text.className = "trowtext";
    var l1 = document.createElement("span");
    l1.className = "ttitle muted";
    l1.textContent = label;
    text.appendChild(l1);
    var l2 = document.createElement("span");
    l2.className = "tsub muted";
    l2.textContent = "Unclear entry";
    text.appendChild(l2);
    row.appendChild(text);
    if (clickable) {
      row.appendChild(magIcon());
      row.setAttribute("aria-label", label + ", unclear entry. Activate to show in resume.");
      row.addEventListener("click", function () { jumpNow(ev, ak); });
    }
    li.appendChild(row);
    return li;
  }

  function renderGapRow(gap) {
    var li = document.createElement("li");
    li.className = "gapline";
    li.textContent = "Gap · " + T.durationWords(gap.months);
    return li;
  }

  /* ---------------- helpers ---------------- */

  function fmtMonthYear(iso) {
    var m = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    var p = String(iso || "").split("-");
    if (p.length < 2) return p[0] || "";
    return m[parseInt(p[1], 10) - 1] + " " + p[0];
  }

  function fmtDay(iso) {
    try {
      var d = new Date(iso);
      var m = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
      return d.getDate() + " " + m[d.getMonth()] + " " + d.getFullYear();
    } catch (e) { return iso; }
  }

  function fmtTime(iso) {
    try {
      return new Date(iso).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
    } catch (e) { return iso; }
  }

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  function announce(msg) {
    el.live.textContent = "";
    window.setTimeout(function () { el.live.textContent = msg; }, 30);
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

  /* Phrase builders reused for text-anchor resolution. */
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
    function take(words, min) {
      if (words.length >= (min || 2) && /[a-zA-Z]{3,}/.test(words.join(" ")) && out.length < 5) {
        var p = words.join(" ");
        if (out.indexOf(p) === -1) out.push(p);
      }
    }
    take(uniqueWords(wordsOf(it.title || "", 6).concat(wordsOf(it.organization || "", 6)), 6));
    take(wordsOf(ev.raw_range || "", 8));
    var firstLine = ((ev.excerpt || "").split("\n").filter(function (l) { return l.trim(); })[0] || "");
    take(wordsOf(firstLine, 8));
    var rare = "";
    wordsOf((it.title || "") + " " + (it.organization || ""), 12).forEach(function (w) {
      if (/^[A-Za-z]{7,}$/.test(w) && w.length > rare.length) rare = w;
    });
    if (rare) take([rare], 1);
    return out;
  }
})();
