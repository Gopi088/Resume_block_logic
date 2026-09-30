/* Content script: scrolls the open resume tab to the exact evidence passage
   and highlights it. Tries verbatim substrings first (raw date range, then
   the entry's first line), then whitespace/dash-normalized matching —
   never a generic section or page top. */
(function () {
  "use strict";

  chrome.runtime.onMessage.addListener(function (msg, sender, sendResponse) {
    if (!msg || msg.type !== "HIGHLIGHT_RESUME_EVIDENCE") return;
    try {
      var found = highlightFirst(msg.texts || []);
      sendResponse({ found: found });
    } catch (e) {
      sendResponse({ found: false });
    }
    return true;
  });

  function norm(s) {
    return (s || "").replace(/\s+/g, " ").replace(/[–—−]/g, "-").trim();
  }

  function highlightFirst(texts) {
    clearOld();
    var needles = [];
    texts.forEach(function (t) {
      var n = norm(t);
      if (n && needles.indexOf(n) === -1) needles.push(n);
    });
    if (!needles.length) return false;

    // Pass 1: exact substring → highlight precisely that span.
    for (var i = 0; i < needles.length; i++) {
      var node = findTextNode(document.body, needles[i], false);
      if (node && markSpan(node, node.textContent.indexOf(needles[i]), needles[i].length)) return true;
    }
    // Pass 2: normalized comparison → highlight the whole matching block.
    // (Covers dash variants, collapsed whitespace, wrapped lines.)
    for (var j = 0; j < needles.length; j++) {
      var block = findTextNode(document.body, needles[j], true);
      if (block && markWhole(block)) return true;
    }
    return false;
  }

  function findTextNode(root, needle, normalized) {
    if (!root) return null;
    var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null);
    var node;
    while ((node = walker.nextNode())) {
      var text = node.textContent || "";
      if (text.length < 4) continue;
      if (!normalized) {
        if (text.indexOf(needle) !== -1) return node;
      } else if (norm(text).indexOf(needle) !== -1) {
        return node;
      }
    }
    return null;
  }

  function markSpan(node, idx, len) {
    if (idx < 0 || len <= 0) return false;
    try {
      var range = document.createRange();
      range.setStart(node, idx);
      range.setEnd(node, idx + len);
      var mark = document.createElement("mark");
      mark.setAttribute("data-resume-evidence", "true");
      mark.style.background = "#fff3c4";
      mark.style.outline = "2px solid #b98a1d";
      range.surroundContents(mark);
      if (mark.scrollIntoView) mark.scrollIntoView({ block: "center", behavior: "auto" });
      return true;
    } catch (e) {
      return markWhole(node);
    }
  }

  function markWhole(node) {
    try {
      var parent = node.parentNode;
      if (!parent || /^(SCRIPT|STYLE)$/.test(parent.tagName)) return false;
      var range = document.createRange();
      range.selectNodeContents(node);
      var mark = document.createElement("mark");
      mark.setAttribute("data-resume-evidence", "true");
      mark.style.background = "#fff3c4";
      mark.style.outline = "2px solid #b98a1d";
      range.surroundContents(mark);
      if (mark.scrollIntoView) mark.scrollIntoView({ block: "center", behavior: "auto" });
      return true;
    } catch (e) {
      return false;
    }
  }

  function clearOld() {
    document.querySelectorAll("mark[data-resume-evidence]").forEach(function (m) {
      var parent = m.parentNode;
      if (!parent) return;
      parent.replaceChild(document.createTextNode(m.textContent), m);
      parent.normalize();
    });
  }
})();
