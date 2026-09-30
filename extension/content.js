/* Content script: scrolls the open resume tab to the exact evidence passage
   and highlights it. Searches verbatim substrings only (raw date range first,
   then the entry's first line) — never a generic section or page top. */
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

  function highlightFirst(texts) {
    clearOld();
    var bodyText = document.body ? document.body.innerText || "" : "";
    for (var i = 0; i < texts.length; i++) {
      var needle = (texts[i] || "").trim();
      if (!needle || bodyText.indexOf(needle) === -1) continue;
      var node = findTextNode(document.body, needle);
      if (!node) continue;
      var range = document.createRange();
      var idx = node.textContent.indexOf(needle);
      range.setStart(node, idx);
      range.setEnd(node, idx + needle.length);
      var mark = document.createElement("mark");
      mark.setAttribute("data-resume-evidence", "true");
      mark.style.background = "#fff3c4";
      mark.style.outline = "2px solid #b98a1d";
      range.surroundContents(mark);
      mark.scrollIntoView({ block: "center", behavior: "auto" });
      return true;
    }
    return false;
  }

  function findTextNode(root, needle) {
    var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null);
    var node;
    while ((node = walker.nextNode())) {
      if (node.textContent && node.textContent.indexOf(needle) !== -1) return node;
    }
    return null;
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
