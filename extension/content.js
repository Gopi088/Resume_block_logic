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

// Floating Action Button to open the side panel
(function () {
  "use strict";

  function createFab() {
    if (document.getElementById("resume-review-fab")) return;

    var fab = document.createElement("button");
    fab.id = "resume-review-fab";
    fab.title = "Open Resume Review";
    fab.setAttribute("aria-label", "Open Resume Review side panel");
    fab.innerHTML = `
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
        <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
      </svg>
    `;

    // Styles for the FAB
    var style = document.createElement("style");
    style.textContent = `
      #resume-review-fab {
        position: fixed;
        bottom: 24px;
        right: 24px;
        width: 56px;
        height: 56px;
        border-radius: 50%;
        background: #2f4b7c;
        color: #fff;
        border: none;
        box-shadow: 0 4px 12px rgba(47, 75, 124, 0.4), 0 2px 4px rgba(47, 75, 124, 0.3);
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 2147483647;
        transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
      }
      #resume-review-fab:hover {
        transform: scale(1.08);
        box-shadow: 0 6px 20px rgba(47, 75, 124, 0.5), 0 4px 8px rgba(47, 75, 124, 0.4);
        background: #253d66;
      }
      #resume-review-fab:active {
        transform: scale(0.95);
      }
      #resume-review-fab:focus {
        outline: 3px solid #2f4b7c;
        outline-offset: 3px;
      }
      #resume-review-fab svg {
        width: 24px;
        height: 24px;
      }
      @media (prefers-reduced-motion: reduce) {
        #resume-review-fab {
          transition: none;
        }
      }
      /* Ensure FAB stays on top of everything */
      #resume-review-fab {
        position: fixed !important;
      }
    `;
    document.head.appendChild(style);
    document.body.appendChild(fab);

    fab.addEventListener("click", function () {
      // Send message to background script to open side panel
      if (typeof chrome !== "undefined" && chrome.runtime) {
        chrome.runtime.sendMessage({ type: "OPEN_SIDE_PANEL" });
      }
    });

    // Keyboard accessibility
    fab.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        fab.click();
      }
    });
  }

  // Create FAB when DOM is ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", createFab);
  } else {
    createFab();
  }

  // Also recreate if page navigates (SPA support)
  var observer = new MutationObserver(function () {
    if (!document.getElementById("resume-review-fab")) {
      createFab();
    }
  });
  observer.observe(document.body, { childList: true, subtree: true });
})();
