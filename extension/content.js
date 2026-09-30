/* Content script: evidence highlighting + simple Career Timeline launcher.
   The launcher is a single clean button that opens the side panel. */
(function () {
  "use strict";

  /* ──────────────────────────────────────────────────────────────
     Evidence highlighting (unchanged)
  ────────────────────────────────────────────────────────────── */
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

    for (var i = 0; i < needles.length; i++) {
      var node = findTextNode(document.body, needles[i], false);
      if (node && markSpan(node, node.textContent.indexOf(needles[i]), needles[i].length)) return true;
    }
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

  /* ──────────────────────────────────────────────────────────────
     Simple Career Timeline Launcher (Shadow DOM isolated)
  ────────────────────────────────────────────────────────────── */
  (function () {
    // Timeline icon: horizontal line with three ascending dots
    var ICON_TIMELINE = `
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M4 12h16" stroke="currentColor" stroke-width="2"/>
        <circle cx="6" cy="12" r="3" fill="currentColor"/>
        <circle cx="12" cy="10" r="3" fill="currentColor"/>
        <circle cx="18" cy="8" r="3" fill="currentColor"/>
      </svg>
    `;

    // Host element in light DOM
    var host = document.createElement("div");
    host.id = "career-timeline-launcher-host";
    host.style.cssText = "all: initial; position: fixed; bottom: 24px; right: 24px; z-index: 2147483647;";

    // Shadow DOM for style isolation
    var shadow = host.attachShadow({ mode: "closed" });

    // Styles inside Shadow DOM
    var style = document.createElement("style");
    style.textContent = `
      :host { all: initial; display: block; }

      :root {
        --ct-brand: #2563EB;
        --ct-brand-hover: #1D4ED8;
        --ct-white: #FFFFFF;
        --ct-outer-ring: #FFFFFF;
        --ct-shadow: rgba(0, 0, 0, 0.35);
        --ct-size: 48px;
        --ct-icon-size: 22px;
        --ct-radius: 50%;
        --ct-transition: 150ms;
      }

      .ct-btn {
        all: initial;
        display: flex;
        align-items: center;
        justify-content: center;
        width: var(--ct-size);
        height: var(--ct-size);
        border-radius: var(--ct-radius);
        background: var(--ct-brand);
        color: var(--ct-white);
        border: 2px solid var(--ct-outer-ring);
        box-shadow: 0 2px 8px var(--ct-shadow);
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: background-color 120ms ease, transform 100ms ease, box-shadow 120ms ease;
        outline: none;
      }
      .ct-btn:focus-visible {
        outline: 3px solid var(--ct-brand);
        outline-offset: 3px;
      }
      .ct-btn:hover { background: #1D4ED8; transform: scale(1.05); }
      .ct-btn:active { background: #1E40AF; transform: scale(0.97); }

      /* White outer ring for contrast on any background */
      .ct-btn::before {
        content: "";
        position: absolute;
        inset: -2px;
        border-radius: inherit;
        border: 2px solid var(--ct-outer-ring);
        pointer-events: none;
        z-index: -1;
      }

      .ct-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 22px;
        height: 22px;
        color: #FFFFFF;
      }
      .ct-icon svg { width: 100%; height: 100%; }

      @media (prefers-reduced-motion: reduce) {
        .ct-btn { transition: none !important; }
      }
    `;
    shadow.appendChild(style);

    // Button element
    var btn = document.createElement("button");
    btn.className = "ct-btn";
    btn.type = "button";
    btn.setAttribute("aria-label", "Open Career Timeline");
    btn.title = "Open Career Timeline";

    var iconWrap = document.createElement("span");
    iconWrap.className = "ct-icon";
    iconWrap.innerHTML = ICON_TIMELINE;

    btn.appendChild(iconWrap);
    shadow.appendChild(btn);

    // Click → open side panel
    btn.addEventListener("click", function () {
      if (typeof chrome !== "undefined" && chrome.runtime) {
        chrome.runtime.sendMessage({ type: "OPEN_SIDE_PANEL" });
      }
    });

    // Keyboard
    btn.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        btn.click();
      }
    });

    // Init
    function init() {
      document.body.appendChild(host);
    }

    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", init);
    } else {
      init();
    }

    // SPA survival
    var observer = new MutationObserver(function () {
      if (!document.getElementById("career-timeline-launcher-host")) {
        init();
      }
    });
    observer.observe(document.body, { childList: true, subtree: true });
  })();

  /* ──────────────────────────────────────────────────────────────
     Evidence highlighting (original logic preserved)
  ────────────────────────────────────────────────────────────── */
  function norm(s) { return (s || "").replace(/\s+/g, " ").replace(/[–—−]/g, "-").trim(); }
  function highlightFirst(texts) { clearOld(); var needles=[]; texts.forEach(function(t){var n=norm(t);if(n&&needles.indexOf(n)===-1)needles.push(n);}); if(!needles.length)return false; for(var i=0;i<needles.length;i++){var node=findTextNode(document.body,needles[i],false);if(node&&markSpan(node,node.textContent.indexOf(needles[i]),needles[i].length))return true;} for(var j=0;j<needles.length;j++){var block=findTextNode(document.body,needles[j],true);if(block&&markWhole(block))return true;} return false; }
  function findTextNode(root,needle,normalized){if(!root)return null;var walker=document.createTreeWalker(root,NodeFilter.SHOW_TEXT,null);var node;while((node=walker.nextNode())){var text=node.textContent||"";if(text.length<4)continue;if(!normalized){if(text.indexOf(needle)!==-1)return node;}else if(norm(text).indexOf(needle)!==-1){return node;}}return null;}
  function markSpan(node,idx,len){if(idx<0||len<=0)return false;try{var range=document.createRange();range.setStart(node,idx);range.setEnd(node,idx+len);var mark=document.createElement("mark");mark.setAttribute("data-resume-evidence","true");mark.style.background="#fff3c4";mark.style.outline="2px solid #b98a1d";range.surroundContents(mark);if(mark.scrollIntoView)mark.scrollIntoView({block:"center",behavior:"auto"});return true;}catch(e){return markWhole(node);}}
  function markWhole(node){try{var parent=node.parentNode;if(!parent||/^(SCRIPT|STYLE)$/.test(parent.tagName))return false;var range=document.createRange();range.selectNodeContents(node);var mark=document.createElement("mark");mark.setAttribute("data-resume-evidence","true");mark.style.background="#fff3c4";mark.style.outline="2px solid #b98a1d";range.surroundContents(mark);if(mark.scrollIntoView)mark.scrollIntoView({block:"center",behavior:"auto"});return true;}catch(e){return false;}}
  function clearOld(){document.querySelectorAll("mark[data-resume-evidence]").forEach(function(m){var p=m.parentNode;if(!p)return;p.replaceChild(document.createTextNode(m.textContent),m);p.normalize();});}
})();