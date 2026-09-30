/* Content script: evidence highlighting + Career Timeline floating action button.
   Exact implementation per design spec. */
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
     Career Timeline Floating Action Button (exact spec)
  ────────────────────────────────────────────────────────────── */
  (function () {
    // White document icon with folded corner + 3 horizontal lines
    var ICON_DOCUMENT = `
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" stroke="currentColor" stroke-width="2"/>
        <path d="M14 2v6h6" stroke="currentColor" stroke-width="2"/>
        <line x1="8" y1="10" x2="16" y2="10" stroke="currentColor" stroke-width="1.5"/>
        <line x1="8" y1="14" x2="14" y2="14" stroke="currentColor" stroke-width="1.5"/>
        <line x1="8" y1="18" x2="12" y2="18" stroke="currentColor" stroke-width="1.5"/>
      </svg>
    `;

    // Host element in light DOM
    var host = document.createElement("div");
    host.id = "career-timeline-launcher-host";
    host.style.cssText = "all: initial; position: fixed; bottom: 20px; right: 20px; z-index: 2147483647;";

    // Shadow DOM for complete style isolation
    var shadow = host.attachShadow({ mode: "closed" });

    // Styles inside Shadow DOM
    var style = document.createElement("style");
    style.textContent = `
      :host { all: initial; display: block; }

      :root {
        /* Gradient: lighter upper-left → deeper lower-right */
        --ct-grad-start: #3B82F6;
        --ct-grad-mid: #2563EB;
        --ct-grad-end: #1D4ED8;
        --ct-white: #FFFFFF;
        --ct-icon: #FFFFFF;
        --ct-ring: rgba(255, 255, 255, 0.25);
        --ct-shadow: rgba(0, 0, 0, 0.3);
        --ct-glow: rgba(59, 130, 246, 0.35);
        --ct-glow-hover: rgba(59, 130, 246, 0.5);
        --ct-badge-red: #EF4444;
        --ct-size: 64px;
        --ct-icon-size: 30px;
        --ct-radius: 50%;
        --ct-transition: 180ms;
      }

      .ct-btn {
        all: initial;
        display: flex;
        align-items: center;
        justify-content: center;
        width: var(--ct-size);
        height: var(--ct-size);
        border-radius: 50%;
        background: linear-gradient(135deg, var(--ct-grad-start) 0%, var(--ct-grad-mid) 50%, var(--ct-grad-end) 100%);
        color: var(--ct-icon);
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: 
          transform 150ms cubic-bezier(0.25, 0.46, 0.45, 0.94),
          box-shadow 150ms ease,
          filter 150ms ease;
        outline: none;
        border: none;
        padding: 0;
        position: relative;
      }
      .ct-btn:focus-visible {
        outline: 2px solid var(--ct-grad-mid);
        outline-offset: 3px;
      }
      .ct-btn:hover {
        filter: brightness(1.08);
        transform: scale(1.04);
      }
      .ct-btn:active {
        filter: brightness(0.92);
        transform: scale(0.98);
      }

      /* Subtle translucent outer ring - visible on both light/dark */
      .ct-btn::before {
        content: "";
        position: absolute;
        inset: -2px;
        border-radius: 50%;
        border: 1px solid var(--ct-ring);
        pointer-events: none;
        opacity: 0.6;
      }

      /* Soft shadow + restrained blue ambient glow */
      .ct-btn {
        box-shadow: 
          0 4px 16px var(--ct-shadow),
          0 0 0 1px rgba(255,255,255,0.08) inset,
          0 0 24px var(--ct-glow);
      }
      .ct-btn:hover {
        box-shadow: 
          0 6px 20px var(--ct-shadow),
          0 0 0 1px rgba(255,255,255,0.12) inset,
          0 0 32px var(--ct-glow-hover);
      }
      .ct-btn:active {
        box-shadow: 
          0 2px 8px var(--ct-shadow),
          0 0 0 1px rgba(255,255,255,0.1) inset,
          0 0 16px var(--ct-glow);
      }

      /* Restrained ambient pulse - very subtle, professional */
      @keyframes ct-ambient-glow {
        0%, 100% { box-shadow: 0 4px 16px var(--ct-shadow), 0 0 0 1px rgba(255,255,255,0.08) inset, 0 0 24px var(--ct-glow); }
        50% { box-shadow: 0 4px 16px var(--ct-shadow), 0 0 0 1px rgba(255,255,255,0.12) inset, 0 0 30px var(--ct-glow); }
      }
      .ct-btn { animation: ct-ambient-glow 4s ease-in-out infinite; }

      .ct-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 26px;
        height: 26px;
        color: var(--ct-icon);
        filter: drop-shadow(0 1px 2px rgba(0,0,0,0.15));
      }
      .ct-icon svg { width: 100%; height: 100%; }

      /* Notification badge - red, upper-right */
      .ct-badge {
        position: absolute;
        top: -4px;
        right: -4px;
        min-width: 18px;
        height: 18px;
        border-radius: 9px;
        background: #EF4444;
        color: #FFFFFF;
        font: 600 11px/18px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        text-align: center;
        padding: 0 5px;
        border: 2px solid #FFFFFF;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        display: none;
        font-feature-settings: "tnum";
      }
      .ct-badge.show { display: flex; align-items: center; justify-content: center; }

      /* Tooltip (in light DOM, not shadow) */
      #ct-tooltip {
        position: fixed;
        z-index: 2147483646;
        background: #1E1E2E;
        color: #FFFFFF;
        font: 500 12px/1 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        padding: 6px 10px;
        border-radius: 6px;
        max-width: 220px;
        opacity: 0;
        pointer-events: none;
        transition: opacity 120ms ease;
        white-space: nowrap;
        box-shadow: 0 4px 16px rgba(0,0,0,0.25);
      }
      #ct-tooltip::after {
        content: "";
        position: absolute;
        width: 0; height: 0;
        border: 6px solid transparent;
        border-top-color: #1E1E2E;
        bottom: -12px;
        left: 50%;
        transform: translateX(-50%);
      }

      .ct-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        color: #FFFFFF;
        filter: drop-shadow(0 1px 2px rgba(0,0,0,0.15));
      }
      .ct-icon svg { width: 100%; height: 100%; }

      @media (prefers-reduced-motion: reduce) {
        .ct-btn { animation: none !important; transition: none !important; }
      }
    `;
    shadow.appendChild(style);

    /* ────────────────── Button element ────────────────── */
    var btn = document.createElement("button");
    btn.className = "ct-btn";
    btn.type = "button";
    btn.setAttribute("aria-label", "Open Resume Timeline Review");
    btn.setAttribute("aria-expanded", "false");

    var iconWrap = document.createElement("span");
    iconWrap.className = "ct-icon";
    iconWrap.innerHTML = ICON_DOCUMENT;

    // Notification badge (upper-right)
    var badge = document.createElement("span");
    badge.className = "ct-badge";
    badge.setAttribute("aria-live", "polite");
    badge.setAttribute("aria-label", "Items requiring review");

    btn.appendChild(iconWrap);
    btn.appendChild(badge);
    shadow.appendChild(btn);

    /* ────────────────── Tooltip (in light DOM) ────────────────── */
    var tooltip = null;
    var tooltipTimeout = null;

    function createTooltip() {
      if (tooltip) return tooltip;
      var el = document.createElement("div");
      el.id = "ct-tooltip";
      el.setAttribute("role", "tooltip");
      el.style.cssText = "position:fixed;z-index:2147483646;background:#1E1E2E;color:#FFFFFF;font:500 12px/1 system-ui,sans-serif;padding:6px 10px;border-radius:6px;max-width:220px;opacity:0;pointer-events:none;transition:opacity 120ms ease;white-space:nowrap;box-shadow:0 4px 16px rgba(0,0,0,0.25);";
      var arrow = document.createElement("div");
      arrow.style.cssText = "position:absolute;width:0;height:0;border:6px solid transparent;border-top-color:#1E1E2E;bottom:-12px;left:50%;transform:translateX(-50%);";
      el.appendChild(arrow);
      document.body.appendChild(el);
      tooltip = el;
      return el;
    }

    function showTooltip(text, target) {
      if (!tooltip) createTooltip();
      tooltip.textContent = text;
      var rect = target.getBoundingClientRect();
      var left = rect.left + rect.width / 2 - tooltip.offsetWidth / 2;
      var top = rect.top - tooltip.offsetHeight - 8;
      var vpW = window.innerWidth;
      if (left < 8) left = 8;
      if (left + tooltip.offsetWidth > vpW - 8) left = vpW - tooltip.offsetWidth - 8;
      if (top < 8) {
        top = rect.bottom + 8;
        tooltip.querySelector("div").style.cssText += "border-top-color:transparent;border-bottom-color:inherit;bottom:auto;top:-12px;";
      } else {
        tooltip.querySelector("div").style.cssText = "border-top-color:inherit;border-bottom-color:transparent;bottom:-12px;top:auto;";
      }
      tooltip.style.left = left + "px";
      tooltip.style.top = top + "px";
      clearTimeout(tooltipTimeout);
      tooltip.style.opacity = "1";
    }
    var tooltipTimeout = null;
    function hideTooltip() {
      if (!tooltip) return;
      clearTimeout(tooltipTimeout);
      tooltipTimeout = setTimeout(function () { tooltip.style.opacity = "0"; }, 120);
    }

    /* ────────────────── Badge state ────────────────── */
    var reviewCount = 0;
    function setBadgeCount(count) {
      reviewCount = count || 0;
      if (reviewCount > 0) {
        badge.textContent = reviewCount > 99 ? "99+" : reviewCount;
        badge.classList.add("show");
      } else {
        badge.classList.remove("show");
      }
    }

    // Expose API for background script
    window.__careerTimelineLauncher = { setBadgeCount: setBadgeCount };

    /* ────────────────── Click → open side panel ────────────────── */
    btn.addEventListener("click", function () {
      if (typeof chrome !== "undefined" && chrome.runtime) {
        chrome.runtime.sendMessage({ type: "OPEN_SIDE_PANEL" });
      }
    });

    /* ────────────────── Tooltip handlers ────────────────── */
    btn.addEventListener("mouseenter", function () { showTooltip("Open Resume Timeline Review", btn); });
    btn.addEventListener("focus", function () { showTooltip("Open Resume Timeline Review", btn); });
    btn.addEventListener("mouseleave", hideTooltip);
    btn.addEventListener("blur", hideTooltip);

    function hideTooltip() {
      if (!tooltip) return;
      clearTimeout(tooltipTimeout);
      tooltipTimeout = setTimeout(function () { tooltip.style.opacity = "0"; }, 120);
    }

    /* ────────────────── Keyboard ────────────────── */
    btn.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        btn.click();
      }
    });

    /* ────────────────── Init ────────────────── */
    function init() {
      document.body.appendChild(host);
    }

    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", init);
    } else {
      init();
    }

    /* ────────────────── SPA / navigation survival ────────────────── */
    var observer = new MutationObserver(function () {
      if (!document.getElementById("career-timeline-launcher-host")) {
        init();
      }
    });
    observer.observe(document.body, { childList: true, subtree: true });

    /* ────────────────── Expose API for background script ────────────────── */
    window.__careerTimelineLauncher = { setBadgeCount: setBadgeCount };
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

  /* ──────────────────────────────────────────────────────────────
     Messages from background (parser state updates)
  ────────────────────────────────────────────────────────────── */
  chrome.runtime.onMessage.addListener(function (msg, sender, sendResponse) {
    if (!msg) return;
    if (msg.type === "HIGHLIGHT_RESUME_EVIDENCE") {
      try { sendResponse({ found: highlightFirst(msg.texts || []) }); } catch (e) { sendResponse({ found: false }); }
      return true;
    }
    if (msg.type === "LAUNCHER_BADGE") {
      if (window.__careerTimelineLauncher) {
        window.__careerTimelineLauncher.setBadgeCount(msg.count || 0);
      }
      sendResponse({ ok: true });
      return true;
    }
  });
})();