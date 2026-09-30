/* Content script: evidence highlighting + Career Timeline launcher.
   The launcher is isolated in a Shadow DOM so host styles cannot affect it
   and its styles cannot leak. It communicates state via chrome.runtime. */
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
     Career Timeline Launcher (isolated in Shadow DOM)
  ────────────────────────────────────────────────────────────── */
  (function () {
    /* ────────────────── Design tokens (CSS variables) ────────────────── */
    var TOKENS = {
      // Colours
      colourBrand: "#2563EB",              // brand blue
      colourBrandHover: "#1D4ED8",
      colourBrandActive: "#1E40AF",
      colourWhite: "#FFFFFF",
      colourTextOnBrand: "#FFFFFF",
      colourOuterRing: "#FFFFFF",
      colourShadow: "rgba(0, 0, 0, 0.35)",
      colourErrorBg: "#FEF2F2",
      colourErrorText: "#B91C1C",
      colourErrorRing: "#FECACA",
      colourUncertainBg: "#FEF3C7",
      colourUncertainText: "#92400E",
      colourUncertainRing: "#FDE68A",
      colourSpinner: "#FFFFFF",
      colourTooltipBg: "#111827",
      colourTooltipText: "#F9FAFB",
      // Sizing
      sizeFabDiameter: "48px",
      sizePillHeight: "44px",
      sizeIcon: "22px",
      sizeBadgeDiameter: "8px",
      sizeTooltipMaxWidth: "220px",
      radiusPill: "9999px",
      radiusCircle: "50%",
      radiusTooltip: "8px",
      // Spacing
      gapIconText: "10px",
      paddingPillHorizontal: "16px",
      gapBadge: "4px",
      marginSafeBottom: "24px",
      marginSafeRight: "24px",
      // Motion
      durationTransition: "200ms",
      durationFade: "150ms",
      durationPulse: "1.2s",
      // Z-index
      zIndexLauncher: 2147483647,
      zIndexTooltip: 2147483646,
    };

    /* ────────────────── SVG icons ────────────────── */
    // Career Timeline icon: horizontal line with three ascending nodes
    var ICON_TIMELINE = `
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M4 12h16" stroke="currentColor" stroke-width="2"/>
        <circle cx="6" cy="12" r="3" fill="currentColor"/>
        <circle cx="12" cy="10" r="3" fill="currentColor"/>
        <circle cx="18" cy="8" r="3" fill="currentColor"/>
      </svg>
    `;
    var ICON_SPINNER = `
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" aria-hidden="true">
        <circle cx="12" cy="12" r="10" stroke-opacity="0.25"/>
        <path d="M12 2a10 10 0 0 1 10 10" stroke-opacity="1">
          <animateTransform attributeName="transform" type="rotate" from="0 12 12" to="360 12 12" dur="1s" repeatCount="indefinite"/>
        </path>
      </svg>
    `;
    var ICON_ERROR = `
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <circle cx="12" cy="12" r="10"/>
        <line x1="12" y1="8" x2="12" y2="12"/>
        <line x1="12" y1="16" x2="12.01" y2="16"/>
      </svg>
    `;

    /* ────────────────── State machine ────────────────── */
    var STATE = {
      current: "reading",            // 'reading' | 'ready' | 'ready_uncertain' | 'error'
      hasUserClicked: false,         // persists in chrome.storage
      confidence: null,              // 0-1 or null, from parser
      summary: null,                 // e.g. "8 yrs · 4 companies"
    };

    /* ────────────────── Storage helpers ────────────────── */
    function getStorage(key, fallback) {
      return new Promise(function (resolve) {
        try {
          chrome.storage.local.get(key, function (obj) { resolve(obj[key] ?? fallback); });
        } catch (e) { resolve(fallback); }
      });
    }
    function setStorage(key, value) {
      return new Promise(function (resolve) {
        try {
          var obj = {}; obj[key] = value; chrome.storage.local.set(obj, resolve);
        } catch (e) { resolve(); }
      });
    }

    /* ────────────────── Tooltip (custom, viewport-aware) ────────────────── */
    var tooltip = null;
    var tooltipTimeout = null;

    function createTooltip() {
      if (tooltip) return tooltip;
      var el = document.createElement("div");
      el.id = "ct-tooltip";
      el.setAttribute("role", "tooltip");
      el.style.cssText = `
        position: fixed;
        z-index: ${TOKENS.zIndexTooltip};
        background: ${TOKENS.colourTooltipBg};
        color: ${TOKENS.colourTooltipText};
        font: 12px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        padding: 6px 10px;
        border-radius: ${TOKENS.radiusTooltip};
        max-width: ${TOKENS.sizeTooltipMaxWidth};
        opacity: 0;
        pointer-events: none;
        transition: opacity ${TOKENS.durationFade} ease;
        white-space: nowrap;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
      `;
      // Arrow
      var arrow = document.createElement("div");
      arrow.style.cssText = `
        position: absolute;
        width: 0; height: 0;
        border: 6px solid transparent;
        border-top-color: ${TOKENS.colourTooltipBg};
        bottom: -12px; left: 50%; transform: translateX(-50%);
      `;
      el.appendChild(arrow);
      document.body.appendChild(el);
      tooltip = el;
      return el;
    }

    function showTooltip(text, target) {
      if (!tooltip) createTooltip();
      tooltip.textContent = text;
      // position: above the launcher, centered, flip if needed
      var rect = target.getBoundingClientRect();
      var tipRect = tooltip.getBoundingClientRect();
      var left = rect.left + rect.width / 2 - tooltip.offsetWidth / 2;
      var top = rect.top - tooltip.offsetHeight - 8;

      // viewport clamp
      var vpW = window.innerWidth, vpH = window.innerHeight;
      if (left < 8) left = 8;
      if (left + tooltip.offsetWidth > vpW - 8) left = vpW - tooltip.offsetWidth - 8;
      if (top < 8) {
        top = rect.bottom + 8;
        tooltip.querySelector("div").style.cssText += "border-top-color: transparent; border-bottom-color: inherit; bottom: auto; top: -12px;";
      } else {
        tooltip.querySelector("div").style.cssText = "border-top-color: inherit; border-bottom-color: transparent; bottom: -12px; top: auto;";
      }
      tooltip.style.left = left + "px";
      tooltip.style.top = top + "px";

      clearTimeout(tooltipTimeout);
      tooltip.style.opacity = "1";
    }

    function hideTooltip() {
      if (!tooltip) return;
      clearTimeout(tooltipTimeout);
      tooltipTimeout = setTimeout(function () { tooltip.style.opacity = "0"; }, TOKENS.durationFade);
    }

    /* ────────────────── Shadow DOM host & internals ────────────────── */
    var host = document.createElement("div");
    host.id = "career-timeline-launcher-host";
    host.style.cssText = "all: initial; position: fixed; bottom: 24px; right: 24px; z-index: " + TOKENS.zIndexLauncher + "; font-family: inherit;";

    // Attach shadow DOM
    var shadow = host.attachShadow({ mode: "closed" });

    /* ────────────────── Styles inside Shadow DOM ────────────────── */
    var style = document.createElement("style");
    style.textContent = `
      :host { all: initial; display: block; }

      /* ───────── Design tokens ───────── */
      :root {
        --ct-brand: ${TOKENS.colourBrand};
        --ct-brand-hover: ${TOKENS.colourBrandHover};
        --ct-brand-active: ${TOKENS.colourBrandActive};
        --ct-white: ${TOKENS.colourWhite};
        --ct-text-on-brand: ${TOKENS.colourTextOnBrand};
        --ct-outer-ring: ${TOKENS.colourOuterRing};
        --ct-shadow: ${TOKENS.colourShadow};
        --ct-error-bg: ${TOKENS.colourErrorBg};
        --ct-error-text: ${TOKENS.colourErrorText};
        --ct-error-ring: ${TOKENS.colourErrorRing};
        --ct-uncertain-bg: ${TOKENS.colourUncertainBg};
        --ct-uncertain-text: ${TOKENS.colourUncertainText};
        --ct-uncertain-ring: ${TOKENS.colourUncertainRing};
        --ct-spinner: ${TOKENS.colourSpinner};
        --ct-fab-diameter: ${TOKENS.sizeFabDiameter};
        --ct-pill-height: ${TOKENS.sizePillHeight};
        --ct-icon-size: ${TOKENS.sizeIcon};
        --ct-badge-diameter: ${TOKENS.sizeBadgeDiameter};
        --ct-radius-pill: ${TOKENS.radiusPill};
        --ct-radius-circle: ${TOKENS.radiusCircle};
        --ct-gap-icon-text: ${TOKENS.gapIconText};
        --ct-padding-pill-h: ${TOKENS.paddingPillHorizontal};
        --ct-gap-badge: ${TOKENS.gapBadge};
        --ct-transition: ${TOKENS.durationTransition};
        --ct-fade: ${TOKENS.durationFade};
        --ct-pulse: ${TOKENS.durationPulse};
        --ct-z-launcher: ${TOKENS.zIndexLauncher};
      }

      /* ───────── Base button ───────── */
      .ct-launcher {
        all: initial;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: var(--ct-gap-icon-text);
        height: var(--ct-pill-height);
        padding: 0 var(--ct-padding-pill-h);
        border-radius: var(--ct-radius-pill);
        background: var(--ct-brand);
        color: var(--ct-text-on-brand);
        border: 2px solid var(--ct-outer-ring);
        box-shadow: 0 2px 8px var(--ct-shadow);
        cursor: pointer;
        font: 500 13px/1 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        white-space: nowrap;
        transition: background-color var(--ct-transition) ease, box-shadow var(--ct-transition) ease, transform 100ms ease, width var(--ct-transition) ease, opacity var(--ct-transition) ease;
        outline: none;
        position: relative;
        min-width: 44px;
        min-height: 44px;
      }
      .ct-launcher:focus-visible {
        outline: 3px solid var(--ct-brand);
        outline-offset: 3px;
      }
      .ct-launcher:hover:not(:disabled) { background: var(--ct-brand-hover); }
      .ct-launcher:active:not(:disabled) { background: var(--ct-brand-active); transform: scale(0.97); }
      .ct-launcher:disabled { cursor: not-allowed; opacity: 0.6; }

      /* ───────── Ring + shadow (always on) ───────── */
      .ct-launcher::before {
        content: "";
        position: absolute;
        inset: -2px;
        border-radius: inherit;
        border: 2px solid var(--ct-outer-ring);
        pointer-events: none;
        z-index: -1;
      }

      /* ───────── Collapsed (icon-only) ───────── */
      .ct-launcher.collapsed {
        width: var(--ct-fab-diameter);
        height: var(--ct-fab-diameter);
        border-radius: var(--ct-radius-circle);
        padding: 0;
        justify-content: center;
      }
      .ct-launcher.collapsed .ct-label { display: none; }
      .ct-launcher.collapsed .ct-badge { display: none; }

      /* ───────── Icon ───────── */
      .ct-icon { display: flex; align-items: center; justify-content: center; flex-shrink: 0; width: var(--ct-icon-size); height: var(--ct-icon-size); color: var(--ct-white); }
      .ct-icon svg { width: 100%; height: 100%; }

      /* ───────── Label ───────── */
      .ct-label { font-weight: 500; font-size: 13px; line-height: 1; white-space: nowrap; overflow: hidden; opacity: 1; transition: opacity var(--ct-transition) ease, width var(--ct-transition) ease; }

      /* ───────── Spinner ───────── */
      .ct-spinner { width: 20px; height: 20px; animation: ct-spin 1s linear infinite; color: var(--ct-spinner); }
      @keyframes ct-spin { to { transform: rotate(360deg); } }

      /* ───────── Badge ───────── */
      .ct-badge {
        position: absolute;
        top: -${TOKENS.sizeBadgeDiameter / 2}px;
        right: -${TOKENS.sizeBadgeDiameter / 2}px;
        width: ${TOKENS.sizeBadgeDiameter}px; height: ${TOKENS.sizeBadgeDiameter}px;
        border-radius: 50%;
        background: #F59E0B;
        border: 2px solid var(--ct-brand);
        flex-shrink: 0;
      }
      .ct-badge.uncertain { background: #F59E0B; }

      /* ───────── Teaser badge (high confidence summary) ───────── */
      .ct-teaser {
        position: absolute;
        bottom: 100%; right: 0;
        margin-bottom: 6px;
        padding: 4px 10px;
        font: 500 11px/1 system-ui, sans-serif;
        color: var(--ct-white);
        background: var(--ct-brand);
        border-radius: 9999px;
        white-space: nowrap;
        opacity: 0; transform: translateY(4px);
        pointer-events: none;
        transition: opacity var(--ct-fade) ease, transform var(--ct-fade) ease;
        border: 2px solid var(--ct-outer-ring);
      }
      .ct-launcher.has-teaser .ct-teaser { opacity: 1; transform: translateY(0); }

      /* ───────── Pulse animation (ready) ───────── */
      @keyframes ct-pulse { 0%, 100% { box-shadow: 0 2px 8px var(--ct-shadow); } 50% { box-shadow: 0 2px 8px var(--ct-shadow), 0 0 0 6px rgba(37, 99, 235, 0.4); } }
      .ct-pulse { animation: ct-pulse var(--ct-pulse) ease-out 1; }

      /* ───────── Reduced motion ───────── */
      @media (prefers-reduced-motion: reduce) {
        .ct-launcher, .ct-teaser, .ct-badge { transition: none !important; animation: none !important; }
      }
    `;
    shadow.appendChild(style);

    /* ────────────────── Launcher element ────────────────── */
    var launcher = document.createElement("button");
    launcher.className = "ct-launcher";
    launcher.setAttribute("type", "button");
    launcher.setAttribute("aria-label", "Open Career Timeline");
    launcher.setAttribute("aria-live", "polite");
    launcher.setAttribute("aria-expanded", "false");

    // Icon slot
    var iconWrap = document.createElement("span");
    iconWrap.className = "ct-icon";
    iconWrap.innerHTML = ICON_TIMELINE;

    // Label
    var label = document.createElement("span");
    label.className = "ct-label";
    label.textContent = "Career Timeline";

    // Spinner (hidden initially)
    var spinner = document.createElement("span");
    spinner.className = "ct-spinner";
    spinner.style.display = "none";
    spinner.innerHTML = ICON_SPINNER;

    // Error icon
    var errorIcon = document.createElement("span");
    errorIcon.className = "ct-icon";
    errorIcon.style.display = "none";
    errorIcon.innerHTML = ICON_ERROR;

    // Label wrapper for state switching
    var labelWrapper = document.createElement("span");
    labelWrapper.style.display = "flex";
    labelWrapper.style.alignItems = "center";
    labelWrapper.style.gap = "8px";
    labelWrapper.appendChild(label);

    // Teaser badge
    var teaser = document.createElement("span");
    teaser.className = "ct-teaser";
    teaser.setAttribute("aria-hidden", "true");

    // Amber uncertainty badge
    var badge = document.createElement("span");
    badge.className = "ct-badge uncertain";
    badge.setAttribute("aria-hidden", "true");
    badge.style.display = "none";

    // Assemble
    launcher.appendChild(iconWrap);
    launcher.appendChild(spinner);
    launcher.appendChild(errorIcon);
    launcher.appendChild(labelWrapper);
    launcher.appendChild(badge);
    launcher.appendChild(teaser);

    shadow.appendChild(launcher);

    /* ────────────────── State rendering ────────────────── */
    function render() {
      var s = STATE.current;
      var isCollapsed = STATE.hasUserClicked;

      // Collapsed / expanded
      launcher.classList.toggle("collapsed", isCollapsed);

      // State-specific rendering
      launcher.classList.remove("ct-pulse");
      launcher.disabled = false;
      spinner.style.display = "none";
      errorIcon.style.display = "none";
      label.textContent = "Career Timeline";
      launcher.style.background = "";
      launcher.style.color = "";
      launcher.style.borderColor = TOKENS.colourOuterRing;
      badge.style.display = "none";
      teaser.style.display = "none";
      launcher.classList.remove("ct-pulse");

      switch (s) {
        case "reading":
          label.textContent = "Reading resume…";
          spinner.style.display = "flex";
          launcher.disabled = true;
          launcher.style.cursor = "not-allowed";
          break;
        case "ready":
          if (!launcher.classList.contains("ct-pulse")) {
            // single pulse on transition to ready
            launcher.classList.add("ct-pulse");
          }
          break;
        case "ready_uncertain":
          badge.style.display = "flex";
          break;
        case "error":
          label.textContent = "Couldn't read this resume";
          errorIcon.style.display = "flex";
          launcher.style.background = TOKENS.colourErrorBg;
          launcher.style.color = TOKENS.colourErrorText;
          launcher.style.borderColor = TOKENS.colourErrorRing;
          break;
      }

      // Teaser badge (only in ready/ready_uncertain with high confidence)
      if ((s === "ready" || s === "ready_uncertain") && STATE.confidence != null && STATE.confidence >= 0.75 && STATE.summary) {
        teaser.textContent = STATE.summary;
        teaser.style.display = "block";
        launcher.classList.add("has-teaser");
      } else {
        teaser.style.display = "none";
        launcher.classList.remove("has-teaser");
      }
    }

    function setState(newState) {
      STATE.current = newState;
      render();
    }

    /* ────────────────── Persistence ────────────────── */
    async function loadPersisted() {
      var clicked = await getStorage("ct_hasUserClicked", false);
      STATE.hasUserClicked = clicked;
    }
    async function persistClick() {
      STATE.hasUserClicked = true;
      await setStorage("ct_hasUserClicked", true);
    }

    /* ────────────────── Public API (called from background) ────────────────── */
    function setStateFromParser(payload) {
      if (payload && typeof payload.confidence === "number") {
        STATE.confidence = payload.confidence;
      }
      if (payload && typeof payload.summary === "string") {
        STATE.summary = payload.summary;
      }
      // If parser succeeded, move to ready/ready_uncertain
      if (payload && payload.success) {
        setState(payload.confidence != null && payload.confidence < 0.75 ? "ready_uncertain" : "ready");
      } else if (payload && payload.success === false) {
        setState("error");
      }
      render();
    }

    /* ────────────────── Click handler ────────────────── */
    function onClick(e) {
      if (STATE.current === "reading") return;
      if (STATE.current === "error") {
        // Open panel with retry option
        chrome.runtime.sendMessage({ type: "OPEN_SIDE_PANEL", retry: true });
        return;
      }
      if (!STATE.hasUserClicked) {
        persistClick();
      }
      chrome.runtime.sendMessage({ type: "OPEN_SIDE_PANEL" });
    }
    launcher.addEventListener("click", onClick);

    /* ────────────────── Tooltip handlers ────────────────── */
    function tooltipText() {
      var s = STATE.current;
      if (s === "reading") return "Reading the resume…";
      if (s === "ready") return "Open Career Timeline";
      if (s === "ready_uncertain") return "Open Career Timeline — some details may need checking";
      if (s === "error") return "Couldn't read this resume — click to retry";
      return "Career Timeline";
    }
    launcher.addEventListener("mouseenter", function () { showTooltip(tooltipText(), launcher); });
    launcher.addEventListener("focus", function () { showTooltip(tooltipText(), launcher); });
    launcher.addEventListener("mouseleave", hideTooltip);
    launcher.addEventListener("blur", hideTooltip);

    /* ────────────────── Keyboard ────────────────── */
    launcher.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault(); launcher.click();
      }
    });

    /* ────────────────── Init ────────────────── */
    function init() {
      document.body.appendChild(host);
      loadPersisted().then(render);
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
    window.__careerTimelineLauncher = { setState: setStateFromParser };
  })();

  /* ──────────────────────────────────────────────────────────────
     Tooltip helpers (in light DOM, not shadow)
  ────────────────────────────────────────────────────────────── */
  var tooltip = null, tooltipTimeout = null;
  function createTooltip() {
    if (tooltip) return tooltip;
    var el = document.createElement("div");
    el.id = "ct-tooltip";
    el.setAttribute("role", "tooltip");
    el.style.cssText = "position:fixed;z-index:2147483646;background:#111827;color:#F9FAFB;font:12px system-ui,sans-serif;padding:6px 10px;border-radius:8px;max-width:220px;opacity:0;pointer-events:none;transition:opacity 150ms ease;white-space:nowrap;box-shadow:0 4px 12px rgba(0,0,0,0.25);";
    var arrow = document.createElement("div");
    arrow.style.cssText = "position:absolute;width:0;height:0;border:6px solid transparent;border-top-color:#111827;bottom:-12px;left:50%;transform:translateX(-50%);";
    el.appendChild(arrow);
    document.body.appendChild(el);
    tooltip = el;
    return el;
  }
  var tooltipTimeout = null;
  function showTooltip(text, target) {
    if (!tooltip) createTooltip();
    tooltip.textContent = text;
    var rect = target.getBoundingClientRect();
    var left = rect.left + rect.width / 2 - tooltip.offsetWidth / 2;
    var top = rect.top - tooltip.offsetHeight - 8;
    var vpW = window.innerWidth, vpH = window.innerHeight;
    if (left < 8) left = 8;
    if (left + tooltip.offsetWidth > window.innerWidth - 8) left = window.innerWidth - tooltip.offsetWidth - 8;
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
  function hideTooltip() {
    if (!tooltip) return;
    clearTimeout(tooltipTimeout);
    tooltipTimeout = setTimeout(function () { tooltip.style.opacity = "0"; }, 150);
  }

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
    if (msg.type === "LAUNCHER_STATE") {
      if (window.__careerTimelineLauncher) {
        window.__careerTimelineLauncher.setState(msg.payload || {});
      }
      sendResponse({ ok: true });
      return true;
    }
  });

  /* ──────────────────────────────────────────────────────────────
     Expose API for background script
  ────────────────────────────────────────────────────────────── */
  window.__careerTimelineLauncher = { setState: function(payload){ /* setState will be assigned after launcher init */ } };
  // The actual setState will be assigned inside the IIFE above
})();