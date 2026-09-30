/* Service worker: opens the review side panel, injects the evidence highlighter,
   and forwards parser state to the Career Timeline launcher. */
chrome.sidePanel
  .setPanelBehavior({ openPanelOnActionClick: true })
  .catch(function () {});

chrome.action.onClicked.addListener(function (tab) {
  if (tab && tab.id != null) {
    chrome.sidePanel.open({ tabId: tab.id }).catch(function () {});
  }
});

chrome.runtime.onMessage.addListener(function (msg, sender, sendResponse) {
  if (msg && msg.type === "ENSURE_CONTENT_SCRIPT" && sender.tab && sender.tab.id != null) {
    chrome.scripting
      .executeScript({ target: { tabId: sender.tab.id }, files: ["content.js"] })
      .then(function () { sendResponse({ ok: true }); })
      .catch(function () { sendResponse({ ok: false }); });
    return true;
  }
  if (msg && msg.type === "OPEN_SIDE_PANEL" && sender.tab && sender.tab.id != null) {
    chrome.sidePanel.open({ tabId: sender.tab.id }).catch(function () {});
    sendResponse({ ok: true });
    return true;
  }
  /* Background -> launcher state propagation.
     The side panel (or parse API response) sends PARSER_RESULT with:
     { success: boolean, confidence: number|null, summary: string|null }
     We forward this to the content script as LAUNCHER_STATE. */
  if (msg && msg.type === "PARSER_RESULT" && sender.tab && sender.tab.id != null) {
    chrome.tabs.sendMessage(sender.tab.id, {
      type: "LAUNCHER_STATE",
      payload: {
        success: msg.success,
        confidence: msg.confidence,
        summary: msg.summary
      }
    }).catch(function () { /* content script not ready */ });
    sendResponse({ ok: true });
    return true;
  }
});

/* Background forwards badge count updates to the launcher. */
chrome.runtime.onMessage.addListener(function (msg, sender, sendResponse) {
  if (msg && msg.type === "LAUNCHER_BADGE" && sender.tab && sender.tab.id != null) {
    chrome.tabs.sendMessage(sender.tab.id, {
      type: "LAUNCHER_BADGE",
      count: msg.count || 0
    }).catch(function () { /* content script not ready */ });
    sendResponse({ ok: true });
    return true;
  }
});

/* Follow the recruiter: when they switch tabs, offer the open resume to the
   side panel (it auto-analyzes only while no real resume has been parsed). */
function notifyOpenTab() {
  chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
    try {
      if (tabs && tabs[0] && tabs[0].url) {
        chrome.runtime.sendMessage({ type: "OPEN_RESUME_TAB", url: tabs[0].url, id: tabs[0].id }).catch(function () {});
      }
    } catch (e) { /* side panel not open */ }
  });
}

chrome.tabs.onActivated.addListener(function () { notifyOpenTab(); });
chrome.tabs.onUpdated.addListener(function (tabId, info) {
  if (info && info.status === "complete") notifyOpenTab();
});