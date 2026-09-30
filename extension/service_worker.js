/* Service worker: opens the review side panel on toolbar click and injects
   the evidence highlighter into the active tab on demand. */

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
});

// Follow the recruiter: when they switch tabs, offer the open resume to the
// side panel (it auto-analyzes only while no real resume has been parsed).
function notifyOpenTab() {
  chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
    try {
      if (tabs && tabs[0] && tabs[0].url) {
        chrome.runtime.sendMessage({ type: "OPEN_RESUME_TAB", url: tabs[0].url }).catch(function () {});
      }
    } catch (e) { /* side panel not open */ }
  });
}

chrome.tabs.onActivated.addListener(function () { notifyOpenTab(); });
chrome.tabs.onUpdated.addListener(function (tabId, info) {
  if (info && info.status === "complete") notifyOpenTab();
});
