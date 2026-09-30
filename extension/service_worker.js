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
