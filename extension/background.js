// SPDX-License-Identifier: Apache-2.0

async function doSummarize(url, background) {
    let port = browser.runtime.connectNative("summarize");

    let tab = await browser.tabs.create({
        url: "/progress.html",
        active: !background
    });

    port.onMessage.addListener(async function (response) {
        browser.tabs.sendMessage(tab.id, response);
        if (response.msg == "complete") {
            port.disconnect();
        }
    });

    let ctx = crypto.randomUUID();
    port.postMessage({
        url: url,
        ctx: ctx,
        action: "process"
    });
}

browser.menus.onClicked.addListener((info) => {
    let cmd = info.modifiers.includes("Command");
    let ctrl = info.modifiers.includes("Ctrl") && !info.modifiers.includes("MacCtrl");
    doSummarize(info.linkUrl, cmd || ctrl);
});

browser.pageAction.onClicked.addListener((tab) => {
    doSummarize(tab.url, true);
})

browser.menus.create({
    contexts: [
        "link", "tab", "video"
    ],
    targetUrlPatterns: [
        "*://*.youtube.com/*",
    ],
    id: "do_run",
    title: "Summarize video"
});
