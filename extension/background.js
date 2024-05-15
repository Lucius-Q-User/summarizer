browser.menus.onClicked.addListener(async function (info) {
    let port = browser.runtime.connectNative("summarize");

    let tab = await browser.tabs.create({
        url: "/progress.html",
    });

    port.onMessage.addListener(async function (response) {
        browser.tabs.sendMessage(tab.id, response);
        if (response.msg == "complete") {
            port.disconnect();
        }
    });

    let ctx = crypto.randomUUID();
    port.postMessage({
        url: info.linkUrl,
        ctx: ctx,
        action: "process"
    });
});

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
