let port = browser.runtime.connectNative("summarize");
port.onMessage.addListener(async function (response) {
    let ctx = response.ctx;
    let data = await browser.storage.session.get(ctx);
    browser.tabs.sendMessage(data[ctx], response);
});

browser.menus.onClicked.addListener(async function (info) {
    let tab = await browser.tabs.create({
        url: "/progress.html",
    });
    let ctx = crypto.randomUUID();
    await browser.storage.session.set({
        [ctx]: tab.id
    });
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
