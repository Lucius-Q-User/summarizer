// SPDX-License-Identifier: Apache-2.0

let id = window.location.hash.substring(1);
let port = browser.runtime.connectNative("summarize");
let uuid = crypto.randomUUID();

port.onMessage.addListener(async function (response) {
    let ctx = response.ctx;
    if (ctx != uuid) {
        return
    }
    document.open();
    document.write(response.data);
    document.close();
    port.disconnect();
});
port.postMessage({
    action: "load",
    id: id,
    ctx: uuid,
});
