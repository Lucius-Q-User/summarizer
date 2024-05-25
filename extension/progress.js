// SPDX-License-Identifier: Apache-2.0

let phaseName = document.querySelector("#phase_name");
let phaseProgress = document.querySelector("#phase_progress");
let subphaseProgress = document.querySelector("#subphase_progress");
let subphaseContainer = document.querySelector("#subphase_container");
let state = {
    substeps: 0,
    cur_substeps: 0
};
browser.runtime.onMessage.addListener((data, sender) => {
    if (data.msg == "phase") {
        phaseProgress.style.width = Math.floor(100 * data.idx / 5) + "%";
        phaseName.innerText = data.name;
        let substeps = data.substeps;
        if (substeps != 0) {
            subphaseContainer.style.display = "flex";
        } else {
            subphaseContainer.style.display = "none";
        }
        state.substeps = substeps;
        state.cur_substeps = 0;
        subphaseProgress.style.width = "0%";
        subphaseProgress.innerText = "";
    } else if (data.msg == "set_substeps") {
        state.substeps = data.num;
    } else if (data.msg == "subphase_step") {
        let val = data.val;
        if (val == null) {
            state.cur_substeps += 1;
        } else {
            state.cur_substeps = val;
        }
        subphaseProgress.style.width = Math.floor(100 * state.cur_substeps / state.substeps) + "%";
        subphaseProgress.innerText = state.cur_substeps + "/" + state.substeps;
    } else if (data.msg == "complete") {
        window.location.replace(browser.runtime.getURL("summary.html") + "#" + data.id);
    }
});
