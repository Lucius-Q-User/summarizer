#!/bin/bash

cd "$(dirname "$0")"
instdir=~/.local/share/summarize/
mkdir -p "${instdir}"
cp launcher.sh "${instdir}"
if grep -q Darwin <<<"$(uname)"; then
    nhdir=~/Library/Application\ Support/Mozilla/NativeMessagingHosts
else
    nhdir=~/.mozilla/native-messaging-hosts
fi
mkdir -p "${nhdir}"
sed -e "s:##PATH##:${instdir}:" <summarize.json >"${nhdir}/summarize.json"
cd "${instdir}"
python3 -m venv venv
source venv/bin/activate
pip3 install llama-summarizer
