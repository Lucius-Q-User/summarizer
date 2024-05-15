#!/bin/bash

instdir=~/.local/share/summarize/
mkdir -p "${instdir}"
cp launcher.sh "${instdir}"
mkdir -p ~/.mozilla/native-messaging-hosts/
sed -e "s:##PATH##:${instdir}:" <summarize.json >~/.mozilla/native-messaging-hosts/summarize.json
cd "${instdir}"
python3 -m venv venv
source venv/bin/activate
pip3 install git+https://github.com/Lucius-Q-User/summarizer
