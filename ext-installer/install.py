#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import json
from pathlib import Path
import venv

install_dir = os.path.join(Path.home(), '.local', 'share', 'summarize')
os.makedirs(install_dir, exist_ok=True)
bin_dir = 'bin'
suffix = ''
if sys.platform == 'darwin':
    nhost_dir = f'{Path.home()}/Library/Application Support/Mozilla/NativeMessagingHosts'
elif sys.platform == 'linux':
    nhost_dir = f'{Path.home()}/.mozilla/native-messaging-hosts'
elif sys.platform == 'win32':
    import winreg
    nhost_dir = install_dir
    bin_dir = 'Scripts'
    suffix = '.exe'
    winreg.SetValue(
        winreg.HKEY_CURRENT_USER,
        'Software\\Mozilla\\NativeMessagingHosts\\summarize',
        winreg.REG_SZ,
        f'{nhost_dir}\\summarize.json'
    )

os.makedirs(nhost_dir, exist_ok=True)
desc = json.load(open(f'{os.path.dirname(__file__)}/summarize.json'))
desc['path'] = os.path.join(install_dir, 'venv', bin_dir, f'summarize_ext{suffix}')
open(f'{nhost_dir}/summarize.json', 'w').write(json.dumps(desc))
venv_dir = os.path.join(install_dir, 'venv')
venv.create(venv_dir, clear=True, with_pip=True)
pip = os.path.join(venv_dir, bin_dir, 'pip')
os.execl(pip, pip, 'install', 'llama-summarizer')
