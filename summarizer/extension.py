# SPDX-License-Identifier: Apache-2.0

import json
import struct
import sys
import os
from .summarizer import process_video, load_config

OUT_DIR = 'out'
GROQ_API_KEY_VAR = 'GROQ_API_KEY'

def output(d):
    js = json.dumps(d).encode()
    length = struct.pack('@I', len(js))
    sys.stdout.buffer.write(length)
    sys.stdout.buffer.write(js)
    sys.stdout.buffer.flush()

class ProgressHooks(object):
    def __init__(self, ctx):
        self.ctx = ctx
    def phase(self, idx, name, substeps=0, bytes=False):
        output({
            'msg': 'phase',
            'idx': idx,
            'name': name,
            'substeps': substeps,
            'bytes': bytes,
            'ctx': self.ctx
        })
    def subphase_step(self, val = None):
        output({
            'msg': 'subphase_step',
            'val': val,
            'ctx': self.ctx
        })
    def set_substeps(self, num):
        output({
            'msg': 'set_substeps',
            'num': num,
            'ctx': self.ctx
        })
    def close(self):
        pass

def summarize(msg):
    ctx = msg['ctx']
    config = load_config()
    api_key = config.get(GROQ_API_KEY_VAR, None)
    api_key = config.get('openai_api_key', api_key)
    api_key = os.environ.get(GROQ_API_KEY_VAR, api_key)
    api_key = os.environ.get('OPENAI_API_KEY', api_key)
    config['openai_api_key'] = api_key
    result = process_video(
        ProgressHooks(ctx), video_url=msg['url'], **config
    )
    filename = f'{OUT_DIR}/{result.video_id}.html'
    os.makedirs(OUT_DIR, exist_ok = True)
    with open(filename, 'wb') as out:
        out.write(result.summary.encode('utf-8'))
    output({
        'msg': 'complete',
        'id': result.video_id,
        'ctx': ctx
    })

def load(msg):
    ctx = msg['ctx']
    try:
        with open(f'{OUT_DIR}/{msg["id"]}.html') as f:
            output({
                'ctx': ctx,
                'data': f.read()
            })
    except:
        output({
            'ctx': ctx,
            'data': 'Not found'
        })

def main():
    while 1:
        raw_len = sys.stdin.buffer.read(4)
        if len(raw_len) == 0:
            return
        length = struct.unpack('@I', raw_len)[0]
        msg = json.loads(sys.stdin.buffer.read(length).decode())
        if msg['action'] == 'process':
            summarize(msg)
        elif msg['action'] == 'load':
            load(msg)
