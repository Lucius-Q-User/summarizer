# SPDX-License-Identifier: Apache-2.0

import os
from argparse import ArgumentParser
import shutil
from .summarizer import process_video, load_config, PROVIDERS, WHISPER_DEFAULT
from tqdm import tqdm

OUT_DIR = 'out'
GROQ_API_KEY_VAR = 'GROQ_API_KEY'

class ProgressHooks(object):
    def __init__(self):
        self.top_bar = tqdm(total=5, position=0, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ')
        self.last_top = 0
        self.sub_bar = None
    def phase(self, idx, name, substeps=0, bytes=False):
        up_by = idx - self.last_top
        self.last_top = idx
        self.last_sub = 0
        if self.sub_bar is not None:
            self.sub_bar.close()
        self.top_bar.set_description(name)
        self.top_bar.update(up_by)
        if substeps != 0:
            extra = {}
            if bytes:
                extra = dict(unit='B', unit_divisor=1024, unit_scale=True)
            self.sub_bar = tqdm(total=substeps, position=1, **extra)
        else:
            self.sub_bar = None
    def subphase_step(self, val = None):
        if val is None:
            self.sub_bar.update()
            return
        up_by = val - self.last_sub
        self.last_sub = val
        self.sub_bar.update(up_by)
    def set_substeps(self, num):
        self.sub_bar.total = num
    def close(self):
        self.top_bar.close()
        if self.sub_bar is not None:
            self.sub_bar.close()

def main():
    config = load_config()
    parser = ArgumentParser(prog='summarize')
    parser.add_argument('video_url')
    parser.add_argument('-lp', '--llm-provider', choices = PROVIDERS.keys(), default = config.get('llm_provider'))
    parser.add_argument('-sb', '--sponsorblock',
                        choices = ['sponsor', 'selfpromo', 'interaction', 'intro', 'outro', 'preview', 'music', 'offtopic', 'filler'],
                        action = 'append', default = config.get('sponsorblock'))
    parser.add_argument('-lmr', '--local-model-repo', default = config.get('local_model_repo'))
    parser.add_argument('-lmf', '--local-model-file', default = config.get('local_model_file'))
    parser.add_argument('-hm', '--huggingchat-model', default = config.get('huggingchat_model'))
    parser.add_argument('-om', '-gm', '--openai-model', '--groq-model', default = config.get('openai_model'))
    parser.add_argument('-ou', '--openai-base-url', default = config.get('openai_base_url'))
    parser.add_argument('-wm', '--whisper-model',
                        choices = ['tiny', 'tiny.en', 'base', WHISPER_DEFAULT, 'small', 'small.en', 'medium', 'medium.en', 'large-v1', 'large-v2', 'large-v3'],
                        default = config.get('whisper_model'))
    parser.add_argument('-v', '--verbose', default = config.get('verbose'), action  = 'store_true')
    parser.add_argument('--force-local-transcribe', action = 'store_true')
    args = parser.parse_args()
    api_key = config.get(GROQ_API_KEY_VAR, None)
    api_key = config.get('openai_api_key', api_key)
    api_key = os.environ.get(GROQ_API_KEY_VAR, api_key)
    api_key = os.environ.get('OPENAI_API_KEY', api_key)
    args.openai_api_key = api_key
    kwargs = {k: v for (k, v) in vars(args).items() if v is not None}
    progress = ProgressHooks()
    result = process_video(progress, **kwargs)
    progress.close()
    filename = os.path.join(OUT_DIR, f'{result.video_id}.html')
    os.makedirs(OUT_DIR, exist_ok = True)
    with open(filename, 'wb') as out:
        out.write(result.summary.encode('utf-8'))
    for opener in ['open', 'xdg-open']:
        if shutil.which(opener) is not None:
            os.execlp(opener, opener, filename)
            return
    if shutil.which('cmd') is not None:
        os.execlp('cmd', 'cmd', '/c', filename)
    print(f'Unable to open the file automatically, the output was written to {filename}')
