# SPDX-License-Identifier: Apache-2.0

import os
from argparse import ArgumentParser
import shutil
from .summarizer import process_video, load_config, LLM_PROVIDERS, LOCAL_WHISPER_DEFAULT, WHISPER_PROVIDERS
from tqdm import tqdm

OUT_DIR = 'out'

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
    preparser = ArgumentParser(add_help=False)
    preparser.add_argument('-aop', '--add-openai-profile', action = 'append', default=config.get('openai_profile', []))
    preargs = preparser.parse_known_args()[0]

    llm_provider_choices = list(LLM_PROVIDERS.keys())
    llm_provider_choices.extend(preargs.add_openai_profile)
    whisper_provider_choices = list(WHISPER_PROVIDERS.keys())
    whisper_provider_choices.extend(preargs.add_openai_profile)
    parser = ArgumentParser(prog='summarize')
    parser.add_argument('video_url')
    parser.add_argument('-lp', '--llm-provider', choices = llm_provider_choices, default = config.get('llm_provider'))
    parser.add_argument('-wp', '--whisper-provider', choices = whisper_provider_choices, default = config.get('whisper_provider'))
    parser.add_argument('-sb', '--sponsorblock',
                        choices = ['sponsor', 'selfpromo', 'interaction', 'intro', 'outro', 'preview', 'music', 'offtopic', 'filler'],
                        action = 'append', default = config.get('sponsorblock'))
    parser.add_argument('--transcript-only', action='store_true', help='Only return the transcript, skip summarization')
    parser.add_argument('-lmr', '--local-model-repo', default = config.get('local_model_repo'))
    parser.add_argument('-lmf', '--local-model-file', default = config.get('local_model_file'))
    parser.add_argument('-om', '--openai-model', default = config.get('openai_model'))
    parser.add_argument('-ou', '--openai-base-url', default = config.get('openai_base_url'))
    parser.add_argument('-lwm', '--local-whisper-model',
                        choices = ['tiny', 'tiny.en', 'base', LOCAL_WHISPER_DEFAULT, 'small', 'small.en', 'medium', 'medium.en', 'large-v1', 'large-v2', 'large-v3'],
                        default = config.get('local_whisper_model'))
    parser.add_argument('-owm', '--openai-whisper-model', default = config.get('openai_whisper_model'))
    parser.add_argument('-v', '--verbose', default = config.get('verbose'), action = 'store_true')
    parser.add_argument('--force-local-transcribe', action = 'store_true')
    parser.add_argument('-aop', '--add-openai-profile', action = 'append', default=config.get('openai_profile', []))
    for item in preargs.add_openai_profile:
        parser.add_argument(f'--{item}-model', default = config.get(f'{item}_model'))
        parser.add_argument(f'--{item}-base-url', default = config.get(f'{item}_base_url'))
        parser.add_argument(f'--{item}-whisper-model', default = config.get(f'{item}_whisper_model'))

    args = parser.parse_args()
    api_key = config.get('openai_api_key', None)
    api_key = os.environ.get('OPENAI_API_KEY', api_key)
    args.openai_api_key = api_key
    for item in preargs.add_openai_profile:
        api_key = config.get(f'{item}_api_key', None)
        api_key = os.environ.get(f'{item.upper()}_API_KEY', api_key)
        setattr(args, f'{item}_api_key', api_key)
    kwargs = {k: v for (k, v) in vars(args).items() if v is not None}
    progress = ProgressHooks()
    result = process_video(progress, **kwargs)
    progress.close()
    output_text = result.summary  # Get the transcript or summary
    file_extension = "txt" if args.transcript_only else "html"
    filename = os.path.join(OUT_DIR, f'{result.video_id}.{file_extension}')
    os.makedirs(OUT_DIR, exist_ok = True)
    with open(filename, 'wb') as out:
        out.write(output_text.encode('utf-8'))

    if not args.transcript_only:
        for opener in ['open', 'xdg-open']:
            if shutil.which(opener) is not None:
                os.execlp(opener, opener, filename)
                return
        if shutil.which('cmd') is not None:
            os.execlp('cmd', 'cmd', '/c', filename)
    print(f'Unable to open the file automatically, the output was written to {filename}')
