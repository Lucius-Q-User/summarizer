# SPDX-License-Identifier: Apache-2.0

import os
from argparse import ArgumentParser
import shutil
from .summarizer import process_video, load_config, PROVIDERS, WHISPER_DEFAULT

OUT_DIR = 'out'
GROQ_API_KEY_VAR = 'GROQ_API_KEY'

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
    parser.add_argument('-om', '-gm', '--openai-model', '--groq-model', default = config.get('openai_model'))
    parser.add_argument('-ou', '--openai-base-url', default = config.get('openai_base_url'))
    parser.add_argument('-wm', '--whisper-model',
                        choices = ['tiny', 'tiny.en', 'base', WHISPER_DEFAULT, 'small', 'small.en', 'medium', 'medium.en', 'large-v1', 'large-v2', 'large-v3'],
                        default = config.get('whisper_model'))
    parser.add_argument('-mp', '--meta-proxy', default = config.get('meta_proxy'))
    parser.add_argument('--force-local-transcribe', action = 'store_true')
    args = parser.parse_args()
    api_key = config.get(GROQ_API_KEY_VAR, None)
    api_key = config.get('openai_api_key', api_key)
    api_key = os.environ.get(GROQ_API_KEY_VAR, api_key)
    api_key = os.environ.get('OPENAI_API_KEY', api_key)
    args.openai_api_key = api_key
    kwargs = {k: v for (k, v) in vars(args).items() if v is not None}
    result = process_video(**kwargs)
    filename = f'{OUT_DIR}/{result.video_id}.html'
    with open(filename, 'w') as out:
        out.write(result.summary)
    for opener in ['open', 'xdg-open']:
        if shutil.which(opener) is not None:
            os.execlp(opener, opener, filename)
            return
    print(f'Unable to open the file automatically, the output was written to {filename}')
