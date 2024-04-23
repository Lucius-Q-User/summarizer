# SPDX-License-Identifier: Apache-2.0

import requests
from io import StringIO, BytesIO
import urllib.parse as parse
import os
from jinja2 import Environment
from collections import namedtuple
import huggingface_hub
from yt_dlp import YoutubeDL
from tempfile import TemporaryDirectory
from argparse import ArgumentParser
import json
import time
import sys
import shutil
import math

Segment = namedtuple('Segment', ['start', 'end', 'text'])
HourSummary = namedtuple('HourSummary', ['overall', 'parts'])
TimeUrlFn = namedtuple('TimeUrlFn', ['extractor', 'fn'])

OUT_DIR = 'out'
AUDIO_FILE = 'audio.m4a'
WHISPER_MODEL = 'ggml-base.en.bin'
GROQ_API_KEY_VAR = 'GROQ_API_KEY'
XDG_CACHE_HOME = 'XDG_CACHE_HOME'
XDG_CONFIG_HOME = 'XDG_CONFIG_HOME'

class LocalLLM(object):
    def __init__(self, args):
        from llama_cpp import Llama
        model = huggingface_hub.hf_hub_download(args.local_model_repo, args.local_model_file)
        self.llama = Llama(
             model, n_gpu_layers=-1, n_ctx=0#, verbose=False
        )
    def run_llm(self, prompt):
        resp = self.llama.create_chat_completion(messages=[
            {'role': 'user', 'content': prompt}
        ], max_tokens=None)
        return resp['choices'][0]['message']['content']

class GroqLLM(object):
    def __init__(self, args):
        if GROQ_API_KEY_VAR in os.environ:
            self.api_key = os.environ[GROQ_API_KEY_VAR]
        else:
            if XDG_CONFIG_HOME in os.environ:
                cfg_dir = os.environ[XDG_CONFIG_HOME]
            else:
                cfg_dir = f'{os.environ["HOME"]}/.config'
            self.api_key = json.load(open(f'{cfg_dir}/summarize.json'))[GROQ_API_KEY_VAR]
        self.model = args.groq_model
    def run_llm(self, prompt):
        req = {
            'model': self.model,
            'max_tokens': 32768,
            'messages': [
                {'role': 'user','content': prompt}
            ]
        }
        headers = {
            'Authorization': f'Bearer {self.api_key}'
        }
        while 1:
            resp = requests.post('https://api.groq.com/openai/v1/chat/completions', headers = headers, data = json.dumps(req))
            if resp.status_code == 429:
                time.sleep(int(resp.headers['retry-after']) + 2)
                continue
            return resp.json()['choices'][0]['message']['content']


def find_json3(fmts):
    for fmt in fmts:
        if fmt['ext'] == 'json3':
            return fmt['url']
    return None

def extract_sub_url(video_info):
    if 'subtitles' in video_info:
        subs = video_info['subtitles']
        lang = None
        for k in subs.keys():
            if k.startswith('en'):
                lang = k
                break
        if lang is not None:
            url = find_json3(subs[lang])
            if url is not None:
                return url
    if 'automatic_captions' in video_info:
        lang = None
        subs = video_info['automatic_captions']
        if 'en-orig' in subs:
            lang = 'en-orig'
        elif 'en' in subs:
            lang = 'en'
        else:
            return None
        return find_json3(subs[lang])
    return None

def download_captions(video_info):
    url = extract_sub_url(video_info)
    if url is None:
        return None
    events = requests.get(url).json()['events']
    segments = []
    for event in events:
        text = ''.join(x['utf8'] for x in event.get('segs', []))
        if text == '':
            continue
        start = event['tStartMs']
        segments.append(Segment(start // 1000, (start + event.get('dDurationMs', 0)) // 1000, text))
    return segments

def fetch_ffmpeg():
    ffmpeg = 'ffmpeg'
    if shutil.which(ffmpeg) is not None:
        return ffmpeg
    if XDG_CACHE_HOME in os.environ:
        cache_path = os.environ[XDG_CACHE_HOME]
    else:
        cache_path = f'{os.environ["HOME"]}/.cache'
    alt_path = f'{cache_path}/summarize'
    os.makedirs(alt_path, exist_ok=True)
    ffmpeg = f'{alt_path}/ffmpeg'
    if os.path.isfile(ffmpeg):
        return ffmpeg
    if sys.platform == 'darwin':
        from zipfile import ZipFile
        zip_data = BytesIO(requests.get('https://evermeet.cx/ffmpeg/get/zip').content)
        with ZipFile(zip_data, 'r') as zf:
            with zf.open(zf.namelist()[0]) as entry:
                with open(ffmpeg, 'wb') as target:
                    target.write(entry.read())
    elif sys.platform == 'linux':
        import tarfile
        data = BytesIO(requests.get('https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz').content)
        with tarfile.open(mode='r:xz', fileobj=data) as tar:
            member = tar.next()
            while member is not None:
                if member.name.endswith('/ffmpeg'):
                    with open(ffmpeg, 'wb') as target:
                        target.write(tar.extractfile(member).read())
                    break
                member = tar.next()
    os.chmod(ffmpeg, 0o755)
    return ffmpeg

AUDIO_RATE = 16000
N_SAMPLES = AUDIO_RATE * 60 * 5
def generate_captions(ydl, video_url, tmpdir):
    import numpy as np
    import whisper_cpp
    import ffmpeg

    ffmpeg_cmd = fetch_ffmpeg()
    ydl.download(video_url)
    samples, _ = ffmpeg.input(f'{tmpdir}/{AUDIO_FILE}').output('-', format='s16le', acodec='pcm_s16le', ac=1, ar=AUDIO_RATE).run(cmd=[ffmpeg_cmd, '-nostdin'], capture_stdout=True, capture_stderr=False)
    samples = np.frombuffer(samples, np.int16).flatten().astype(np.float32) / 32768.0
    model = huggingface_hub.hf_hub_download('ggerganov/whisper.cpp', WHISPER_MODEL)
    ws = whisper_cpp.Whisper(model, whisper_cpp.WHISPER_AHEADS_BASE_EN)
    segments = []
    for i in range(math.ceil(len(samples) / N_SAMPLES)):
        seg = ws.transcribe(samples[i * N_SAMPLES:(i + 1) * N_SAMPLES])
        for s in seg:
            segments.append(Segment(s.start + i * 300, s.end + i * 300, s.text))
    return segments

def sectionize_captions(captions):
    sections = []
    for caption in captions:
        hour = caption.start // 3600
        minute = (caption.start % 3600) // 60
        while len(sections) - 1 < hour:
            sections.append([])
        hr_sect = sections[hour]
        while len(hr_sect) - 1 < minute // 5:
            hr_sect.append([])
        hr_sect[minute // 5].append(caption.text)
    return sections

def summarize_hour(llm, hr_sect):
    summaries = []
    for min_sect in hr_sect:
        if len(min_sect) != 0:
            prompt = f'The following is a transcript of a section of a video.\n{" ".join(min_sect)}\n Based on the previous transcript, describe what is happening in this section'
            summaries.append(llm.run_llm(prompt))
        else:
            summaries.append('')
    if len(summaries) == 1:
        return HourSummary(summaries[0], [])
    all_sects = '\n'.join(summaries)
    prompt = f'The following is a set of summaries of sections of a video.\n{all_sects}\nTake those summaries of individual sections and distill it into a consolidated summary of the entire video.'
    hr_summary = llm.run_llm(prompt)
    return HourSummary(hr_summary, summaries)

def caption_in_segment(caption, segment):
    return not (caption.end < segment['segment'][0] or segment['segment'][1] < caption.start)

def caption_in_segments(caption, segments):
    return any(caption_in_segment(caption, segment) for segment in segments)

def remove_sponsored(video_id, types, captions):
    if len(types) == 0:
        return captions
    url = f'https://sponsor.ajay.app/api/skipSegments?videoID={video_id}'
    for t in types:
        url += f'&category={t}'
    resp = requests.get(url)
    if resp.status_code == 404:
        return captions
    segments = resp.json()
    return [caption for caption in captions if not caption_in_segments(caption, segments)]

def time_url_yt(url, h, fm):
    return f'{url}&t={h * 3600 + fm * 300}'

def time_url_twitch(url, h, fm):
    return f'{url}?t={h}h{fm * 5}m00s'

TIME_URL_FNS = [
    TimeUrlFn('youtube', time_url_yt),
    TimeUrlFn('twitch', time_url_twitch)
]

LOCAL_PROVIDER = 'local'
PROVIDERS = {
    LOCAL_PROVIDER: LocalLLM,
    'groq': GroqLLM
}

def main():
    parser = ArgumentParser(prog='summarize')
    parser.add_argument('video_url')
    parser.add_argument('-lp', '--llm-provider', choices = PROVIDERS.keys(), default = LOCAL_PROVIDER)
    parser.add_argument('-sb', '--sponsorblock', choices = ['sponsor', 'selfpromo', 'interaction', 'intro', 'outro', 'preview', 'music', 'offtopic', 'filler'], action = 'append', default = [])
    parser.add_argument('-lmr', '--local-model-repo', default = 'bartowski/Meta-Llama-3-8B-Instruct-GGUF')
    parser.add_argument('-lmf', '--local-model-file', default = 'mistral-7b-instruct-v0.2.Q8_0.gguf')
    parser.add_argument('-gm', '--groq-model', default = 'llama3-8b-8192')
    parser.add_argument('--force-local-transcribe', action = 'store_true')
    args = parser.parse_args()

    with TemporaryDirectory() as tmpdir:
        info = {
            'format': 'm4a/bestaudio/best',
            'paths': {'temp': tmpdir, 'home': tmpdir},
            'outtmpl': {'default': AUDIO_FILE}
        }
        with YoutubeDL(info) as ydl:
            video_info = ydl.extract_info(args.video_url, download=False)
            captions = None
            if not args.force_local_transcribe:
                captions = download_captions(video_info)
            if captions is None:
                captions = generate_captions(ydl, args.video_url, tmpdir)
    video_id = video_info['id']
    if video_info['extractor'].startswith('youtube'):
        captions = remove_sponsored(video_id, args.sponsorblock, captions)
    sections = sectionize_captions(captions)
    duration = video_info['duration']
    if duration % 300 < 60 and math.ceil(duration % 3600 / 300) == len(sections[-1]):
        sections[-1][-2].extend(sections[-1][-1])
        del sections[-1][-1]
    llm = PROVIDERS[args.llm_provider](args)
    summaries = [summarize_hour(llm, x) for x in sections]

    env = Environment()
    template_path = f'{os.path.dirname(__file__)}/template.j'
    templ = env.from_string(open(template_path).read())
    os.makedirs(OUT_DIR, exist_ok=True)
    title = video_info['title']
    filename = f'{OUT_DIR}/{video_id}.html'

    time_url = lambda x, y, z: x
    for f in TIME_URL_FNS:
        if video_info['extractor'].startswith(f.extractor):
            time_url = f.fn
            break
    with open(filename, 'w') as out:
        out.write(templ.render(
            title=title, summaries=summaries, enumerate=enumerate,
            video_url=video_info['webpage_url'], time_url=time_url
        ))
    for opener in ['open', 'xdg-open']:
        if shutil.which(opener) is not None:
            os.execlp(opener, opener, filename)
            return
    print(f'Unable to open the file automatically, the output was written to {filename}')

if __name__ == '__main__':
    main()
