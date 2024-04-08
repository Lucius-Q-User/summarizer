# SPDX-License-Identifier: Apache-2.0

import requests
import webvtt
from io import StringIO, BytesIO
import urllib.parse as parse
import os
from jinja2 import Environment
from collections import namedtuple
import huggingface_hub
from yt_dlp import YoutubeDL
from tempfile import TemporaryDirectory
from argparse import ArgumentParser
from threading import Thread
import json
import time
import sys
import shutil


Segment = namedtuple('Segment', ['start', 'end', 'text'])
HourSummary = namedtuple('HourSummary', ['overall', 'parts'])

OUT_DIR = 'out'
AUDIO_FILE = 'audio.m4a'
WHISPER_MODEL = 'ggml-base.en.bin'
MISTRAL_MODEL = 'mistral-7b-instruct-v0.2.Q8_0.gguf'
GROQ_API_KEY_VAR = 'GROQ_API_KEY'
XDG_CACHE_HOME = 'XDG_CACHE_HOME'
XDG_CONFIG_HOME = 'XDG_CONFIG_HOME'

class LocalLLMWorker(object):
    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt
    def join(self):
        inst = f'[INST] {self.prompt} [/INST]'
        return self.cleanup(self.llm.llama(inst, max_tokens=None)['choices'][0]['text'])
    def cleanup(self, st):
        if len(st) == 0:
            return st
        for i, v in enumerate(st):
            if v.isalnum():
                break
        return st[i:]

class LocalLLM(object):
    def __init__(self):
        from llama_cpp import Llama
        model = huggingface_hub.hf_hub_download('TheBloke/Mistral-7B-Instruct-v0.2-GGUF', MISTRAL_MODEL)
        self.llama = Llama(
             model, n_gpu_layers=-1, n_ctx=32768#, verbose=False
        )
    def run_llm(self, prompt):
        return LocalLLMWorker(self, prompt)

class GroqLLMWorker(object):
    def __init__(self, api_key, prompt):
        self.api_key = api_key
        self.prompt = prompt
        self.thread = Thread(target = self)
        self.thread.start()
    def __call__(self):
        req = {
            'model': 'mixtral-8x7b-32768',
            'max_tokens': 32768,
            'messages': [
                {'role': 'user','content': self.prompt}
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
            self.result = resp.json()['choices'][0]['message']['content']
            return

    def join(self):
        self.thread.join()
        return self.result

class GroqLLM(object):
    def __init__(self):
        if GROQ_API_KEY_VAR in os.environ:
            self.api_key = os.environ[GROQ_API_KEY_VAR]
        else:
            if XDG_CONFIG_HOME in os.environ:
                cfg_dir = os.environ[XDG_CONFIG_HOME]
            else:
                cfg_dir = f'{os.environ["HOME"]}/.config'
            self.api_key = json.load(open(f'{cfg_dir}/summarize.json'))[GROQ_API_KEY_VAR]
    def run_llm(self, prompt):
        return GroqLLMWorker(self.api_key, prompt)

def find_vtt(fmts):
    for fmt in fmts:
        if fmt['ext'] == 'vtt':
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
            url = find_vtt(subs[lang])
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
        return find_vtt(subs[lang])
    return None

def time_to_secs(time):
    parts = time.split(':')
    hour = int(parts[0])
    minute = int(parts[1])
    second, _ = parts[2].split('.')
    return hour * 3600 + minute * 60 + int(second)

def download_captions(video_info):
    url = extract_sub_url(video_info)
    if url is None:
        return None
    vtt = requests.get(url).text
    segments = []
    for caption in webvtt.read_buffer(StringIO(vtt)):
        start = time_to_secs(caption.start)
        end = time_to_secs(caption.end)
        segments.append(Segment(start, end, caption.text))
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


def generate_captions(ydl, video_url, tmpdir):
    import numpy as np
    import whisper_cpp
    import ffmpeg

    ffmpeg_cmd = fetch_ffmpeg()
    ydl.download(video_url)
    samples, _ = ffmpeg.input(f'{tmpdir}/{AUDIO_FILE}').output('-', format='s16le', acodec='pcm_s16le', ac=1, ar=16000).run(cmd=[ffmpeg_cmd, '-nostdin'], capture_stdout=True, capture_stderr=False)
    samples = np.frombuffer(samples, np.int16).flatten().astype(np.float32) / 32768.0
    model = huggingface_hub.hf_hub_download('ggerganov/whisper.cpp', WHISPER_MODEL)
    ws = whisper_cpp.Whisper(model, whisper_cpp.WHISPER_AHEADS_BASE_EN)
    return ws.transcribe(samples)

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
            summaries.append(None)
    summaries = [(x.join() if x is not None else "") for x in summaries]
    all_sects = '\n'.join(summaries)
    prompt = f'The following is a set of summaries of sections of a video.\n{all_sects}\nTake those summaries of individual sections and distill it into a consolidated summary of the entire video.'
    hr_summary = llm.run_llm(prompt).join()
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
    segments = requests.get(url).json()
    return [caption for caption in captions if not caption_in_segments(caption, segments)]

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
    args = parser.parse_args()

    with TemporaryDirectory() as tmpdir:
        info = {
            'format': 'm4a/bestaudio/best',
            'paths': {'temp': tmpdir, 'home': tmpdir},
            'outtmpl': {'default': AUDIO_FILE}
        }
        with YoutubeDL(info) as ydl:
            video_info = ydl.extract_info(args.video_url, download=False)
            captions = download_captions(video_info)
            if captions is None:
                captions = generate_captions(ydl, args.video_url, tmpdir)
    video_id = video_info['id']
    captions = remove_sponsored(video_id, args.sponsorblock, captions)
    sections = sectionize_captions(captions)
    llm = PROVIDERS[args.llm_provider]()
    summaries = [summarize_hour(llm, x) for x in sections]

    env = Environment()
    template_path = f'{os.path.dirname(__file__)}/template.j'
    templ = env.from_string(open(template_path).read())
    os.makedirs(OUT_DIR, exist_ok=True)
    title = video_info['title']
    filename = f'{OUT_DIR}/{video_id}.html'
    with open(filename, 'w') as out:
        out.write(templ.render(title=title, video_id=video_id, summaries=summaries, enumerate=enumerate))
    for opener in ['open', 'xdg-open']:
        if shutil.which(opener) is not None:
            os.execlp(opener, opener, filename)
            return
    print(f'Unable to open the file automatically, the output was written to {filename}')

if __name__ == '__main__':
    main()
