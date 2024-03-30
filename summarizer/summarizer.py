# SPDX-License-Identifier: Apache-2.0

import requests
import sys
import webvtt
from io import StringIO, BytesIO
from llama_cpp import Llama
import urllib.parse as parse
import os
from jinja2 import Environment
from collections import namedtuple
import huggingface_hub
import time
from yt_dlp import YoutubeDL
from tempfile import TemporaryDirectory

Segment = namedtuple('Segment', ['hour', 'minute', 'text'])
HourSummary = namedtuple('HourSummary', ['overall', 'parts'])

OUT_DIR = 'out'
AUDIO_FILE = 'audio.m4a'
WHISPER_MODEL = 'ggml-base.en.bin'
MISTRAL_MODEL = 'mistral-7b-instruct-v0.2.Q8_0.gguf'

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

def download_captions(video_info):
    url = extract_sub_url(video_info)
    if url is None:
        return None
    vtt = requests.get(url).text
    segments = []
    for caption in webvtt.read_buffer(StringIO(vtt)):
        start_parts = caption.start.split(':')
        hour = int(start_parts[0])
        minute = int(start_parts[1])
        second, _ = start_parts[2].split('.')
        segments.append(Segment(hour, minute, caption.text))
    return segments

def fetch_ffmpeg():
    import shutil
    ffmpeg = 'ffmpeg'
    if shutil.which(ffmpeg) is not None:
        return ffmpeg
    from zipfile import ZipFile
    alt_path = f'{os.environ["HOME"]}/.cache/summarize'
    os.makedirs(alt_path, exist_ok=True)
    ffmpeg = f'{alt_path}/ffmpeg'
    if os.path.isfile(ffmpeg):
        return ffmpeg
    zip_data = BytesIO(requests.get('https://evermeet.cx/ffmpeg/get/zip').content)
    with ZipFile(zip_data, 'r') as zf:
        with zf.open(zf.namelist()[0]) as entry:
            with open(ffmpeg, 'wb') as target:
                target.write(entry.read())
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
    return [Segment(x.start // 3600, x.start // 60 % 60, x.text) for x in ws.transcribe(samples)]

def sectionize_captions(captions):
    sections = []
    for caption in captions:
        if len(sections) - 1 < caption.hour:
            sections.append([])
        hr_sect = sections[caption.hour]
        if len(hr_sect) - 1 < caption.minute // 5:
            hr_sect.append([])
        hr_sect[caption.minute // 5].append(caption.text)
    return sections

def cleanup(st):
    if len(st) == 0:
        return st
    for i, v in enumerate(st):
        if v.isalnum():
            break
    return st[i:]

def summarize_hour(llm, hr_sect):
    summaries = []
    for min_sect in hr_sect:
        prompt = f'[INST] The following is a transcript of a section of a video.\n{" ".join(min_sect)}\n Based on the previous transcript, describe what is happening in this section [/INST]'
        summaries.append(cleanup(llm(prompt, max_tokens=None)['choices'][0]['text']))
        llm.reset()
    all_sects = '\n'.join(summaries)
    prompt = f'[INST] The following is a set of summaries of sections of a video.\n{all_sects}\nTake those summaries of individual sections and distill it into a consolidated summary of the entire video. [/INST]'
    hr_summary = cleanup(llm(prompt, max_tokens=None)['choices'][0]['text'])
    return HourSummary(hr_summary, summaries)

def summarize_all(sections):
    model = model = huggingface_hub.hf_hub_download('TheBloke/Mistral-7B-Instruct-v0.2-GGUF', MISTRAL_MODEL)
    llm = Llama(
        model, n_gpu_layers=-1, n_ctx=32768#, verbose=False
    )
    return [summarize_hour(llm, x) for x in sections]

def main():
    video_url = sys.argv[1]
    with TemporaryDirectory() as tmpdir:
        info = {
            'format': 'm4a/bestaudio/best',
            'paths': {'temp': tmpdir, 'home': tmpdir},
            'outtmpl': {'default': AUDIO_FILE}
        }
        with YoutubeDL(info) as ydl:
            video_info = ydl.extract_info(video_url, download=False)
            captions = download_captions(video_info)
            if captions is None:
                captions = generate_captions(ydl, video_url, tmpdir)
    sections = sectionize_captions(captions)
    summaries = summarize_all(sections)

    env = Environment()
    template_path = f'{os.path.dirname(__file__)}/template.j'
    templ = env.from_string(open(template_path).read())
    os.makedirs(OUT_DIR, exist_ok=True)
    title = video_info['title']
    video_id = video_info['id']
    filename = f'{OUT_DIR}/{video_id}.html'
    with open(filename, 'w') as out:
        out.write(templ.render(title=title, video_id=video_id, summaries=summaries, enumerate=enumerate))
    os.execlp('open', 'open', filename)

if __name__ == '__main__':
    main()
