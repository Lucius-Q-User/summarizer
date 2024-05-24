# SPDX-License-Identifier: Apache-2.0

import requests
import os
from jinja2 import Environment
from collections import namedtuple
import huggingface_hub
from yt_dlp import YoutubeDL
from tempfile import TemporaryDirectory
import json
import time
import sys
import shutil
import math
from datetime import datetime
from pathlib import Path

Segment = namedtuple('Segment', ['start', 'end', 'text'])
HourSummary = namedtuple('HourSummary', ['overall', 'parts'])
TimeUrlFn = namedtuple('TimeUrlFn', ['extractor', 'fn'])
ProcessResult = namedtuple('ProcessResult', ['video_id', 'summary'])

AUDIO_FILE = 'audio.m4a'
WHISPER_DEFAULT = 'base.en'
XDG_CACHE_HOME = 'XDG_CACHE_HOME'
XDG_CONFIG_HOME = 'XDG_CONFIG_HOME'

class LocalLLM(object):
    def __init__(self,
                 local_model_repo = 'bartowski/Meta-Llama-3-8B-Instruct-GGUF',
                 local_model_file = 'Meta-Llama-3-8B-Instruct-Q8_0.gguf',
                 verbose = False,
                 **kwargs):
        from llama_cpp import Llama
        model = huggingface_hub.hf_hub_download(local_model_repo, local_model_file)
        self.llama = Llama(
             model, n_gpu_layers=-1, n_ctx=0, verbose=verbose
        )
    def run_llm(self, prompt):
        resp = self.llama.create_chat_completion(messages=[
            {'role': 'user', 'content': prompt}
        ], max_tokens=None)
        return resp['choices'][0]['message']['content']
    def save_statitstics(self):
        pass

class ChatgptLLM(object):
    def __init__(self, **kwargs):
        pass
    def run_llm(self, prompt):
        from . import chatgpt
        return chatgpt.send_request(prompt)
    def save_statitstics(self):
        pass

class HuggingchatLLM(object):
    def __init__(self, huggingchat_model = 'meta-llama/Meta-Llama-3-70B-Instruct', **kwargs):
        self.model = huggingchat_model
        self.reinitialize()
    def reinitialize(self):
        from .huggingchat import HuggingchatSession
        self.session = HuggingchatSession(self.model)
    def run_llm(self, prompt):
        from .huggingchat import TooManyRequestsError
        for i in range(5):
            try:
                return self.session.send_request(prompt)
            except TooManyRequestsError:
                self.reinitialize()
        raise Exception('Max retries exceeded')
    def save_statitstics(self):
        pass

class OpenaiLLM(object):
    def __init__(self, openai_api_key,
                 openai_model = 'llama3-8b-8192',
                 openai_base_url = 'https://api.groq.com/openai/v1',
                 **kwargs):
        self.api_key = openai_api_key
        self.model = openai_model
        self.tokens_used = 0
        self.base_url = openai_base_url
    def run_llm(self, prompt):
        req = {
            'model': self.model,
            'max_tokens': 32768,
            'messages': [
                {'role': 'user','content': prompt}
            ]
        }
        headers = {
            'Content-Type': 'application/json'
        }
        if self.api_key is not None:
            headers['Authorization'] = f'Bearer {self.api_key}'
        while 1:
            resp = requests.post(f'{self.base_url}/chat/completions', headers = headers, data = json.dumps(req))
            if resp.status_code == 429:
                time.sleep(int(resp.headers['retry-after']) + 2)
                continue
            jresp = resp.json()
            self.tokens_used += jresp['usage']['total_tokens']
            return jresp['choices'][0]['message']['content']
    def save_statitstics(self):
        alt_path = make_cache_dir()
        with open(f'{alt_path}/usage.csv', 'a') as usage:
            usage.write(f'{datetime.now().isoformat()},{self.tokens_used}\n')


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

def make_cache_dir():
    if XDG_CACHE_HOME in os.environ:
        cache_path = os.environ[XDG_CACHE_HOME]
    else:
        cache_path = f'{Path.home()}/.cache'
    alt_path = f'{cache_path}/summarize'
    os.makedirs(alt_path, exist_ok=True)
    return alt_path

AUDIO_RATE = 16000
N_SAMPLES = AUDIO_RATE * 60 * 5
def decode_audio(tmpdir, callback):
    import ctypes
    import numpy as np
    if sys.platform == 'darwin':
        name = 'libnative.dylib'
    elif sys.platform == 'linux':
        name = 'libnative.so'
    elif sys.platform == 'win32':
        name = 'native.dll'
    lib = ctypes.CDLL(f'{os.path.dirname(__file__)}/{name}')

    cb_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t)

    c_decode_audio = lib.decode_audio
    c_decode_audio.restype = ctypes.c_char_p
    c_decode_audio.argtypes = [ctypes.c_char_p, ctypes.c_size_t, cb_type]
    exc = None
    def wrap(array, size):
        try:
            ty = ctypes.c_float * size
            array = ctypes.cast(array, ctypes.POINTER(ty)).contents
            callback(np.frombuffer(array, dtype=np.float32, count=size))
        except BaseException as e:
            nonlocal exc
            exc = e
            return -1
        return 0
    err = c_decode_audio(f'{tmpdir}/{AUDIO_FILE}'.encode(), N_SAMPLES, cb_type(wrap))
    if exc is not None:
        raise exc
    if err is not None:
        raise Exception(f'Decode error: {err.decode()}')

def generate_captions(progress_hooks, duration, ydl, video_url, tmpdir, model_name, verbose):
    import whisper_cpp

    progress_hooks.phase(2, 'Downloading audio track', 1, bytes = True)
    ydl.download(video_url)
    aheads_name = model_name.replace('.', '_').replace('-', '_').upper()
    model = huggingface_hub.hf_hub_download('ggerganov/whisper.cpp', f'ggml-{model_name}.bin')
    ws = whisper_cpp.Whisper(model, getattr(whisper_cpp, f'WHISPER_AHEADS_{aheads_name}'), verbose)
    progress_hooks.phase(3, 'Generating transcript', math.ceil(duration / 300))
    segments = []
    i = 0
    def callback(samples):
        nonlocal i
        seg = ws.transcribe(samples)
        for s in seg:
            segments.append(Segment(s.start + i * 300, s.end + i * 300, s.text))
        progress_hooks.subphase_step()
        i += 1
    decode_audio(tmpdir, callback)
    return segments

def sectionize_captions(captions, duration):
    sections = []
    for i in range(duration // 3600):
        sections.append([[] for j in range(12)])
    if duration % 3600 > 0:
        sections.append([])
    for i in range(math.ceil(duration % 3600 / 300)):
        sections[-1].append([])
    for caption in captions:
        hour = caption.start // 3600
        minute = (caption.start % 3600) // 60
        hr_sect = sections[hour]
        hr_sect[minute // 5].append(caption.text)
    return sections

def summarize_hour(progress_hooks, llm, hr_sect):
    summaries = []
    for min_sect in hr_sect:
        if len(min_sect) != 0:
            prompt = f'The following is a transcript of a section of a video.\n{" ".join(min_sect)}\n Based on the previous transcript, describe what is happening in this section'
            summaries.append(llm.run_llm(prompt))
        else:
            summaries.append('')
        progress_hooks.subphase_step()
    if len(summaries) == 1:
        return HourSummary(summaries[0], [])
    all_sects = '\n'.join(summaries)
    prompt = f'The following is a set of summaries of sections of a video.\n{all_sects}\nTake those summaries of individual sections and distill it into a consolidated summary of the entire video.'
    hr_summary = llm.run_llm(prompt)
    progress_hooks.subphase_step()
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
    'openai': OpenaiLLM,
    'groq': OpenaiLLM,
    'chatgpt': ChatgptLLM,
    'huggingchat': HuggingchatLLM
}

def load_config():
    if XDG_CONFIG_HOME in os.environ:
        cfg_dir = os.environ[XDG_CONFIG_HOME]
    else:
        cfg_dir = f'{Path.home()}/.config'
    try:
        with open(f'{cfg_dir}/summarize.json') as cfgfile:
            return json.load(cfgfile)
    except FileNotFoundError:
        return {}

def ydl_progress(d, progress_hooks):
    if d['status'] not in ['downloading', 'finished']:
        return
    total = int(d.get('total_bytes', d.get('total_bytes_estimate', 0)))
    if total != 0:
        progress_hooks.set_substeps(total)
    progress_hooks.subphase_step(d['downloaded_bytes'])

class YdlLogger(object):
    def debug(self, msg):
        pass
    def info(self, msg):
        pass
    def warning(self, msg):
        pass
    def error(self, msg):
        pass

def process_video(progress_hooks, video_url, *,
                  llm_provider = LOCAL_PROVIDER,
                  sponsorblock = [],
                  whisper_model = WHISPER_DEFAULT,
                  force_local_transcribe = False,
                  verbose = False,
                  **kwargs):
    with TemporaryDirectory() as tmpdir:
        info = {
            'format': 'm4a/bestaudio/best',
            'paths': {'temp': tmpdir, 'home': tmpdir},
            'outtmpl': {'default': AUDIO_FILE},
            'progress_hooks': [lambda d: ydl_progress(d, progress_hooks)]
        }
        if not verbose:
            info['logger'] = YdlLogger()
        with YoutubeDL(info) as ydl:
            progress_hooks.phase(0, 'Getting video info')
            video_info = ydl.extract_info(video_url, download=False)
            duration = video_info['duration']
            captions = None
            if not force_local_transcribe:
                progress_hooks.phase(1, 'Downloading captions')
                captions = download_captions(video_info)
            if captions is None:
                captions = generate_captions(progress_hooks, duration, ydl, video_url, tmpdir, whisper_model, verbose)
    video_id = video_info['id']
    if video_info['extractor'].startswith('youtube'):
        captions = remove_sponsored(video_id, sponsorblock, captions)

    sections = sectionize_captions(captions, duration)
    if duration > 3600 and duration % 3600 < 60:
        sections[-2][-1].extend(sections[-1][0])
        del sections[-1]
    elif duration > 300 and duration % 300 < 60:
        sections[-1][-2].extend(sections[-1][-1])
        del sections[-1][-1]
    llm_runs = sum(len(x) for x in sections) + len(sections)
    progress_hooks.phase(4, 'Generating summaries', llm_runs)
    llm = PROVIDERS[llm_provider](verbose = verbose, **kwargs)
    try:
        summaries = [summarize_hour(progress_hooks, llm, x) for x in sections]
    finally:
        llm.save_statitstics()

    env = Environment()
    template_path = f'{os.path.dirname(__file__)}/template.html'
    templ = env.from_string(open(template_path).read())
    title = video_info['title']

    time_url = lambda x, y, z: x
    for f in TIME_URL_FNS:
        if video_info['extractor'].startswith(f.extractor):
            time_url = f.fn
            break
    summary = templ.render(
        title=title, summaries=summaries, enumerate=enumerate,
        video_url=video_info['webpage_url'], time_url=time_url
    )
    return ProcessResult(video_id, summary)
