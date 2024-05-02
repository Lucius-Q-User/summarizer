# SPDX-License-Identifier: Apache-2.0

import requests
import random
import time
import json
import hashlib
import base64
import uuid

BASE_URL = 'https://chat.openai.com'
USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:125.0) Gecko/20100101 Firefox/125.0'
DEFAULT_HEADERS = {
    'accept': '*/*',
    'accept-language': 'en-US,en;q=0.5',
    'content-type': 'application/json',
    'oai-language': 'en-US',
    'origin': BASE_URL,
    'referer': BASE_URL,
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'user-agent': USER_AGENT,
    'priority': 'u=4'
}

def generate_proof_token(seed, diff, user_agent):
    core = random.choice([8, 12, 16, 24])
    screen = random.choice([3000, 4000, 6000])
    now = time.gmtime(time.time() - 8 * 3600)
    parse_time = time.strftime('%a, %d %b %Y %H:%M:%S GMT-0500 (Eastern Time)', now)

    diff_len = len(diff) // 2
    encoder = json.JSONEncoder(separators=(',', ':'))
    for i in range(100000):
        config = [core + screen, parse_time, 4294705152, i, user_agent]
        json_data = encoder.encode(config)
        base = base64.b64encode(json_data.encode()).decode()
        hash_value = hashlib.sha3_512((seed + base).encode()).digest()
        if hash_value.hex()[:diff_len] <= diff:
            return 'gAAAAAB' + base

    fallback_base = base64.b64encode(seed.encode()).decode()
    return 'gAAAAABwQ8Lk5FbGpA2NcR9dShT6gYjU7VxZ4D' + fallback_base

def get_new_session():
    uid = str(uuid.uuid4())
    headers = DEFAULT_HEADERS.copy()
    headers['oai-device-id'] = uid
    resp = requests.post(f'{BASE_URL}/backend-anon/sentinel/chat-requirements', headers=headers, data='{}')
    resp.raise_for_status()
    return (uid, resp.json())

def parse_sse(text):
    for event in text.split('\n\n'):
        data = []
        for line in event.split('\n'):
            if line.startswith('data: '):
                data.append(line[6:])
        data = '\n'.join(data)
        if data == '' or data == '[DONE]':
            continue
        js = json.loads(data)
        if js['error'] is not None:
            raise Exception(f'Chatgpt error: {js["error"]}')
        msg = js['message']
        if msg['author']['role'] != 'assistant':
            continue
        if msg['status'] != 'finished_successfully':
            continue
        return ''.join(msg['content']['parts'])
    raise Exception('Failed to find response message')

def send_request(prompt):
    (dev_id, session) = get_new_session()
    pow = session['proofofwork']
    token = generate_proof_token(pow['seed'], pow['difficulty'], USER_AGENT)
    body = {
        'action': 'next',
        'messages': [
            {
                'author': {
                    'role': 'user'
                },
                'content': {
                    'content_type': 'text',
                    'parts': [prompt]
                }
            }
        ],
        'parent_message_id': str(uuid.uuid4()),
        'model': 'text-davinci-002-render-sha',
        'timezone_offset_min': -180,
        'suggestions': [],
        'history_and_training_disabled': True,
        'conversation_mode': {
            'kind': 'primary_assistant'
        },
        'websocket_request_id': str(uuid.uuid4()),
    }
    headers = DEFAULT_HEADERS.copy()
    headers.update({
        'oai-device-id': dev_id,
        'openai-sentinel-chat-requirements-token': session['token'],
        'openai-sentinel-proof-token': token,
    })
    resp = requests.post(f'{BASE_URL}/backend-anon/conversation', headers=headers, data=json.dumps(body))
    resp.raise_for_status()
    return parse_sse(resp.text)
