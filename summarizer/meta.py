# SPDX-License-Identifier: Apache-2.0

import requests
import uuid
import json
import random
import time
from html.parser import HTMLParser
from threading import Thread

BASE_URL = 'https://www.meta.ai/'
ROOT_HEADERS = {
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:125.0) Gecko/20100101 Firefox/125.0',
    'accept-language': 'en-US,en;q=0.5',
    'referer': BASE_URL,
    'origin': BASE_URL,
    'sec-fetch-dest': 'document',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-site': 'cross-site',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8'
}

SCRIPT_HEADERS = ROOT_HEADERS.copy()
SCRIPT_HEADERS.update({
    'accept': '*/*',
    'sec-fetch-dest': 'script',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'cross-site'
})

QUERY_HEADERS = ROOT_HEADERS.copy()
QUERY_HEADERS.update({
    'accept': '*/*',
    'content-type': 'application/x-www-form-urlencoded',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
})

class LSDTokenExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_json = False
        self.scripts = []
    def handle_starttag(self, tag, attrs):
        if tag != 'script':
            return
        attrs = dict(attrs)
        if attrs.get('type', '') == 'application/json':
            self.in_json = True
            return
        src = attrs.get('src', '')
        if src.startswith('https'):
            self.scripts.append(src)
    def handle_endtag(self, tag):
        self.in_json = False
    def handle_data(self, data):
        if not self.in_json:
            return
        data = json.loads(data)
        for req in data.get('require', []):
            if req[0] != 'ScheduledServerJS' or req[1] != 'handle':
                continue
            for v in req[3]:
                box = v['__bbox']
                if box is None:
                    continue
                for define in box.get('define', []):
                    if define[0] == 'LSD':
                        self.lsd = define[2]['token']
                for subreq in box.get('require', []):
                    if subreq[0].startswith('CometPlatformRootClient') and subreq[1] == 'setInitDeferredPayload':
                        cookies = subreq[3][0]['deferredCookies']
                        self.csrf = cookies['abra_csrf']['value']
                        self.datr = cookies['_js_datr']['value']

def extract_doc_id(name, script):
    prefix = '__d("' + name + '",[],(function(a,b,c,d,e,f){e.exports="'
    start = script.find(prefix)
    if start == -1:
        return 0
    start += len(prefix)
    tail = script[start:]
    end = tail.find('"')
    return int(tail[:end])

class Conversation(object):
    def __init__(self):
        self.uuid = str(uuid.uuid4())
        self.instance = random.randrange(0, 0x3ff)
        self.last_ts = 0
        self.counter = 0
    def make_snowflake(self):
        ts = int(time.time() * 1000)
        if ts == self.last_ts:
            self.counter += 1
        else:
            self.counter = 0
        self.last_ts = ts
        return str(ts << 22 | self.instance << 12 | self.counter)

class MetaSession(object):
    def __init__(self, proxy):
        self.proxies = {}
        if proxy is not None:
            self.proxies['https'] = proxy
        self._get_lsd_tokens()
        self._get_temp_user()
    def _get_lsd_tokens(self):
        resp = requests.get(BASE_URL, headers = ROOT_HEADERS, proxies = self.proxies)
        resp.raise_for_status()
        tokens = LSDTokenExtractor()
        tokens.feed(resp.text)
        res = [None for _ in tokens.scripts]
        threads = []
        def worker(i, url):
            res[i] = requests.get(url, headers=SCRIPT_HEADERS, proxies = self.proxies)
        for args in enumerate(tokens.scripts):
            t = Thread(target=worker, args=args)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        for r in res:
            script = r.text
            doc_id = extract_doc_id('useAbraSendMessageMutation_abraRelayOperation', script)
            if doc_id != 0:
                self.query_doc_id = doc_id
            doc_id = extract_doc_id('useAbraAcceptTOSForTempUserMutation_abraRelayOperation', script)
            if doc_id != 0:
                self.temp_user_doc_id = doc_id
        self.lsd = tokens.lsd
        self.csrf = tokens.csrf
        self.datr = tokens.datr

    def _cookies(self):
        return {
            'datr': self.datr,
            'abra_csrf': self.csrf
        }
    def _get_temp_user(self):
        vars = {
            'dob': '2000-01-01',
            'icebreaker_type': 'TEXT',
            '__relay_internal__pv__WebPixelRatiorelayprovider': 2
        }
        data = {
            'doc_id': self.temp_user_doc_id,
            'lsd': self.lsd,
            'variables': json.dumps(vars)
        }
        resp = requests.post(f'{BASE_URL}api/graphql/', data=data, headers=QUERY_HEADERS, cookies=self._cookies(), proxies = self.proxies)
        resp.raise_for_status()
        user = resp.json()['data']['xab_abra_accept_terms_of_service']['new_temp_user_auth']
        self.access_token = user['access_token']
        self.graph_api_url = user['graph_api_url']
    def new_conversation(self):
        return Conversation()
    def send_message(self, convo, prompt):
        vars = {
            'message': {
                'sensitive_string_value': prompt
            },
            'externalConversationId': convo.uuid,
            'offlineThreadingId': convo.make_snowflake(),
            'entrypoint': 'ABRA__CHAT__TEXT',
            '__relay_internal__pv__WebPixelRatiorelayprovider': 2,
            '__relay_internal__pv__AbraDebugDevOnlyrelayprovider': False,
            'icebreaker_type': 'TEXT'
        }
        data = {
            'access_token': self.access_token,
            'doc_id': self.query_doc_id,
            'lsd': self.lsd,
            'variables': json.dumps(vars)
        }
        resp = requests.post(self.graph_api_url, data=data, headers=QUERY_HEADERS, cookies=self._cookies(), proxies = self.proxies)
        resp.raise_for_status()
        for part in resp.text.split('\r\n'):
            jsp = json.loads(part)
            data = jsp['data']
            if 'node' not in data:
                continue
            msg = data['node']['bot_response_message']
            if msg['streaming_state'] == 'OVERALL_DONE':
                return msg['snippet']
        raise Exception('Failed to parse response')
