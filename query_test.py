import urllib.request
import json

req = urllib.request.Request(
    'http://127.0.0.1:8000/api/v1/query',
    data=json.dumps({
        'question': 'What is globalization?',
        'top_k': 3
    }).encode(),
    headers={'Content-Type': 'application/json'}
)

resp = json.loads(urllib.request.urlopen(req).read().decode())
print('ANSWER:', resp['answer'])
print()
for s in resp['sources']:
    print(f"Page {s['page']} | Score {s['score']:.3f}")