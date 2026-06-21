## mattermost_02_post.py
import requests


SERVER = "http://localhost:8065"
TOKEN = "61t,,,,,,,,,,,,,,,,,,,,,,,,,,,8q9oo"
CHANNEL_ID = "4qp,,,,,,,,,,,,,,,,,,,,,,,,,,,za"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}

payload = {
    "channel_id": CHANNEL_ID,
    "message": """### Nmap Test Alert

Host: `127.0.0.1`

Detected service:
- PostgreSQL on port `5432`

Status: Python successfully posted this alert to Mattermost.
""",
}

r = requests.post(
    f"{SERVER}/api/v4/posts",
    headers=headers,
    json=payload,
)

print(r.status_code)
print(r.text)