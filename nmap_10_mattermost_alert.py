# nmap_10_mattermost_alert.py
# MongoDB compare -> OpenAI explanation -> Mattermost API alert
# Run: python nmap_10_mattermost_alert.py

import os
import requests
from pymongo import MongoClient
from openai import OpenAI

MONGO_URI = os.getenv("MONGO_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MATTERMOST_SERVER = "http://localhost:8065"
MATTERMOST_TOKEN = "61t,,,,,,,,,,,,,,,,,,,,,,,,,,,8q9oo"
CHANNEL_ID = "4qp,,,,,,,,,,,,,,,,,,,,,,,,,,,za"

DB_NAME = "zai_demo"
COLLECTION_NAME = "nmap_scans"

if not MONGO_URI:
    raise SystemExit("ERROR: MONGO_URI is not set.")

if not OPENAI_API_KEY:
    raise SystemExit("ERROR: OPENAI_API_KEY is not set.")


def get_port_set(scan_doc):
    return {int(p["port"]) for p in scan_doc.get("open_ports", [])}


# 1. Read newest two scans from MongoDB
mongo = MongoClient(MONGO_URI)
collection = mongo[DB_NAME][COLLECTION_NAME]

scans = list(
    collection.find({"scanner": "nmap"})
    .sort("scan_time", -1)
    .limit(2)
)

if len(scans) < 2:
    raise SystemExit("Need at least 2 scans to compare.")

new_scan = scans[0]
old_scan = scans[1]

new_ports = get_port_set(new_scan)
old_ports = get_port_set(old_scan)

added = sorted(new_ports - old_ports)
removed = sorted(old_ports - new_ports)
same = sorted(new_ports & old_ports)

summary = {
    "old_scan_time": str(old_scan["scan_time"]),
    "new_scan_time": str(new_scan["scan_time"]),
    "added_ports": added,
    "removed_ports": removed,
    "unchanged_ports": same,
}

# 2. Ask OpenAI for explanation
openai_client = OpenAI(api_key=OPENAI_API_KEY)

response = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": "You explain network scan changes clearly and briefly.",
        },
        {
            "role": "user",
            "content": f"Explain this Nmap scan comparison briefly:\n{summary}",
        },
    ],
)

explanation = response.choices[0].message.content

# 3. Send Mattermost alert
message = f"""
### Nmap Scan Alert

**Added ports:** {added if added else "none"}
**Removed ports:** {removed if removed else "none"}
**Unchanged ports:** {same if same else "none"}

**Explanation:**
{explanation}
"""

headers = {
    "Authorization": f"Bearer {MATTERMOST_TOKEN}",
    "Content-Type": "application/json",
}

payload = {
    "channel_id": CHANNEL_ID,
    "message": message,
}

r = requests.post(
    f"{MATTERMOST_SERVER}/api/v4/posts",
    headers=headers,
    json=payload,
    timeout=10,
)

print("Mattermost status:", r.status_code)
print(r.text)

if r.status_code not in [200, 201]:
    raise SystemExit("Mattermost alert failed.")

print()
print("Mattermost alert sent.")
print(explanation)