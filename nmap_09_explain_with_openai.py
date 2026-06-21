# nmap_09_explain_with_openai.py
# MongoDB scan compare -> OpenAI explanation
# Run: python nmap_09_explain_with_openai.py

import os
from pymongo import MongoClient
from openai import OpenAI

MONGO_URI = os.getenv("MONGO_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DB_NAME = "zai_demo"
COLLECTION_NAME = "nmap_scans"

if not MONGO_URI:
    raise SystemExit("ERROR: MONGO_URI is not set.")

if not OPENAI_API_KEY:
    raise SystemExit("ERROR: OPENAI_API_KEY is not set.")


def get_port_set(scan_doc):
    return {
        int(p["port"])
        for p in scan_doc.get("open_ports", [])
    }


client_mongo = MongoClient(MONGO_URI)
db = client_mongo[DB_NAME]
collection = db[COLLECTION_NAME]

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

prompt = f"""
Explain this Nmap scan comparison in plain English.

Keep it short.
Mention whether anything changed.
Mention added ports and removed ports if any.

Scan comparison:
{summary}
"""

client_openai = OpenAI(api_key=OPENAI_API_KEY)

response = client_openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You explain network scan changes clearly and briefly."},
        {"role": "user", "content": prompt},
    ],
)

explanation = response.choices[0].message.content

print("=== Port Comparison ===")
print("Added ports:", added if added else "none")
print("Removed ports:", removed if removed else "none")
print("Same ports:", same if same else "none")
print()

print("=== OpenAI Explanation ===")
print(explanation)