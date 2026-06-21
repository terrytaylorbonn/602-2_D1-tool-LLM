# nmap_08_compare_scans.py
# Compare the two newest Nmap scan records in MongoDB
# Run: python nmap_08_compare_scans.py

import os
from pymongo import MongoClient

MONGO_URI = os.getenv("MONGO_URI")

DB_NAME = "zai_demo"
COLLECTION_NAME = "nmap_scans"

if not MONGO_URI:
    raise SystemExit("ERROR: MONGO_URI is not set.")


def get_port_set(scan_doc):
    return {
        int(p["port"])
        for p in scan_doc.get("open_ports", [])
    }


client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

scans = list(
    collection.find({"scanner": "nmap"})
    .sort("scan_time", -1)
    .limit(2)
)

print("Database:", DB_NAME)
print("Collection:", COLLECTION_NAME)
print("Nmap scan records found:", len(scans))
print()

if len(scans) < 2:
    raise SystemExit("Need at least 2 scans to compare.")

new_scan = scans[0]
old_scan = scans[1]

new_ports = get_port_set(new_scan)
old_ports = get_port_set(old_scan)

added = sorted(new_ports - old_ports)
removed = sorted(old_ports - new_ports)
same = sorted(new_ports & old_ports)

print("OLD scan:", old_scan["scan_time"])
print("NEW scan:", new_scan["scan_time"])
print()

print("Added ports:", added if added else "none")
print("Removed ports:", removed if removed else "none")
print("Same ports:", same if same else "none")