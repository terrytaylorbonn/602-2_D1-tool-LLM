# nmap_07_store_scan.py
# scan.xml -> parse -> store scan record in MongoDB
# Run: python nmap_07_store_scan.py

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from pymongo import MongoClient

SCAN_FILE = Path("scan.xml")

# MONGO_URI = "mongodb://localhost:27017"
import os
MONGO_URI = os.getenv("MONGO_URI")

DB_NAME = "zai_demo"
COLLECTION_NAME = "nmap_scans"


def parse_nmap_xml(path: Path) -> dict:
    tree = ET.parse(path)
    root = tree.getroot()

    open_ports = []

    for port in root.findall(".//ports/port"):
        state = port.find("state")
        service = port.find("service")

        if state is not None and state.attrib.get("state") == "open":
            open_ports.append({
                "port": port.attrib.get("portid"),
                "protocol": port.attrib.get("protocol"),
                "service": service.attrib.get("name") if service is not None else "unknown",
                "product": service.attrib.get("product") if service is not None else "unknown",
                "version": service.attrib.get("version") if service is not None else "unknown",
            })

    return {
        "scan_time": datetime.now(timezone.utc),
        "scanner": root.attrib.get("scanner"),
        "nmap_version": root.attrib.get("version"),
        "target": "127.0.0.1",
        "open_ports": open_ports,
        "open_port_count": len(open_ports),
    }


def main():

    if not MONGO_URI:
        print("ERROR: MONGO_URI is not set.")
        return
    
    if not SCAN_FILE.exists():
        print("ERROR: scan.xml not found.")
        print("Run: nmap -sV -oX scan.xml 127.0.0.1")
        return


    scan_doc = parse_nmap_xml(SCAN_FILE)

    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    result = collection.insert_one(scan_doc)

    print("Stored Nmap scan in MongoDB.")
    print("Database:", DB_NAME)
    print("Collection:", COLLECTION_NAME)
    print("Inserted ID:", result.inserted_id)
    print("Open ports:", scan_doc["open_port_count"])

    for p in scan_doc["open_ports"]:
        print(f"- {p['port']}/{p['protocol']} {p['service']} {p['product']} {p['version']}")


if __name__ == "__main__":
    main()