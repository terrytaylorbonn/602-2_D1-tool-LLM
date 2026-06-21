# nmap_01_parse.py
# Reads Nmap XML output and prints a clean JSON summary.
# Run: python nmap_01_parse.py

import json
import xml.etree.ElementTree as ET
from pathlib import Path

SCAN_FILE = Path("scan.xml")


def parse_nmap_xml(path: Path) -> dict:
    tree = ET.parse(path)
    root = tree.getroot()

    result = {
        "scanner": root.attrib.get("scanner"),
        "nmap_version": root.attrib.get("version"),
        "hosts": [],
    }

    for host in root.findall("host"):
        host_data = {
            "addresses": [],
            "open_ports": [],
        }

        for addr in host.findall("address"):
            host_data["addresses"].append({
                "addr": addr.attrib.get("addr"),
                "type": addr.attrib.get("addrtype"),
            })

        for port in host.findall("./ports/port"):
            state = port.find("state")
            service = port.find("service")

            if state is not None and state.attrib.get("state") == "open":
                host_data["open_ports"].append({
                    "port": port.attrib.get("portid"),
                    "protocol": port.attrib.get("protocol"),
                    "service": service.attrib.get("name") if service is not None else None,
                    "product": service.attrib.get("product") if service is not None else None,
                    "version": service.attrib.get("version") if service is not None else None,
                })

        result["hosts"].append(host_data)

    return result


def main():
    if not SCAN_FILE.exists():
        print(f"ERROR: {SCAN_FILE} not found.")
        print("Run this first:")
        print("nmap -sV -oX scan.xml 127.0.0.1")
        return

    summary = parse_nmap_xml(SCAN_FILE)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()