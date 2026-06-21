# nmap_03_ai_report.py
# Nmap XML -> JSON summary -> OpenAI security report
# Run: python nmap_03_ai_report.py

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from openai import OpenAI

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
        host_data = {"addresses": [], "open_ports": []}

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


def make_ai_report(summary: dict) -> str:
    client = OpenAI()

    prompt = f"""
You are helping analyze an authorized Nmap scan of localhost.

This is for a defensive security report only.
Do not provide exploit steps.
Do not provide instructions for attacking systems.

Nmap JSON summary:
{json.dumps(summary, indent=2)}

Write a concise report with these sections:

1. System Summary
2. Notable Services
3. Possible Security Concerns
4. Recommended Next Checks
5. Public Documentation Note

Keep it practical and beginner-friendly.
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
    )

    return response.output_text


def main():
    if not SCAN_FILE.exists():
        print("ERROR: scan.xml not found.")
        print("Run: nmap -sV -oX scan.xml 127.0.0.1")
        return

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.")
        return

    summary = parse_nmap_xml(SCAN_FILE)

    print("JSON SUMMARY")
    print("============")
    print(json.dumps(summary, indent=2))

    print("\nAI SECURITY REPORT")
    print("==================")
    print(make_ai_report(summary))


if __name__ == "__main__":
    main()