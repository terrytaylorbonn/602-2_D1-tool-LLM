# nmap_06_ai_mattermost_alert.py
# scan.xml -> parse -> OpenAI summary -> Mattermost alert
# Run: python nmap_06_ai_mattermost_alert.py

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from openai import OpenAI

SCAN_FILE = Path("scan.xml")

MATTERMOST_SERVER = "http://localhost:8065"
MATTERMOST_TOKEN = "61t,,,,,,,,,,,,,,,,,,,,,,,,,,,8q9oo"
CHANNEL_ID = "4qp,,,,,,,,,,,,,,,,,,,,,,,,,,,za"


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
        "scanner": root.attrib.get("scanner"),
        "nmap_version": root.attrib.get("version"),
        "target": "127.0.0.1",
        "open_ports": open_ports,
    }


def make_ai_summary(summary: dict) -> str:
    client = OpenAI()

    prompt = f"""
You are helping analyze an authorized Nmap scan of localhost.

This is defensive security learning only.
Do not provide exploit steps or attack instructions.

Nmap JSON summary:
{json.dumps(summary, indent=2)}

Write a concise Mattermost alert with:

1. Short system summary
2. Notable services
3. Possible concerns
4. Recommended next checks

Keep it practical and beginner-friendly.
Use Markdown.
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
    )

    return response.output_text


def post_to_mattermost(message: str) -> None:
    headers = {
        "Authorization": f"Bearer {MATTERMOST_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "channel_id": CHANNEL_ID,
        "message": message,
    }

    response = requests.post(
        f"{MATTERMOST_SERVER}/api/v4/posts",
        headers=headers,
        json=payload,
        timeout=10,
    )

    print("Mattermost status:", response.status_code)
    response.raise_for_status()


def main():
    if not SCAN_FILE.exists():
        print("ERROR: scan.xml not found.")
        print("Run: nmap -sV -oX scan.xml 127.0.0.1")
        return

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.")
        return

    summary = parse_nmap_xml(SCAN_FILE)
    ai_report = make_ai_summary(summary)

    message = f"""### Nmap AI Security Alert

Target: `{summary["target"]}`  
Open ports found: **{len(summary["open_ports"])}**

---

{ai_report}
"""

    print(message)
    print("\nPosting AI report to Mattermost...\n")
    post_to_mattermost(message)


if __name__ == "__main__":
    main()