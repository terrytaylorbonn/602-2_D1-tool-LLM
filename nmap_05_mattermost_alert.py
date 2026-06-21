# nmap_05_mattermost_alert.py
# Reads scan.xml and posts a simple Nmap summary to Mattermost.
# Run: python nmap_05_mattermost_alert.py

import xml.etree.ElementTree as ET
from pathlib import Path

import requests

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
        "open_ports": open_ports,
    }


def build_message(summary: dict) -> str:
    lines = []

    lines.append("### Nmap Scan Alert")
    lines.append("")
    lines.append(f"Scanner: `{summary['scanner']}`")
    lines.append(f"Nmap version: `{summary['nmap_version']}`")
    lines.append(f"Open ports found: **{len(summary['open_ports'])}**")
    lines.append("")
    lines.append("#### Open Services")

    for p in summary["open_ports"]:
        lines.append(
            f"- `{p['port']}/{p['protocol']}` — "
            f"{p['service']} | {p['product']} | {p['version']}"
        )

    lines.append("")
    lines.append("Status: Python parsed `scan.xml` and posted this summary to Mattermost.")

    return "\n".join(lines)


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
    print(response.text)

    response.raise_for_status()


def main():
    if not SCAN_FILE.exists():
        print("ERROR: scan.xml not found.")
        print("Run: nmap -sV -oX scan.xml 127.0.0.1")
        return

    summary = parse_nmap_xml(SCAN_FILE)
    message = build_message(summary)

    print(message)
    print("\nPosting to Mattermost...\n")

    post_to_mattermost(message)


if __name__ == "__main__":
    main()