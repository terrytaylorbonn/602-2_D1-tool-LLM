# nmap_02_report.py
# Reads scan.xml, extracts open ports, and prints a simple report.
# Run: python nmap_02_report.py

import xml.etree.ElementTree as ET
from pathlib import Path

SCAN_FILE = Path("scan.xml")


def parse_nmap_xml(path: Path) -> dict:
    tree = ET.parse(path)
    root = tree.getroot()

    hosts = []

    for host in root.findall("host"):
        host_data = {
            "addresses": [],
            "open_ports": [],
        }

        for addr in host.findall("address"):
            host_data["addresses"].append(addr.attrib.get("addr"))

        for port in host.findall("./ports/port"):
            state = port.find("state")
            service = port.find("service")

            if state is not None and state.attrib.get("state") == "open":
                host_data["open_ports"].append({
                    "port": port.attrib.get("portid"),
                    "protocol": port.attrib.get("protocol"),
                    "service": service.attrib.get("name") if service is not None else "unknown",
                    "product": service.attrib.get("product") if service is not None else "",
                    "version": service.attrib.get("version") if service is not None else "",
                })

        hosts.append(host_data)

    return {"hosts": hosts}


def make_report(summary: dict) -> str:
    lines = []

    lines.append("NMAP SECURITY SUMMARY")
    lines.append("=====================")
    lines.append("")

    for host in summary["hosts"]:
        addresses = ", ".join(host["addresses"])
        ports = host["open_ports"]

        lines.append(f"Host: {addresses}")
        lines.append(f"Open ports found: {len(ports)}")
        lines.append("")

        lines.append("Open Services:")
        for p in ports:
            product = p["product"] or "unknown product"
            version = p["version"] or "unknown version"

            lines.append(
                f"- Port {p['port']}/{p['protocol']}: "
                f"{p['service']} | {product} | {version}"
            )

        lines.append("")
        lines.append("Basic Observations:")

        port_numbers = {p["port"] for p in ports}
        services = {p["service"] for p in ports}

        if "5432" in port_numbers:
            lines.append("- PostgreSQL is open. Confirm it is not exposed beyond localhost unless required.")

        if "8000" in port_numbers:
            lines.append("- Uvicorn is open. This may indicate a local Python/FastAPI development server.")

        if "8080" in port_numbers or "8443" in port_numbers:
            lines.append("- Web services are running on alternate HTTP/HTTPS ports.")

        if "135" in port_numbers or "445" in port_numbers:
            lines.append("- Windows networking/RPC services are open on localhost.")

        if "http" in services or "http-proxy" in services or "https-alt" in services:
            lines.append("- Multiple web-facing services were detected.")

        lines.append("")
        lines.append("Recommended Next Checks:")
        lines.append("- Confirm which services are intentionally running.")
        lines.append("- Check whether any services are reachable from other machines.")
        lines.append("- Stop unused development servers.")
        lines.append("- Avoid publishing full raw fingerprints or session IDs in public docs.")

    return "\n".join(lines)


def main():
    if not SCAN_FILE.exists():
        print("ERROR: scan.xml not found.")
        print("Run: nmap -sV -oX scan.xml 127.0.0.1")
        return

    summary = parse_nmap_xml(SCAN_FILE)
    report = make_report(summary)
    print(report)


if __name__ == "__main__":
    main()