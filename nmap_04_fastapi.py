# nmap_04_fastapi.py
# Simple web UI for Nmap XML -> AI security report
# Run: uvicorn nmap_04_fastapi:app --reload

import json
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from openai import OpenAI

app = FastAPI(title="Nmap AI Security Assistant")


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
You are helping analyze an authorized Nmap scan.

This is for defensive security review only.
Do not provide exploit steps or attack instructions.

Nmap JSON summary:
{json.dumps(summary, indent=2)}

Write a concise report with:

1. System Summary
2. Notable Services
3. Possible Security Concerns
4. Recommended Next Checks
5. Public Documentation Note
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
    )

    return response.output_text


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>Nmap AI Security Assistant</title>
    </head>
    <body style="font-family: Arial; max-width: 900px; margin: 40px auto;">
        <h1>Nmap AI Security Assistant</h1>

        <p>Upload an Nmap XML file generated with:</p>

        <pre>nmap -sV -oX scan.xml 127.0.0.1</pre>

        <form action="/analyze" enctype="multipart/form-data" method="post">
            <input name="file" type="file" accept=".xml">
            <button type="submit">Analyze</button>
        </form>
    </body>
    </html>
    """


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(file: UploadFile = File(...)):
    if not os.getenv("OPENAI_API_KEY"):
        return "<h2>Error: OPENAI_API_KEY is not set.</h2>"

    contents = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        summary = parse_nmap_xml(tmp_path)
        report = make_ai_report(summary)
    finally:
        tmp_path.unlink(missing_ok=True)

    return f"""
    <html>
    <body style="font-family: Arial; max-width: 1000px; margin: 40px auto;">
        <h1>Nmap AI Security Report</h1>

        <h2>JSON Summary</h2>
        <pre>{json.dumps(summary, indent=2)}</pre>

        <h2>AI Report</h2>
        <pre style="white-space: pre-wrap;">{report}</pre>

        <p><a href="/">Analyze another file</a></p>
    </body>
    </html>
    """