# ai_demo_03_filesystem.py

# LLM = propose
# Code = executes filesystem tool

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------------
# Load API key
# -----------------------------------

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------------
# Demo file setup
# -----------------------------------

DATA_DIR = Path("demo_files")
DATA_DIR.mkdir(exist_ok=True)

(DATA_DIR / "taipei_shipments.txt").write_text(
    "Truck 12 delayed in Taipei due to flooding.\n"
    "Truck 18 on schedule in Taipei.\n",
    encoding="utf-8"
)

(DATA_DIR / "supplier_notes.txt").write_text(
    "Supplier A reported outage affecting brake components.\n",
    encoding="utf-8"
)

# -----------------------------------
# Filesystem tool
# -----------------------------------

def read_file(filename):
    safe_path = DATA_DIR / filename

    if not safe_path.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    return safe_path.read_text(encoding="utf-8")

# -----------------------------------
# Prompt
# -----------------------------------

user_prompt = "Read the Taipei shipment file."

system_prompt = """
You are an AI agent.

Return ONLY JSON.

You may use this tool:

{
  "tool": "read_file",
  "filename": "taipei_shipments.txt"
}

Allowed filenames:
- taipei_shipments.txt
- supplier_notes.txt
"""

# -----------------------------------
# LLM proposes tool call
# -----------------------------------

response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0
)

llm_output = response.choices[0].message.content

print("\nLLM OUTPUT:")
print(llm_output)

# -----------------------------------
# Code executes tool
# -----------------------------------

action = json.loads(llm_output)

if action["tool"] == "read_file":
    file_content = read_file(action["filename"])

    result = {
        "tool": "read_file",
        "filename": action["filename"],
        "content": file_content
    }

    print("\nCODE EXECUTED:")
    print(json.dumps(result, indent=2))