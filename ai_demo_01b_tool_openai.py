# ai_demo_01b_tool_openai.py
# Basic OpenAI tool-calling style demo

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------------
# Load API key
# -----------------------------------

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# -----------------------------------
# Tool implementation
# -----------------------------------

def calculator(operation, a, b):

    if operation == "add":
        return a + b

    if operation == "subtract":
        return a - b

    if operation == "multiply":
        return a * b

    if operation == "divide":
        return a / b

    raise ValueError(f"Unknown operation: {operation}")

# -----------------------------------
# Prompt
# -----------------------------------

user_prompt = "What is 11 multiplied by 6?"

system_prompt = """
You are an AI agent.

Return ONLY JSON.

Valid format:

{
  "tool": "calculator",
  "operation": "multiply",
  "a": 12,
  "b": 7
}
"""

# -----------------------------------
# Call OpenAI
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
# Parse JSON
# -----------------------------------

action = json.loads(llm_output)

# -----------------------------------
# External agent executes tool
# -----------------------------------

if action["tool"] == "calculator":

    result = calculator(
        action["operation"],
        action["a"],
        action["b"]
    )

    final_result = {
        "tool": action["tool"],
        "operation": action["operation"],
        "result": result
    }

    print("\nCODE EXECUTED:")
    print(json.dumps(final_result, indent=2))