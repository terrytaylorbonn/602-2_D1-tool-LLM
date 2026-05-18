# ai_demo_01_tool.py

# Basic tool demo:
# LLM = propose
# Code = executes

import json

# -----------------------------
# 1. Fake LLM output
# -----------------------------
# In real agentic AI, the LLM would generate this JSON.
# For now, we hardcode it so the tool concept is clear.

llm_output = """
{
  "tool": "calculator",
  "operation": "multiply",
  "a": 12,
  "b": 7
}
"""

# -----------------------------
# 2. Tool implementation
# -----------------------------

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

# -----------------------------
# 3. External agent executes tool
# -----------------------------

def run_agent(llm_json):
    action = json.loads(llm_json)

    if action["tool"] == "calculator":
        result = calculator(
            action["operation"],
            action["a"],
            action["b"]
        )

        return {
            "tool": action["tool"],
            "operation": action["operation"],
            "result": result
        }

    raise ValueError(f"Unknown tool: {action['tool']}")

# -----------------------------
# 4. Run demo
# -----------------------------

result = run_agent(llm_output)

print("LLM proposed:")
print(llm_output)

print("\nCode executed:")
print(json.dumps(result, indent=2))