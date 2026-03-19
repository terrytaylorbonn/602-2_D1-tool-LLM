import json
import os
from pathlib import Path
from openai import OpenAI


def load_dotenv(dotenv_path: str = ".env") -> None:
    path = Path(dotenv_path)
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is missing. Add it to the .env file.")

client = OpenAI(api_key=api_key)

# -----------------------------
# TOOL (deterministic)
# -----------------------------
def get_distance(city_a: str, city_b: str) -> float:
    # fake deterministic data (no APIs needed)
    distances = {
        ("taipei", "kaohsiung"): 350,
        ("new york", "boston"): 340,
        ("paris", "london"): 450,
    }
    key = (city_a.lower(), city_b.lower())
    return distances.get(key, 999.0)

# -----------------------------
# TOOL SCHEMA (IMPORTANT)
# -----------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_distance",
            "description": "Get distance between two cities",
            "parameters": {
                "type": "object",
                "properties": {
                    "city_a": {"type": "string"},
                    "city_b": {"type": "string"},
                },
                "required": ["city_a", "city_b"],
            },
        },
    }
]

# -----------------------------
# USER INPUT
# -----------------------------
user_input = "How far is Taipei from Kaohsiung?"

# -----------------------------
# STEP 1: LLM decides tool call
# -----------------------------
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a planner. Use tools when needed."},
        {"role": "user", "content": user_input},
    ],
    tools=tools,
    tool_choice="auto",
)

msg = response.choices[0].message

# -----------------------------
# STEP 2: Execute tool
# -----------------------------
if msg.tool_calls:
    tool_call = msg.tool_calls[0]
    args = json.loads(tool_call.function.arguments)

    result = get_distance(**args)

    print("=== TOOL CALL ===")
    print(tool_call.function.name, args)

    print("\n=== RESULT ===")
    print(result)

else:
    print("No tool call. LLM said:")
    print(msg.content)
