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

#######################################

# import json
# from openai import OpenAI

# client = OpenAI()

# --------------------------------------------------
# TOOLS (deterministic)
# --------------------------------------------------
def get_distance(city_a: str, city_b: str) -> float:
    distances = {
        ("taipei", "kaohsiung"): 350,
        ("new york", "boston"): 340,
        ("paris", "london"): 450,
        ("tokyo", "osaka"): 500,
    }
    key = (city_a.lower(), city_b.lower())
    return distances.get(key, 999.0)


# Tool registry
TOOL_REGISTRY = {
    "get_distance": get_distance,
}


# --------------------------------------------------
# PLAN SCHEMA
# --------------------------------------------------
PLAN_SCHEMA_DESCRIPTION = """
Return a JSON object with this exact structure:

{
  "steps": [
    {
      "step_id": "s1",
      "tool": "get_distance",
      "args": {
        "city_a": "Taipei",
        "city_b": "Kaohsiung"
      }
    }
  ]
}

Rules:
- Return valid JSON only.
- Do not include markdown fences.
- Use only the tool name: get_distance
- Each step must include: step_id, tool, args
- step_id should be s1, s2, s3, ...
"""


# --------------------------------------------------
# STEP 1: LLM MAKES PLAN
# --------------------------------------------------
def make_plan(user_input: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a planning agent. "
                    "Your job is to convert the user's request into a JSON execution plan. "
                    + PLAN_SCHEMA_DESCRIPTION
                ),
            },
            {"role": "user", "content": user_input},
        ],
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    return json.loads(content)


# --------------------------------------------------
# STEP 2: EXECUTOR RUNS PLAN
# --------------------------------------------------
def execute_plan(plan: dict) -> list[dict]:
    results = []

    steps = plan.get("steps", [])
    if not isinstance(steps, list):
        raise ValueError("Plan must contain a list under 'steps'.")

    for step in steps:
        step_id = step.get("step_id")
        tool_name = step.get("tool")
        args = step.get("args", {})

        if tool_name not in TOOL_REGISTRY:
            raise ValueError(f"Unknown tool: {tool_name}")

        tool_fn = TOOL_REGISTRY[tool_name]
        result = tool_fn(**args)

        results.append(
            {
                "step_id": step_id,
                "tool": tool_name,
                "args": args,
                "result": result,
            }
        )

    return results


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    user_input = (
        "Compare the distances from Taipei to Kaohsiung and from Paris to London."
    )

    print("=== USER INPUT ===")
    print(user_input)

    plan = make_plan(user_input)

    print("\n=== PLAN ===")
    print(json.dumps(plan, indent=2))

    results = execute_plan(plan)

    print("\n=== EXECUTION RESULTS ===")
    for r in results:
        print(
            f"{r['step_id']}: tool={r['tool']} "
            f"args={r['args']} -> result={r['result']}"
        )


if __name__ == "__main__":
    main()

