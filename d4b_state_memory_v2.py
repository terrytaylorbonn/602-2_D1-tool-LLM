# d4b_state_memory_v2.py

import json
import os
from pathlib import Path
from datetime import datetime, timezone
from openai import OpenAI
from typing import Dict, List, Tuple

# --------------------------------------------------
# ENV / API KEY
# --------------------------------------------------
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

# --------------------------------------------------
# PATHS / STATE FILE
# --------------------------------------------------
STATE_FILE = Path("agent_state.json")

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

TOOL_REGISTRY = {
    "get_distance": get_distance,
}

TOOL_REQUIRED_ARGS = {
    "get_distance": ["city_a", "city_b"],
}

ALLOWED_TOOLS = list(TOOL_REGISTRY.keys())

# --------------------------------------------------
# PLAN SCHEMA / PROMPTS
# --------------------------------------------------
PLAN_SCHEMA_DESCRIPTION = """
Return a JSON object with this structure:

{
  "steps": [
    {
      "step_id": "s1",
      "tool": "tool_name",
      "args": {}
    }
  ]
}

Rules:
- Return valid JSON only.
- Each step must include: step_id, tool, args
"""

def build_initial_messages(user_input: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a planning agent. "
                "Your job is to convert the user's request into a JSON execution plan.\n\n"
                f"{PLAN_SCHEMA_DESCRIPTION}\n\n"
                f"Allowed tools: {ALLOWED_TOOLS}"
            ),
        },
        {"role": "user", "content": user_input},
    ]

def build_repair_messages(
    user_input: str,
    bad_plan_text: str,
    validation_errors: List[str],
) -> List[Dict[str, str]]:
    error_text = "\n".join(f"- {e}" for e in validation_errors)

    return [
        {
            "role": "system",
            "content": (
                "You are a planning agent repairing an invalid JSON execution plan.\n\n"
                f"{PLAN_SCHEMA_DESCRIPTION}\n\n"
                f"Allowed tools: {ALLOWED_TOOLS}\n\n"
                "Your previous plan was invalid. "
                "Regenerate the full plan so it passes validation."
            ),
        },
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": bad_plan_text},
        {
            "role": "user",
            "content": (
                "Validation errors:\n"
                f"{error_text}\n\n"
                "Please return a corrected JSON plan only."
            ),
        },
    ]

# --------------------------------------------------
# STATE / MEMORY HELPERS
# --------------------------------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def default_state() -> dict:
    return {
        "version": "D4b-2",
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "run_count": 0,
        "last_user_input": None,
        "last_valid_plan": None,
        "last_execution_results": None,
        "memory": {
            "last_cities_mentioned": [],
            "notes": []
        },
        "runs": []
    }

def load_state(state_file: Path = STATE_FILE) -> dict:
    if not state_file.exists():
        return default_state()

    try:
        return json.loads(state_file.read_text(encoding="utf-8"))
    except Exception:
        # fallback if state file is corrupted
        bad_backup = state_file.with_suffix(".corrupt.json")
        try:
            state_file.rename(bad_backup)
        except Exception:
            pass
        return default_state()

def save_state(state: dict, state_file: Path = STATE_FILE) -> None:
    state["updated_at"] = utc_now_iso()
    state_file.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

def extract_cities_from_plan(plan: dict) -> List[str]:
    cities: List[str] = []
    steps = plan.get("steps", [])
    for step in steps:
        args = step.get("args", {})
        for k in ["city_a", "city_b"]:
            v = args.get(k)
            if isinstance(v, str) and v not in cities:
                cities.append(v)
    return cities

def append_run_to_state(
    state: dict,
    user_input: str,
    history: List[dict],
    accepted_plan: dict,
    results: List[dict],
) -> None:
    state["run_count"] += 1
    state["last_user_input"] = user_input
    state["last_valid_plan"] = accepted_plan
    state["last_execution_results"] = results

    cities = extract_cities_from_plan(accepted_plan)
    state["memory"]["last_cities_mentioned"] = cities

    run_record = {
        "run_id": f"run_{state['run_count']}",
        "timestamp_utc": utc_now_iso(),
        "user_input": user_input,
        "planning_attempts": history,
        "accepted_plan": accepted_plan,
        "execution_results": results,
    }

    state["runs"].append(run_record)

# --------------------------------------------------
# LLM CALL
# --------------------------------------------------
def request_plan(messages: List[Dict[str, str]]) -> Tuple[dict, str]:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    plan = json.loads(content)
    return plan, content

# --------------------------------------------------
# VALIDATION
# --------------------------------------------------
def validate_plan(plan: dict) -> List[str]:
    errors: List[str] = []

    if not isinstance(plan, dict):
        return ["Plan must be a JSON object."]

    if "steps" not in plan:
        return ["Plan must contain top-level key 'steps'."]

    steps = plan["steps"]
    if not isinstance(steps, list):
        return ["'steps' must be a list."]

    if len(steps) == 0:
        return ["'steps' must not be empty."]

    expected_step_num = 1

    for i, step in enumerate(steps):
        prefix = f"steps[{i}]"

        if not isinstance(step, dict):
            errors.append(f"{prefix} must be an object.")
            continue

        for key in ["step_id", "tool", "args"]:
            if key not in step:
                errors.append(f"{prefix} missing required key: '{key}'.")

        if "step_id" in step:
            step_id = step["step_id"]
            expected_step_id = f"s{expected_step_num}"
            if not isinstance(step_id, str):
                errors.append(f"{prefix}.step_id must be a string.")
            elif step_id != expected_step_id:
                errors.append(
                    f"{prefix}.step_id must be '{expected_step_id}', got '{step_id}'."
                )

        if "tool" in step:
            tool_name = step["tool"]
            if not isinstance(tool_name, str):
                errors.append(f"{prefix}.tool must be a string.")
            elif tool_name not in TOOL_REGISTRY:
                errors.append(
                    f"{prefix}.tool '{tool_name}' is not allowed. Allowed tools: {ALLOWED_TOOLS}."
                )

        if "args" in step:
            args = step["args"]
            if not isinstance(args, dict):
                errors.append(f"{prefix}.args must be an object.")
            else:
                tool_name = step.get("tool")
                if tool_name in TOOL_REQUIRED_ARGS:
                    required_args = TOOL_REQUIRED_ARGS[tool_name]
                    for arg_name in required_args:
                        if arg_name not in args:
                            errors.append(
                                f"{prefix}.args missing required arg '{arg_name}' for tool '{tool_name}'."
                            )
                        elif not isinstance(args[arg_name], str):
                            errors.append(
                                f"{prefix}.args['{arg_name}'] must be a string."
                            )

        expected_step_num += 1

    return errors

# --------------------------------------------------
# PLAN GENERATION WITH RETRY
# --------------------------------------------------
def make_valid_plan(user_input: str, max_attempts: int = 3) -> Tuple[dict, List[dict]]:
    history: List[dict] = []
    messages = build_initial_messages(user_input)

    for attempt in range(1, max_attempts + 1):
        plan, raw_text = request_plan(messages)
        errors = validate_plan(plan)

        history.append(
            {
                "attempt": attempt,
                "raw_text": raw_text,
                "plan": plan,
                "errors": errors,
            }
        )

        if not errors:
            return plan, history

        messages = build_repair_messages(
            user_input=user_input,
            bad_plan_text=raw_text,
            validation_errors=errors,
        )

    raise ValueError("Failed to obtain a valid plan after retry attempts.")

# --------------------------------------------------
# EXECUTOR
# --------------------------------------------------
def execute_plan(plan: dict) -> List[dict]:
    results = []
    for step in plan["steps"]:
        step_id = step["step_id"]
        tool_name = step["tool"]
        args = step["args"]

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
# DISPLAY HELPERS
# --------------------------------------------------
def print_state_summary(state: dict) -> None:
    print("\n=== PRIOR STATE SUMMARY ===")
    print(f"state file        : {STATE_FILE}")
    print(f"version           : {state.get('version')}")
    print(f"run_count         : {state.get('run_count')}")
    print(f"last_user_input   : {state.get('last_user_input')}")
    print(f"last_cities       : {state.get('memory', {}).get('last_cities_mentioned', [])}")

def print_latest_memory(state: dict) -> None:
    print("\n=== CURRENT MEMORY SNAPSHOT ===")
    print(json.dumps({
        "run_count": state.get("run_count"),
        "last_user_input": state.get("last_user_input"),
        "last_valid_plan": state.get("last_valid_plan"),
        "last_execution_results": state.get("last_execution_results"),
        "memory": state.get("memory"),
    }, indent=2, ensure_ascii=False))

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    state = load_state()
    print_state_summary(state)

    # You can change this each run
    user_input = "Compare the distances from Taipeei to Kaaaohsiung and Parrris to Londddon."

    print("\n=== USER INPUT ===")
    print(user_input)

    try:
        plan, history = make_valid_plan(user_input, max_attempts=3)

        print("\n=== PLANNING ATTEMPTS ===")
        for h in history:
            print(f"\n--- attempt {h['attempt']} ---")
            print("raw plan:")
            print(h["raw_text"])
            if h["errors"]:
                print("validation errors:")
                for err in h["errors"]:
                    print(f"  - {err}")
            else:
                print("validation: PASS")

        print("\n=== ACCEPTED PLAN ===")
        print(json.dumps(plan, indent=2, ensure_ascii=False))

        results = execute_plan(plan)

        print("\n=== EXECUTION RESULTS ===")
        for r in results:
            print(
                f"{r['step_id']}: tool={r['tool']} "
                f"args={r['args']} -> result={r['result']}"
            )

        append_run_to_state(
            state=state,
            user_input=user_input,
            history=history,
            accepted_plan=plan,
            results=results,
        )
        save_state(state)

        print("\n=== STATE SAVED ===")
        print(f"Saved structured memory to: {STATE_FILE.resolve()}")

        print_latest_memory(state)

    except Exception as e:
        print("\n=== FAILURE ===")
        print(str(e))

if __name__ == "__main__":
    main()
