# pal_core_04_graph_semantic.py
#
# PAL Core 04 - Graph + Semantic Command
#
# Commands:
#   1) demo
#   2) status
#   3) reset
#   4) related <ENTITY_ID>
#   5) path <START_ID> <END_ID>
#   6) command "<natural language instruction>"
#
# Examples:
#   python pal_core_04_graph_semantic.py demo
#   python pal_core_04_graph_semantic.py status
#   python pal_core_04_graph_semantic.py related supplier_1
#   python pal_core_04_graph_semantic.py path supplier_1 site_1
#   python pal_core_04_graph_semantic.py command "For this investigation, treat all suppliers owned by same parent company as related."
#
# Notes:
# - Structured ontology-like entities + relations
# - Deterministic graph traversal
# - OpenAI used only to convert human command -> structured semantic rule
# - Semantic rules create temporary graph edges during analysis

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from openai import OpenAI

# --------------------------------------------------
# 0 ENV / OPENAI
# --------------------------------------------------
def load_dotenv(dotenv_path: str = ".env") -> None:
    path = Path(dotenv_path)
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line or line.startswith("REM "):
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

# --------------------------------------------------
# 1 FILES / CONSTANTS
# --------------------------------------------------
STATE_FILE = Path("pal_core_graph_semantic_state.json")

COMMAND_SCHEMA_TEXT = """
Return valid JSON only, with this exact top-level structure:

{
  "action": "string",
  "relationship_rule": {
    "name": "string",
    "source_type": "string",
    "target_type": "string",
    "through_relation": "string",
    "new_relation": "string"
  }
}

Rules:
- Return valid JSON only.
- Do not include markdown.
- action must be one of:
  - "add_semantic_rule"
  - "unknown"
- Use "add_semantic_rule" only if the user is clearly defining a relationship rule.
- Example:
  Input:
    For this investigation, treat all suppliers owned by same parent company as related.
  Output:
  {
    "action": "add_semantic_rule",
    "relationship_rule": {
      "name": "same_parent_company_suppliers_are_related",
      "source_type": "supplier",
      "target_type": "supplier",
      "through_relation": "owned_by",
      "new_relation": "related_to"
    }
  }
"""

DEMO_STATE = {
    "entities": [
        {"id": "supplier_1", "type": "supplier", "name": "Supplier 1"},
        {"id": "supplier_2", "type": "supplier", "name": "Supplier 2"},
        {"id": "supplier_3", "type": "supplier", "name": "Supplier 3"},
        {"id": "parent_1", "type": "parent_company", "name": "Parent 1"},
        {"id": "parent_2", "type": "parent_company", "name": "Parent 2"},
        {"id": "site_1", "type": "site", "name": "Site 1"},
        {"id": "site_2", "type": "site", "name": "Site 2"},
        {"id": "incident_1", "type": "incident", "name": "Incident 1 Site 1"},
    ],
    "relations": [
        {"from": "supplier_1", "relation": "owned_by", "to": "parent_1"},
        {"from": "supplier_2", "relation": "owned_by", "to": "parent_1"},
        {"from": "supplier_3", "relation": "owned_by", "to": "parent_2"},
        {"from": "supplier_1", "relation": "serves", "to": "site_1"},
        {"from": "supplier_2", "relation": "serves", "to": "site_2"},
        {"from": "supplier_3", "relation": "serves", "to": "site_1"},
        {"from": "incident_1", "relation": "occurred_at", "to": "site_1"},
    ],
    "semantic_rules": []
}

# --------------------------------------------------
# 2 HELPERS
# --------------------------------------------------
def load_state() -> Dict[str, Any]:
    if not STATE_FILE.exists():
        return json.loads(json.dumps(DEMO_STATE))
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))

def save_state(state: Dict[str, Any]) -> None:
    STATE_FILE.write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

def reset_state() -> Dict[str, Any]:
    state = json.loads(json.dumps(DEMO_STATE))
    save_state(state)
    return state

def print_usage() -> None:
    print(
        "Usage:\n"
        "  python pal_core_04_graph_semantic.py demo\n"
        "  python pal_core_04_graph_semantic.py status\n"
        "  python pal_core_04_graph_semantic.py reset\n"
        "  python pal_core_04_graph_semantic.py related <ENTITY_ID>\n"
        "  python pal_core_04_graph_semantic.py path <START_ID> <END_ID>\n"
        '  python pal_core_04_graph_semantic.py command "<natural language instruction>"\n'
    )

def entity_by_id(state: Dict[str, Any], entity_id: str) -> Optional[Dict[str, Any]]:
    for e in state["entities"]:
        if e["id"] == entity_id:
            return e
    return None

def build_base_neighbors(state: Dict[str, Any]) -> Dict[str, List[Tuple[str, str]]]:
    neighbors: Dict[str, List[Tuple[str, str]]] = {}
    for e in state["entities"]:
        neighbors[e["id"]] = []

    for r in state["relations"]:
        src = r["from"]
        rel = r["relation"]
        dst = r["to"]
        neighbors.setdefault(src, []).append((rel, dst))
        neighbors.setdefault(dst, []).append((f"rev_{rel}", src))
    return neighbors

def find_entities_of_type(state: Dict[str, Any], entity_type: str) -> List[Dict[str, Any]]:
    return [e for e in state["entities"] if e["type"] == entity_type]

def build_through_index(state: Dict[str, Any], source_type: str, through_relation: str) -> Dict[str, List[str]]:
    """
    Example:
    supplier_1 --owned_by--> parent_1
    returns:
      {
        "parent_1": ["supplier_1", "supplier_2"]
      }
    for source_type=supplier, through_relation=owned_by
    """
    out: Dict[str, List[str]] = {}
    valid_ids = {e["id"] for e in find_entities_of_type(state, source_type)}

    for r in state["relations"]:
        if r["from"] in valid_ids and r["relation"] == through_relation:
            out.setdefault(r["to"], []).append(r["from"])

    return out

def build_semantic_edges(state: Dict[str, Any]) -> List[Dict[str, str]]:
    new_edges: List[Dict[str, str]] = []

    for rule in state.get("semantic_rules", []):
        source_type = rule["source_type"]
        target_type = rule["target_type"]
        through_relation = rule["through_relation"]
        new_relation = rule["new_relation"]

        if source_type != target_type:
            continue

        groups = build_through_index(state, source_type, through_relation)

        for _, members in groups.items():
            members = sorted(set(members))
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a = members[i]
                    b = members[j]
                    new_edges.append({"from": a, "relation": new_relation, "to": b})
                    new_edges.append({"from": b, "relation": new_relation, "to": a})

    return new_edges

def build_all_neighbors(state: Dict[str, Any]) -> Dict[str, List[Tuple[str, str]]]:
    neighbors = build_base_neighbors(state)

    semantic_edges = build_semantic_edges(state)
    for r in semantic_edges:
        src = r["from"]
        rel = r["relation"]
        dst = r["to"]
        neighbors.setdefault(src, []).append((rel, dst))

    return neighbors

def bfs_related(state: Dict[str, Any], start_id: str, max_hops: int = 2) -> List[Dict[str, Any]]:
    neighbors = build_all_neighbors(state)

    if start_id not in neighbors:
        return []

    results: List[Dict[str, Any]] = []
    visited: Set[str] = {start_id}
    queue: List[Tuple[str, int, Optional[str]]] = [(start_id, 0, None)]

    while queue:
        node, depth, via = queue.pop(0)
        if depth >= max_hops:
            continue

        for rel, nxt in neighbors.get(node, []):
            if nxt in visited:
                continue
            visited.add(nxt)

            ent = entity_by_id(state, nxt)
            results.append({
                "entity_id": nxt,
                "entity_type": ent["type"] if ent else "unknown",
                "via_relation": rel,
                "hops": depth + 1,
            })
            queue.append((nxt, depth + 1, rel))

    return results

def bfs_path(state: Dict[str, Any], start_id: str, end_id: str) -> Optional[List[Dict[str, str]]]:
    neighbors = build_all_neighbors(state)

    if start_id not in neighbors or end_id not in neighbors:
        return None

    queue: List[Tuple[str, List[Dict[str, str]]]] = [(start_id, [])]
    visited: Set[str] = {start_id}

    while queue:
        node, path = queue.pop(0)
        if node == end_id:
            return path

        for rel, nxt in neighbors.get(node, []):
            if nxt in visited:
                continue
            visited.add(nxt)
            queue.append((
                nxt,
                path + [{"from": node, "relation": rel, "to": nxt}]
            ))

    return None

def build_command_messages(user_command: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You convert semantic investigation instructions into structured relationship rules.\n\n"
                f"{COMMAND_SCHEMA_TEXT}"
            ),
        },
        {
            "role": "user",
            "content": user_command,
        },
    ]

def interpret_command_with_llm(user_command: str) -> Dict[str, Any]:
    if client is None:
        raise RuntimeError("OPENAI_API_KEY missing. Put it in .env or environment.")

    messages = build_command_messages(user_command)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    return json.loads(content)

# --------------------------------------------------
# 3 COMMANDS
# --------------------------------------------------
def cmd_demo() -> None:
    reset_state()
    print("DEMO OK")
    print(f"Saved state to: {STATE_FILE.resolve()}")
    print("\n=== STATUS ===")
    cmd_status()

def cmd_status() -> None:
    state = load_state()

    print("=== ENTITIES ===")
    print(json.dumps(state["entities"], indent=2, ensure_ascii=False))

    print("\n=== BASE RELATIONS ===")
    print(json.dumps(state["relations"], indent=2, ensure_ascii=False))

    print("\n=== SEMANTIC RULES ===")
    print(json.dumps(state["semantic_rules"], indent=2, ensure_ascii=False))

    print("\n=== DERIVED SEMANTIC EDGES ===")
    print(json.dumps(build_semantic_edges(state), indent=2, ensure_ascii=False))

def cmd_related(entity_id: str) -> None:
    state = load_state()

    if entity_by_id(state, entity_id) is None:
        print("RELATED FAILED")
        print(f"Unknown entity: {entity_id}")
        return

    results = bfs_related(state, entity_id, max_hops=2)

    print(f"=== RELATED TO {entity_id} ===")
    print(json.dumps(results, indent=2, ensure_ascii=False))

def cmd_path(start_id: str, end_id: str) -> None:
    state = load_state()

    if entity_by_id(state, start_id) is None:
        print("PATH FAILED")
        print(f"Unknown start entity: {start_id}")
        return
    if entity_by_id(state, end_id) is None:
        print("PATH FAILED")
        print(f"Unknown end entity: {end_id}")
        return

    path = bfs_path(state, start_id, end_id)

    print(f"=== PATH {start_id} -> {end_id} ===")
    if path is None:
        print("null")
        return
    print(json.dumps(path, indent=2, ensure_ascii=False))

def cmd_command(user_command: str) -> None:
    state = load_state()

    try:
        parsed = interpret_command_with_llm(user_command)
    except Exception as e:
        print("COMMAND FAILED")
        print(str(e))
        return

    print("=== LLM PARSED COMMAND ===")
    print(json.dumps(parsed, indent=2, ensure_ascii=False))

    action = parsed.get("action")
    if action != "add_semantic_rule":
        print("\nNo semantic rule added.")
        return

    rule = parsed.get("relationship_rule", {})
    required = {"name", "source_type", "target_type", "through_relation", "new_relation"}
    missing = [k for k in required if k not in rule]
    if missing:
        print("\nCOMMAND FAILED")
        print(f"Missing rule fields: {missing}")
        return

    existing_names = {r["name"] for r in state.get("semantic_rules", [])}
    if rule["name"] in existing_names:
        print("\nCOMMAND OK")
        print("Rule already exists.")
        return

    state.setdefault("semantic_rules", []).append(rule)
    save_state(state)

    print("\n=== RULE ADDED ===")
    print(json.dumps(rule, indent=2, ensure_ascii=False))

    print("\n=== DERIVED SEMANTIC EDGES ===")
    print(json.dumps(build_semantic_edges(state), indent=2, ensure_ascii=False))

# --------------------------------------------------
# 4 MAIN
# --------------------------------------------------
def main() -> None:
    if len(sys.argv) < 2:
        print_usage()
        return

    command = sys.argv[1].strip().lower()

    if command == "demo":
        cmd_demo()

    elif command == "status":
        cmd_status()

    elif command == "reset":
        reset_state()
        print("RESET OK")
        print(f"Saved state to: {STATE_FILE.resolve()}")

    elif command == "related":
        if len(sys.argv) < 3:
            print("Missing ENTITY_ID\n")
            print_usage()
            return
        cmd_related(sys.argv[2])

    elif command == "path":
        if len(sys.argv) < 4:
            print("Missing START_ID and END_ID\n")
            print_usage()
            return
        cmd_path(sys.argv[2], sys.argv[3])

    elif command == "command":
        if len(sys.argv) < 3:
            print("Missing natural language command\n")
            print_usage()
            return
        cmd_command(sys.argv[2])

    else:
        print(f"Unknown command: {command}\n")
        print_usage()

if __name__ == "__main__":
    main()
