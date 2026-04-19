#!/usr/bin/env python3
"""
pal_core_06_meta_llm_roles.py

One compact meta-demo showing the main ways an LLM can add value
inside a structured / deterministic PAL-style application.

Design goal
-----------
Keep the execution core deterministic.
Use LLM only at the boundaries where language, semantics, rule authoring,
and explanations add value.

This script demonstrates 8 roles:
1) Human Command Interpreter
2) Messy Data Ingest / Extraction
3) Planner / Task Decomposer
4) Rule Generator / Rule Injection
5) Semantic Mapping / Ontology Translation
6) Search / Retrieval Helper
7) Explanation / Summarization
8) Interactive Analyst Assistant

How to run
----------
python pal_core_06_meta_llm_roles.py demo
python pal_core_06_meta_llm_roles.py demo --llm off
python pal_core_06_meta_llm_roles.py role command
python pal_core_06_meta_llm_roles.py role ingest
python pal_core_06_meta_llm_roles.py role planner
python pal_core_06_meta_llm_roles.py role rulegen
python pal_core_06_meta_llm_roles.py role ontology
python pal_core_06_meta_llm_roles.py role retrieval
python pal_core_06_meta_llm_roles.py role explain
python pal_core_06_meta_llm_roles.py role analyst

Notes
-----
- By default this file runs in MOCK LLM mode, so it is deterministic and easy to study.
- You can later replace MockLLM with a real adapter (OpenAI / local model / etc.).
- The ontology/graph and execution engine remain deterministic.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# 1) Shared helpers
# ============================================================

def jprint(x: Any):
    print(json.dumps(x, indent=2, ensure_ascii=False))


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


# ============================================================
# 2) Deterministic in-memory ontology / graph store
# ============================================================

@dataclass
class Entity:
    entity_id: str
    entity_type: str
    name: str
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Event:
    event_id: str
    entity_id: str
    event_type: str
    location: str
    status: str
    note: str = ""
    supplier_id: Optional[str] = None
    route_id: Optional[str] = None
    site_id: Optional[str] = None
    severity: int = 1


class OntologyStore:
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.events: List[Event] = []
        self.edges_out: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.edges_in: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    # --------------------------------------------------------
    # 2.1 Core record mutation helpers
    # --------------------------------------------------------
    def add_entity(self, entity: Entity):
        self.entities[entity.entity_id] = entity

    def add_edge(self, src: str, rel: str, dst: str):
        self.edges_out[src].append((rel, dst))
        self.edges_in[dst].append((rel, src))

    def add_event(self, event: Event):
        self.events.append(event)

    # --------------------------------------------------------
    # 2.2 Graph traversal helpers
    # --------------------------------------------------------
    def neighbors(self, node_id: str) -> List[Dict[str, str]]:
        out = []
        for rel, dst in self.edges_out.get(node_id, []):
            out.append({"from": node_id, "relation": rel, "to": dst})
        for rel, src in self.edges_in.get(node_id, []):
            out.append({"from": src, "relation": rel, "to": node_id})
        return out

    def two_hop_impacts_from_site(self, site_id: str) -> List[Dict[str, Any]]:
        results = []
        seen = set()
        for rel1, n1 in self.edges_in.get(site_id, []) + self.edges_out.get(site_id, []):
            key1 = (site_id, rel1, n1, 1)
            if key1 not in seen:
                results.append({
                    "entity_id": n1,
                    "via_relation": rel1,
                    "hops": 1,
                })
                seen.add(key1)
            for rel2, n2 in self.edges_in.get(n1, []) + self.edges_out.get(n1, []):
                if n2 == site_id:
                    continue
                key2 = (site_id, n1, rel2, n2, 2)
                if key2 not in seen:
                    results.append({
                        "entity_id": n2,
                        "via_relation": f"{rel1}->{rel2}",
                        "hops": 2,
                    })
                    seen.add(key2)
        return results

    # --------------------------------------------------------
    # 2.3 Deterministic query / aggregation helpers
    # --------------------------------------------------------
    def filter_events(self, filt: Dict[str, Any]) -> List[Event]:
        out = []
        for ev in self.events:
            ok = True
            for k, v in filt.items():
                if v is None:
                    continue
                if not hasattr(ev, k):
                    ok = False
                    break
                if getattr(ev, k) != v:
                    ok = False
                    break
            if ok:
                out.append(ev)
        return out

    def aggregate_counts(self, events: List[Event], field_name: str) -> Dict[str, int]:
        c = Counter()
        for ev in events:
            c[str(getattr(ev, field_name, None))] += 1
        return dict(c)

    def top_impacted_suppliers(self, events: List[Event]) -> List[Dict[str, Any]]:
        c = Counter()
        for ev in events:
            if ev.supplier_id:
                c[ev.supplier_id] += 1
        out = []
        for supplier_id, count in c.most_common():
            name = self.entities.get(supplier_id).name if supplier_id in self.entities else supplier_id
            out.append({"supplier_id": supplier_id, "supplier_name": name, "count": count})
        return out


# ============================================================
# 3) Demo data seeding
# ============================================================

def seed_demo_store() -> OntologyStore:
    db = OntologyStore()

    # entities
    db.add_entity(Entity("site_1", "site", "Site 1", {"city": "taipei"}))
    db.add_entity(Entity("site_2", "site", "Site 2", {"city": "tainan"}))
    db.add_entity(Entity("route_7", "route", "Route 7", {"region": "north"}))
    db.add_entity(Entity("route_9", "route", "Route 9", {"region": "south"}))
    db.add_entity(Entity("supplier_1", "supplier", "Acme Rubber"))
    db.add_entity(Entity("supplier_2", "supplier", "Bolt Logistics"))
    db.add_entity(Entity("supplier_3", "supplier", "Metro Steel"))
    db.add_entity(Entity("truck_17", "shipment", "Truck 17"))
    db.add_entity(Entity("truck_21", "shipment", "Truck 21"))
    db.add_entity(Entity("truck_33", "shipment", "Truck 33"))

    # graph relations
    db.add_edge("supplier_1", "supplies", "site_1")
    db.add_edge("supplier_2", "supplies", "site_1")
    db.add_edge("supplier_2", "supplies", "site_2")
    db.add_edge("supplier_3", "supplies", "site_2")
    db.add_edge("truck_17", "uses_route", "route_7")
    db.add_edge("truck_21", "uses_route", "route_7")
    db.add_edge("truck_33", "uses_route", "route_9")
    db.add_edge("route_7", "serves", "site_1")
    db.add_edge("route_9", "serves", "site_2")
    db.add_edge("supplier_1", "related_to", "supplier_2")

    # events
    db.add_event(Event("E001", "truck_17", "shipment", "taipei", "delayed", "flat tire on highway", "supplier_1", "route_7", "site_1", 2))
    db.add_event(Event("E002", "truck_21", "shipment", "taipei", "blocked", "road closure near tunnel", "supplier_2", "route_7", "site_1", 3))
    db.add_event(Event("E003", "truck_33", "shipment", "tainan", "delayed", "supplier backlog", "supplier_3", "route_9", "site_2", 2))
    db.add_event(Event("E004", "truck_17", "shipment", "taipei", "delayed", "blowout and late unloading", "supplier_1", "route_7", "site_1", 2))
    db.add_event(Event("E005", "truck_21", "shipment", "taipei", "normal", "route reopened", "supplier_2", "route_7", "site_1", 1))
    db.add_event(Event("E006", "truck_33", "shipment", "tainan", "blocked", "port equipment outage", "supplier_3", "route_9", "site_2", 3))

    return db


# ============================================================
# 4) Mock LLM boundary adapter
# ============================================================

class MockLLM:
    """
    Deterministic mock that simulates LLM outputs.
    This keeps the demo runnable and easy to inspect.
    """

    def command_to_json(self, text: str) -> Dict[str, Any]:
        t = norm_text(text)
        if "delayed shipments in taipei" in t:
            return {
                "action": "query_events",
                "filter": {"status": "delayed", "location": "taipei"},
            }
        if "what affects site 1" in t:
            return {
                "action": "graph_impact",
                "site_id": "site_1",
            }
        return {
            "action": "query_events",
            "filter": {},
        }

    def extract_event_json(self, raw_text: str) -> Dict[str, Any]:
        t = norm_text(raw_text)
        location = "taipei" if "taipei" in t else "tainan"
        status = "delayed" if ("delay" in t or "late" in t) else "normal"
        supplier_id = "supplier_1" if ("acme" in t or "rubber" in t) else None
        route_id = "route_7" if "route 7" in t else None
        site_id = "site_1" if location == "taipei" else "site_2"
        note = raw_text.strip()
        return {
            "entity_id": "truck_900",
            "event_type": "shipment",
            "location": location,
            "status": status,
            "note": note,
            "supplier_id": supplier_id,
            "route_id": route_id,
            "site_id": site_id,
            "severity": 2 if status == "delayed" else 1,
        }

    def make_plan(self, text: str) -> Dict[str, Any]:
        t = norm_text(text)
        if "compare delayed shipments in taipei vs blocked shipments in tainan" in t:
            return {
                "steps": [
                    {
                        "op": "filter_events",
                        "name": "A",
                        "args": {"status": "delayed", "location": "taipei"},
                    },
                    {
                        "op": "filter_events",
                        "name": "B",
                        "args": {"status": "blocked", "location": "tainan"},
                    },
                    {
                        "op": "aggregate_counts",
                        "name": "A_counts",
                        "input": "A",
                        "field": "supplier_id",
                    },
                    {
                        "op": "aggregate_counts",
                        "name": "B_counts",
                        "input": "B",
                        "field": "supplier_id",
                    },
                    {
                        "op": "compare_named_results",
                        "name": "cmp",
                        "left": "A_counts",
                        "right": "B_counts",
                    },
                ]
            }
        return {
            "steps": [
                {"op": "filter_events", "name": "A", "args": {}},
                {"op": "aggregate_counts", "name": "A_counts", "input": "A", "field": "status"},
            ]
        }

    def generate_rule_json(self, text: str) -> Dict[str, Any]:
        return {
            "rule_id": "R100",
            "name": "outage_delay_supplier_overlap",
            "when": {
                "all": [
                    {"field": "status", "in": ["delayed", "blocked"]},
                    {"field": "note_contains_any", "value": ["outage", "closure", "backlog"]},
                    {"field": "supplier_id", "exists": True},
                ]
            },
            "then": {
                "action": "create_alert",
                "priority": 9,
                "reason": "compound disruption: outage + delay/blocked + supplier overlap",
            },
        }

    def semantic_map(self, phrase: str) -> Dict[str, str]:
        t = norm_text(phrase)
        if "flat tire" in t or "blowout" in t:
            return {"canonical": "tire_failure"}
        if "road closure" in t or "tunnel blocked" in t:
            return {"canonical": "route_blockage"}
        return {"canonical": t.replace(" ", "_")}

    def retrieval_query(self, text: str) -> Dict[str, Any]:
        t = norm_text(text)
        if "what affects site 1" in t:
            return {"type": "graph_impact", "site_id": "site_1", "depth": 2}
        return {"type": "event_search", "filter": {}}

    def explain_result(self, result: Dict[str, Any]) -> str:
        if result.get("kind") == "query_events":
            return f"Found {result['count']} matching events. Main statuses and suppliers were computed by deterministic filters over the ontology records."
        if result.get("kind") == "graph_impact":
            return f"Site impact analysis found {result['count']} linked entities across one-hop and two-hop graph relationships."
        return "This result was produced by deterministic execution over structured records."

    def analyst_reply(self, context: Dict[str, Any], question: str) -> Dict[str, Any]:
        q = norm_text(question)
        if q == "why?":
            return {
                "answer": "Because multiple disruptions cluster around Site 1 via Route 7 and supplier links.",
                "suggested_next": ["show evidence", "what changed", "what if road closed"],
            }
        if q == "show evidence":
            return {
                "answer": "Evidence includes delayed and blocked shipment events plus supplier-to-site and route-to-site graph edges.",
                "suggested_next": ["what changed", "what affects site 1"],
            }
        if q == "what changed":
            return {
                "answer": "Route 7 had both a blocked event and a later normal event, indicating partial recovery.",
                "suggested_next": ["show evidence", "what if road closed"],
            }
        if q == "what if road closed":
            return {
                "answer": "A road closure would likely increase Site 1 disruption because Route 7 serves Site 1 and multiple shipments depend on it.",
                "suggested_next": ["show evidence"],
            }
        return {
            "answer": "I can help interpret the deterministic results and suggest follow-up questions.",
            "suggested_next": ["why?", "show evidence"],
        }


# ============================================================
# 5) Deterministic PAL core execution layer
# ============================================================

class RuleEngine:
    def __init__(self):
        self.rules: List[Dict[str, Any]] = []

    def add_rule(self, rule: Dict[str, Any]):
        self.rules.append(rule)

    def evaluate_event(self, ev: Event) -> List[Dict[str, Any]]:
        alerts = []
        for rule in self.rules:
            if self._match_rule(ev, rule):
                alerts.append({
                    "rule_id": rule["rule_id"],
                    "event_id": ev.event_id,
                    "priority": rule["then"].get("priority", 5),
                    "reason": rule["then"].get("reason", "rule matched"),
                })
        return alerts

    def _match_rule(self, ev: Event, rule: Dict[str, Any]) -> bool:
        conds = rule.get("when", {}).get("all", [])
        for cond in conds:
            field = cond.get("field")
            if field == "note_contains_any":
                phrases = cond.get("value", [])
                note_l = ev.note.lower()
                if not any(p.lower() in note_l for p in phrases):
                    return False
                continue
            if cond.get("exists") is True:
                if getattr(ev, field, None) in (None, ""):
                    return False
                continue
            if "in" in cond:
                if getattr(ev, field, None) not in cond["in"]:
                    return False
                continue
            if "eq" in cond:
                if getattr(ev, field, None) != cond["eq"]:
                    return False
                continue
        return True


class ExecutionEngine:
    def __init__(self, db: OntologyStore):
        self.db = db

    def run_action(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        action = spec.get("action")
        if action == "query_events":
            filt = spec.get("filter", {})
            events = self.db.filter_events(filt)
            return {
                "kind": "query_events",
                "count": len(events),
                "events": [self._event_to_dict(ev) for ev in events],
            }
        if action == "graph_impact":
            site_id = spec["site_id"]
            impacts = self.db.two_hop_impacts_from_site(site_id)
            return {
                "kind": "graph_impact",
                "site_id": site_id,
                "count": len(impacts),
                "impacts": impacts,
            }
        raise ValueError(f"Unknown action: {action}")

    def run_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        env: Dict[str, Any] = {}
        steps = plan.get("steps", [])
        trace = []

        for step in steps:
            op = step["op"]
            name = step["name"]

            if op == "filter_events":
                result = self.db.filter_events(step.get("args", {}))
                env[name] = result
                trace.append({"step": name, "op": op, "count": len(result)})
                continue

            if op == "aggregate_counts":
                src = env[step["input"]]
                result = self.db.aggregate_counts(src, step["field"])
                env[name] = result
                trace.append({"step": name, "op": op, "keys": list(result.keys())})
                continue

            if op == "compare_named_results":
                left = env[step["left"]]
                right = env[step["right"]]
                result = self._compare_dicts(left, right)
                env[name] = result
                trace.append({"step": name, "op": op})
                continue

            raise ValueError(f"Unknown plan op: {op}")

        last_name = steps[-1]["name"] if steps else None
        return {
            "trace": trace,
            "result": env.get(last_name),
        }

    @staticmethod
    def _compare_dicts(left: Dict[str, int], right: Dict[str, int]) -> Dict[str, Any]:
        keys = sorted(set(left) | set(right))
        rows = []
        for k in keys:
            a = left.get(k, 0)
            b = right.get(k, 0)
            rows.append({"key": k, "left": a, "right": b, "delta": a - b})
        return {"rows": rows}

    @staticmethod
    def _event_to_dict(ev: Event) -> Dict[str, Any]:
        return {
            "event_id": ev.event_id,
            "entity_id": ev.entity_id,
            "event_type": ev.event_type,
            "location": ev.location,
            "status": ev.status,
            "note": ev.note,
            "supplier_id": ev.supplier_id,
            "route_id": ev.route_id,
            "site_id": ev.site_id,
            "severity": ev.severity,
        }


# ============================================================
# 6) PAL meta-demo wrapper with 8 LLM roles
# ============================================================

class PALMetaDemo:
    ROLE_DEFINITIONS = {
        "command": (1, "Human Command Interpreter", "role_command_interpreter"),
        "ingest": (2, "Messy Data Ingest / Extraction", "role_ingest_extraction"),
        "planner": (3, "Planner / Task Decomposer", "role_planner"),
        "rulegen": (4, "Rule Generator / Rule Injection", "role_rule_generator"),
        "ontology": (5, "Semantic Mapping / Ontology Translation", "role_ontology_mapping"),
        "retrieval": (6, "Search / Retrieval Helper", "role_retrieval_helper"),
        "explain": (7, "Explanation / Summarization", "role_explanation"),
        "analyst": (8, "Interactive Analyst Assistant", "role_analyst_assistant"),
    }

    def __init__(self, llm_mode: str = "mock"):
        self.db = seed_demo_store()
        self.engine = ExecutionEngine(self.db)
        self.rules = RuleEngine()
        self.llm = MockLLM() if llm_mode == "mock" else MockLLM()
        self.llm_mode = llm_mode

    # --------------------------------------------------------
    # 6.1 Shared role output helpers
    # --------------------------------------------------------
    def _role_payload(self, role_key: str, **payload: Any) -> Dict[str, Any]:
        role_number, role_name, _ = self.ROLE_DEFINITIONS[role_key]
        return {
            "role": role_number,
            "name": role_name,
            **payload,
        }

    def _run_query_action(self, query_filter: Dict[str, Any]) -> Dict[str, Any]:
        return self.engine.run_action({
            "action": "query_events",
            "filter": query_filter,
        })

    def _run_graph_impact_action(self, site_id: str) -> Dict[str, Any]:
        return self.engine.run_action({
            "action": "graph_impact",
            "site_id": site_id,
        })

    def _build_event_from_json(self, event_json: Dict[str, Any]) -> Event:
        return Event(
            event_id="E900",
            entity_id=event_json["entity_id"],
            event_type=event_json["event_type"],
            location=event_json["location"],
            status=event_json["status"],
            note=event_json["note"],
            supplier_id=event_json.get("supplier_id"),
            route_id=event_json.get("route_id"),
            site_id=event_json.get("site_id"),
            severity=event_json.get("severity", 1),
        )

    def get_role_runner_map(self) -> Dict[str, Any]:
        return {
            role_key: getattr(self, method_name)
            for role_key, (_, _, method_name) in self.ROLE_DEFINITIONS.items()
        }

    # --------------------------------------------------------
    # 6.2 Role 1: Human Command Interpreter
    # --------------------------------------------------------
    def role_command_interpreter(self) -> Dict[str, Any]:
        user_text = "Show delayed shipments in Taipei"
        action_spec = self.llm.command_to_json(user_text)
        deterministic_result = self.engine.run_action(action_spec)
        return self._role_payload(
            "command",
            input_text=user_text,
            llm_output=action_spec,
            deterministic_result=deterministic_result,
        )

    # --------------------------------------------------------
    # 6.3 Role 2: Messy Data Ingest / Extraction
    # --------------------------------------------------------
    def role_ingest_extraction(self) -> Dict[str, Any]:
        raw_email = (
            "Truck 900 is running late in Taipei after a flat tire on Route 7. "
            "Acme Rubber may be involved. Please log it for Site 1."
        )
        event_json = self.llm.extract_event_json(raw_email)
        new_event = self._build_event_from_json(event_json)
        self.db.add_event(new_event)
        deterministic_result = self._run_query_action({"entity_id": "truck_900"})
        return self._role_payload(
            "ingest",
            raw_text=raw_email,
            llm_output=event_json,
            deterministic_result=deterministic_result,
        )

    # --------------------------------------------------------
    # 6.4 Role 3: Planner / Task Decomposer
    # --------------------------------------------------------
    def role_planner(self) -> Dict[str, Any]:
        user_text = "Compare delayed shipments in Taipei vs blocked shipments in Tainan"
        plan = self.llm.make_plan(user_text)
        deterministic_result = self.engine.run_plan(plan)
        return self._role_payload(
            "planner",
            input_text=user_text,
            llm_output=plan,
            deterministic_result=deterministic_result,
        )

    # --------------------------------------------------------
    # 6.5 Role 4: Rule Generator / Rule Injection
    # --------------------------------------------------------
    def role_rule_generator(self) -> Dict[str, Any]:
        user_text = "Alert if outage + delay + supplier overlap"
        rule = self.llm.generate_rule_json(user_text)
        self.rules.add_rule(rule)

        alerts = []
        for ev in self.db.events:
            alerts.extend(self.rules.evaluate_event(ev))

        return self._role_payload(
            "rulegen",
            input_text=user_text,
            llm_output=rule,
            deterministic_result={
                "rule_count": len(self.rules.rules),
                "alerts": alerts,
            },
        )

    # --------------------------------------------------------
    # 6.6 Role 5: Semantic Mapping / Ontology Translation
    # --------------------------------------------------------
    def role_ontology_mapping(self) -> Dict[str, Any]:
        phrases = ["flat tire", "blowout", "road closure"]
        mapped = {p: self.llm.semantic_map(p)["canonical"] for p in phrases}
        return self._role_payload(
            "ontology",
            input_phrases=phrases,
            llm_output=mapped,
            deterministic_result={
                "canonical_terms": sorted(set(mapped.values()))
            },
        )

    # --------------------------------------------------------
    # 6.7 Role 6: Search / Retrieval Helper
    # --------------------------------------------------------
    def role_retrieval_helper(self) -> Dict[str, Any]:
        user_text = "What affects Site 1?"
        retrieval_spec = self.llm.retrieval_query(user_text)
        if retrieval_spec["type"] == "graph_impact":
            deterministic_result = self._run_graph_impact_action(retrieval_spec["site_id"])
        else:
            deterministic_result = {"kind": "unknown"}
        return self._role_payload(
            "retrieval",
            input_text=user_text,
            llm_output=retrieval_spec,
            deterministic_result=deterministic_result,
        )

    # --------------------------------------------------------
    # 6.8 Role 7: Explanation / Summarization
    # --------------------------------------------------------
    def role_explanation(self) -> Dict[str, Any]:
        machine_result = self._run_query_action({"status": "delayed", "location": "taipei"})
        llm_output = self.llm.explain_result(machine_result)
        return self._role_payload("explain", machine_result=machine_result, llm_output=llm_output)

    # --------------------------------------------------------
    # 6.9 Role 8: Interactive Analyst Assistant
    # --------------------------------------------------------
    def role_analyst_assistant(self) -> Dict[str, Any]:
        context = self._run_graph_impact_action("site_1")
        dialogue = []
        for q in ["Why?", "Show evidence", "What changed?", "What if road closed?"]:
            reply = self.llm.analyst_reply(context, q)
            dialogue.append({"user": q, "assistant": reply})
        return self._role_payload("analyst", context=context, dialogue=dialogue)

    # --------------------------------------------------------
    # 6.10 Run all role demos
    # --------------------------------------------------------
    def run_all(self) -> Dict[str, Any]:
        role_runner_map = self.get_role_runner_map()
        return {
            "meta_demo": "pal_core_06_meta_llm_roles",
            "llm_mode": self.llm_mode,
            "roles": [role_runner_map[role_key]() for role_key in self.ROLE_DEFINITIONS],
        }


# ============================================================
# 7) CLI entrypoint
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="PAL core 06 meta LLM roles demo")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_demo = sub.add_parser("demo", help="run all 8 roles")
    p_demo.add_argument("--llm", default="mock", choices=["mock", "off"], help="LLM mode")

    p_role = sub.add_parser("role", help="run one role")
    p_role.add_argument(
        "name",
        choices=["command", "ingest", "planner", "rulegen", "ontology", "retrieval", "explain", "analyst"],
    )
    p_role.add_argument("--llm", default="mock", choices=["mock", "off"], help="LLM mode")

    args = ap.parse_args()
    app = PALMetaDemo(llm_mode=args.llm)

    if args.cmd == "demo":
        out = app.run_all()
        jprint(out)
        return

    role_map = {
        "command": app.role_command_interpreter,
        "ingest": app.role_ingest_extraction,
        "planner": app.role_planner,
        "rulegen": app.role_rule_generator,
        "ontology": app.role_ontology_mapping,
        "retrieval": app.role_retrieval_helper,
        "explain": app.role_explanation,
        "analyst": app.role_analyst_assistant,
    }
    out = role_map[args.name]()
    jprint(out)


if __name__ == "__main__":
    main()
