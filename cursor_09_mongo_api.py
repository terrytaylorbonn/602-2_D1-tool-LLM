# cursor_09_mongo_api.py
# ### 09 mongo: This file starts from cursor_08_fastapi.py and migrates read APIs to MongoDB.
#
# 3.9 -- FastAPI read API backed by MongoDB
#
# Purpose:
#   Read selected Gmail messages via IMAP and ingest local n8n webhooks,
#   then convert both into normalized PAL events.
#
# Environment variables:
#   PAL_EMAIL_ADDRESS=your_email@gmail.com
#   PAL_EMAIL_APP_PASSWORD=your_16_char_app_password
#   PAL_IMAP_SERVER=imap.gmail.com
#   PAL_WEBHOOK_TOKEN=your_local_shared_token
#
# Notes:
#   - For Gmail, use an App Password, not your normal password.
#   - This demo is deterministic on purpose.
#   - No LLM used here yet. We only ingest and normalize.
#
# Commands:
#   python cursor_08_fastapi.py reset
#   python cursor_08_fastapi.py fetch
#   python cursor_08_fastapi.py show
#   python cursor_08_fastapi.py loop --interval 60
#   python cursor_08_fastapi.py serve --port 8080
#   uvicorn cursor_09_mongo_api:app --reload --port 8091
#
# Example:
#   Add these values to .env:
#   PAL_EMAIL_ADDRESS=you@gmail.com
#   PAL_EMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx
#   PAL_IMAP_SERVER=imap.gmail.com
#   python cursor_07_n8n_ingest.py fetch

import argparse
import email
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import imaplib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from email.message import Message
from email.header import decode_header
# ### 08 fastapi: Import FastAPI to expose REST endpoints over existing event data.
from fastapi import FastAPI, HTTPException, Query
from typing import Any, Dict, List, Optional, Tuple
try:
    # ### 09 mongo: Keep import guarded so the module can still load even if pymongo is missing.
    from pymongo import MongoClient
except Exception:
    MongoClient = None


# --------------------------------------------------
# 1. Config
# --------------------------------------------------

# CODEX CHANGE: Load PAL Gmail settings from a local .env file before reading os.getenv.
def load_dotenv_file(path: str = ".env") -> None:
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value


# CODEX CHANGE: Make .env values available to the config below.
load_dotenv_file()

DB_FILE = "pal_events07.json"
STATE_FILE = "pal_email_state07.json"

# CODEX CHANGE: PAL_IMAP_SERVER now comes from .env, with Gmail as the fallback.
DEFAULT_IMAP_SERVER = os.getenv("PAL_IMAP_SERVER", "imap.gmail.com")
DEFAULT_LABEL = "INBOX"
DEFAULT_MAX_RESULTS = 10
DEFAULT_WEBHOOK_HOST = "127.0.0.1"
DEFAULT_WEBHOOK_PORT = 8080
MAX_WEBHOOK_BODY_BYTES = 32_000
# ### 09 mongo: Mongo constants mirror pal_v5_mongo.py defaults.
MONGO_DB_NAME = "pal_db"
MONGO_COLLECTION_NAME = "events"
MONGO_URI = os.getenv("MONGO_URI", "").strip()

# ### 09 mongo: Create API app instance for Mongo-backed read routes.
app = FastAPI(title="Cursor 09 Mongo API", version="0.1.0")

mongo_client = None
mongo_db = None
mongo_events_collection = None

# ### 09 mongo: Initialize Mongo client once at import-time when MONGO_URI is present.
if MongoClient and MONGO_URI:
    try:
        mongo_client = MongoClient(MONGO_URI)
        mongo_db = mongo_client[MONGO_DB_NAME]
        mongo_events_collection = mongo_db[MONGO_COLLECTION_NAME]
    except Exception:
        mongo_client = None
        mongo_db = None
        mongo_events_collection = None

# Optional filters to keep the demo clean.
# You can expand these later.
WATCH_SENDERS = [
    # examples:
    # "alerts@company.com",
    # "noreply@monitoring.com",
]
WATCH_SUBJECT_KEYWORDS = [
    "alert",
    "incident",
    "delay",
    "blocked",
    "failure",
    "warning",
    "shipment",
    "delivery",
]


# --------------------------------------------------
# 2. Small helpers
# --------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def decode_mime_words(value: Optional[str]) -> str:
    if not value:
        return ""
    parts = decode_header(value)
    out = []
    for part, enc in parts:
        if isinstance(part, bytes):
            out.append(part.decode(enc or "utf-8", errors="replace"))
        else:
            out.append(part)
    return "".join(out)


def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def safe_lower(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def ensure_db() -> Dict[str, Any]:
    db = load_json(DB_FILE, {"events": []})
    if "events" not in db or not isinstance(db["events"], list):
        db = {"events": []}
    return db


def ensure_state() -> Dict[str, Any]:
    state = load_json(STATE_FILE, {"seen_email_ids": []})
    if "seen_email_ids" not in state or not isinstance(state["seen_email_ids"], list):
        state = {"seen_email_ids": []}
    return state


def load_events_for_api() -> List[Dict[str, Any]]:
    # ### 09 mongo: Read from Mongo first, then fallback to local JSON so dev flow never hard-breaks.
    if mongo_events_collection is not None:
        try:
            raw_events = list(mongo_events_collection.find({}, {"_id": 0}))
            return [normalize_event_for_api(event) for event in raw_events]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Mongo read failed: {e}")

    # ### 09 mongo: Fallback to file-backed events if Mongo is not configured.
    db = ensure_db()
    events = db.get("events", [])
    if not isinstance(events, list):
        return []
    return [normalize_event_for_api(event) for event in events]


def derive_priority_from_status(status: str) -> int:
    # ### 09 mongo: Backfill priority for legacy records that only stored status.
    status_l = safe_lower(status)
    if status_l in {"blocked", "failed"}:
        return 9
    if status_l == "delayed":
        return 7
    if status_l == "warning":
        return 5
    if status_l in {"resolved", "ok"}:
        return 2
    return 4


def normalize_event_for_api(event: Dict[str, Any]) -> Dict[str, Any]:
    # ### 09 mongo: Present a consistent event schema regardless of source/history.
    if not isinstance(event, dict):
        return {}

    out = dict(event)
    entity = normalize_whitespace(str(out.get("entity", "")))
    note = normalize_whitespace(str(out.get("note", "")))
    status = normalize_whitespace(str(out.get("status", "")))
    location = normalize_whitespace(str(out.get("location", "")))
    event_type = normalize_whitespace(str(out.get("event_type", "")))
    timestamp = normalize_whitespace(str(out.get("timestamp", ""))) or utc_now_iso()

    out["entity"] = entity or "unknown_entity"
    out["note"] = note
    out["status"] = status or "unknown"
    out["location"] = location or "unknown"
    out["event_type"] = event_type or "event"
    out["timestamp"] = timestamp
    out["event_time_raw"] = normalize_whitespace(str(out.get("event_time_raw", ""))) or timestamp
    out["source"] = normalize_whitespace(str(out.get("source", ""))) or "mongo_legacy"
    out["source_message_id"] = normalize_whitespace(str(out.get("source_message_id", ""))) or f"legacy_{entity or 'event'}_{timestamp}"
    out["subject"] = normalize_whitespace(str(out.get("subject", ""))) or f"{out['event_type']} {out['entity']}".strip()
    out["sender"] = normalize_whitespace(str(out.get("sender", ""))) or "unknown"

    if "priority" in out:
        try:
            out["priority"] = int(out.get("priority", 0))
        except Exception:
            out["priority"] = derive_priority_from_status(out["status"])
    else:
        out["priority"] = derive_priority_from_status(out["status"])

    return out


@app.get("/events")
def get_events(
    source: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    entity: Optional[str] = Query(default=None),
    location: Optional[str] = Query(default=None),
    event_type: Optional[str] = Query(default=None),
    min_priority: Optional[int] = Query(default=None, ge=0, le=10),
    max_priority: Optional[int] = Query(default=None, ge=0, le=10),
    q: Optional[str] = Query(default=None),
    sort: str = Query(default="timestamp_desc"),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> Dict[str, Any]:
    # ### 08 fastapi: GET /events now supports filtering, simple text search, sorting, and pagination.
    events = load_events_for_api()

    filtered = events

    def _eq(field: str, value: Optional[str]) -> None:
        nonlocal filtered
        if value:
            val_l = safe_lower(value)
            filtered = [e for e in filtered if safe_lower(str(e.get(field, ""))) == val_l]

    _eq("source", source)
    _eq("status", status)
    _eq("entity", entity)
    _eq("location", location)
    _eq("event_type", event_type)

    if min_priority is not None:
        filtered = [e for e in filtered if int(e.get("priority", 0)) >= min_priority]
    if max_priority is not None:
        filtered = [e for e in filtered if int(e.get("priority", 0)) <= max_priority]

    if q:
        q_l = safe_lower(q)
        filtered = [
            e for e in filtered
            if q_l in safe_lower(str(e.get("subject", "")))
            or q_l in safe_lower(str(e.get("note", "")))
            or q_l in safe_lower(str(e.get("entity", "")))
            or q_l in safe_lower(str(e.get("sender", "")))
            or q_l in safe_lower(str(e.get("location", "")))
        ]

    if sort == "timestamp_desc":
        filtered = sorted(filtered, key=lambda e: str(e.get("timestamp", "")), reverse=True)
    elif sort == "timestamp_asc":
        filtered = sorted(filtered, key=lambda e: str(e.get("timestamp", "")))
    elif sort == "priority_desc":
        filtered = sorted(filtered, key=lambda e: int(e.get("priority", 0)), reverse=True)
    elif sort == "priority_asc":
        filtered = sorted(filtered, key=lambda e: int(e.get("priority", 0)))
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid sort. Use: timestamp_desc, timestamp_asc, priority_desc, priority_asc",
        )

    total = len(filtered)
    page = filtered[offset : offset + limit]
    return {
        "ok": True,
        "count": len(page),
        "total": total,
        "limit": limit,
        "offset": offset,
        "events": page,
    }


@app.get("/events/{source_message_id}")
def get_event_by_id(source_message_id: str) -> Dict[str, Any]:
    # ### 08 fastapi: Add point-lookup endpoint for one event by source_message_id.
    events = load_events_for_api()

    target = source_message_id.strip()
    for event in events:
        if str(event.get("source_message_id", "")).strip() == target:
            return {"ok": True, "event": event}

    raise HTTPException(status_code=404, detail=f"Event not found: {source_message_id}")


@app.get("/events/summary/status")
def get_event_status_summary() -> Dict[str, Any]:
    # ### 08 fastapi: Add summary endpoint to quickly inspect distribution by status.
    events = load_events_for_api()

    summary: Dict[str, int] = {}
    for event in events:
        status_key = str(event.get("status", "unknown")) or "unknown"
        summary[status_key] = summary.get(status_key, 0) + 1

    return {"ok": True, "total": len(events), "by_status": summary}


@app.get("/health")
def health_check() -> Dict[str, Any]:
    # ### 09 mongo: Health now reports Mongo connectivity in addition to local file fallback info.
    db_exists = os.path.exists(DB_FILE)
    state_exists = os.path.exists(STATE_FILE)
    mongo_configured = bool(MONGO_URI)
    mongo_import_ok = MongoClient is not None
    mongo_connected = False
    mongo_error = None

    # ### 09 mongo: Try a lightweight collection operation to verify connectivity.
    if mongo_events_collection is not None:
        try:
            mongo_events_collection.estimated_document_count()
            mongo_connected = True
        except Exception as e:
            mongo_error = str(e)

    return {
        "ok": True,
        "db_file": DB_FILE,
        "db_exists": db_exists,
        "state_exists": state_exists,
        "mongo_configured": mongo_configured,
        "mongo_import_ok": mongo_import_ok,
        "mongo_connected": mongo_connected,
        "mongo_db": MONGO_DB_NAME,
        "mongo_collection": MONGO_COLLECTION_NAME,
        "mongo_error": mongo_error,
    }


def severity_to_priority(severity: str) -> int:
    sev = safe_lower(severity)
    if sev == "critical":
        return 9
    if sev == "high":
        return 7
    if sev == "medium":
        return 5
    if sev == "low":
        return 3
    return 4


def webhook_payload_to_event(payload: Dict[str, Any]) -> Dict[str, Any]:
    title = normalize_whitespace(str(payload.get("title", "")))
    text = normalize_whitespace(str(payload.get("text", "")))
    body = f"{title} {text}".strip()
    source_msg_id = normalize_whitespace(str(payload.get("id", ""))) or f"n8n_{int(time.time()*1000)}"
    severity = normalize_whitespace(str(payload.get("severity", "")))
    entity = normalize_whitespace(str(payload.get("entity", ""))) or extract_entity_id(title, text)
    location = normalize_whitespace(str(payload.get("location", ""))) or extract_location(title, text)

    status = infer_status(title, text)
    priority = severity_to_priority(severity) if severity else infer_priority(status, title, text)
    event_type = infer_event_type(status, title, text)

    return {
        "source": "n8n_webhook",
        "source_message_id": source_msg_id,
        "timestamp": utc_now_iso(),
        "event_time_raw": utc_now_iso(),
        "entity": entity or "webhook_signal",
        "event_type": event_type,
        "status": status,
        "priority": priority,
        "location": location or "unknown",
        "subject": title or "n8n_alert",
        "sender": "n8n_local",
        "note": body[:500],
    }


def add_event_if_new(event: Dict[str, Any]) -> Tuple[bool, str]:
    db = ensure_db()
    state = ensure_state()
    seen = set(state["seen_email_ids"])
    msg_id = event.get("source_message_id", "").strip()

    if not msg_id:
        return False, "missing_source_message_id"
    if msg_id in seen:
        return False, "duplicate"

    db["events"].append(event)
    seen.add(msg_id)
    state["seen_email_ids"] = sorted(seen)
    save_json(DB_FILE, db)
    save_json(STATE_FILE, state)
    return True, "added"


# --------------------------------------------------
# 3. Email body extraction
# --------------------------------------------------

# CODEX CHANGE: Use Message directly so the type annotation does not depend on email.message being loaded as a module attribute.
def extract_text_from_message(msg: Message) -> str:
    """
    Return a plain text body if possible.
    Keep this simple and deterministic.
    """
    if msg.is_multipart():
        chunks = []
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition", "")).lower()

            if "attachment" in content_disposition:
                continue

            if content_type == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    chunks.append(payload.decode(charset, errors="replace"))
        return normalize_whitespace(" ".join(chunks))

    payload = msg.get_payload(decode=True)
    if payload:
        charset = msg.get_content_charset() or "utf-8"
        return normalize_whitespace(payload.decode(charset, errors="replace"))
    return ""


# --------------------------------------------------
# 4. Deterministic parsing
# --------------------------------------------------

def infer_status(subject: str, body: str) -> str:
    text = f"{subject} {body}".lower()

    if any(x in text for x in ["blocked", "stuck", "halted"]):
        return "blocked"
    if any(x in text for x in ["delayed", "delay", "late"]):
        return "delayed"
    if any(x in text for x in ["failed", "failure", "error", "exception"]):
        return "failed"
    if any(x in text for x in ["warning", "degraded", "risk"]):
        return "warning"
    if any(x in text for x in ["delivered", "resolved", "recovered", "restored"]):
        return "resolved"
    return "alert"


def infer_priority(status: str, subject: str, body: str) -> int:
    text = f"{subject} {body}".lower()

    if "critical" in text or "sev1" in text or status == "blocked":
        return 9
    if "high" in text or "sev2" in text or status in ("failed", "delayed"):
        return 7
    if status == "warning":
        return 5
    if status == "resolved":
        return 2
    return 4


def extract_entity_id(subject: str, body: str) -> str:
    text = f"{subject} {body}"

    patterns = [
        r"\b(truck[_\- ]?\d+)\b",
        r"\b(shipment[_\- ]?\d+)\b",
        r"\b(order[_\- ]?\d+)\b",
        r"\b(site[_\- ]?\d+)\b",
        r"\b(sensor[_\- ]?\d+)\b",
        r"\b(device[_\- ]?\d+)\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).replace(" ", "_").lower()

    return "email_signal"


def extract_location(subject: str, body: str) -> str:
    text = f"{subject} {body}".lower()

    known_locations = [
        "taipei",
        "tainan",
        "kaohsiung",
        "tokyo",
        "berlin",
        "kyiv",
        "warsaw",
        "new york",
        "hoboken",
        "site_1",
        "site_2",
        "site_3",
    ]
    for loc in known_locations:
        if loc in text:
            return loc.replace(" ", "_")
    return "unknown"


def infer_event_type(status: str, subject: str, body: str) -> str:
    text = f"{subject} {body}".lower()

    if any(x in text for x in ["shipment", "delivery", "carrier", "warehouse"]):
        return "logistics_alert"
    if any(x in text for x in ["server", "service", "api", "database", "latency"]):
        return "system_alert"
    if any(x in text for x in ["supplier", "vendor"]):
        return "supplier_alert"
    if status == "resolved":
        return "resolution_notice"
    return "email_alert"


def should_keep_email(sender: str, subject: str) -> bool:
    sender_l = safe_lower(sender)
    subject_l = safe_lower(subject)

    sender_match = (not WATCH_SENDERS) or any(x in sender_l for x in WATCH_SENDERS)
    subject_match = any(k in subject_l for k in WATCH_SUBJECT_KEYWORDS)

    return sender_match or subject_match


# CODEX CHANGE: Use Message directly for the same reason as above.
def email_to_pal_event(msg: Message, raw_body: str) -> Dict[str, Any]:
    subject = decode_mime_words(msg.get("Subject", ""))
    sender = decode_mime_words(msg.get("From", ""))
    msg_id = decode_mime_words(msg.get("Message-ID", "")) or f"no_msgid_{int(time.time()*1000)}"
    date_raw = decode_mime_words(msg.get("Date", ""))

    status = infer_status(subject, raw_body)
    priority = infer_priority(status, subject, raw_body)
    entity_id = extract_entity_id(subject, raw_body)
    location = extract_location(subject, raw_body)
    event_type = infer_event_type(status, subject, raw_body)

    return {
        "source": "gmail",
        "source_message_id": msg_id,
        "timestamp": utc_now_iso(),
        "event_time_raw": date_raw,
        "entity": entity_id,
        "event_type": event_type,
        "status": status,
        "priority": priority,
        "location": location,
        "subject": subject,
        "sender": sender,
        "note": raw_body[:500],
    }


# --------------------------------------------------
# 5. IMAP fetch
# --------------------------------------------------

def connect_imap() -> imaplib.IMAP4_SSL:
    email_addr = os.getenv("PAL_EMAIL_ADDRESS", "").strip()
    email_pw = os.getenv("PAL_EMAIL_APP_PASSWORD", "").strip()

    if not email_addr or not email_pw:
        print("ERROR: Missing PAL_EMAIL_ADDRESS or PAL_EMAIL_APP_PASSWORD")
        sys.exit(1)

    mail = imaplib.IMAP4_SSL(DEFAULT_IMAP_SERVER)
    mail.login(email_addr, email_pw)
    return mail


# CODEX CHANGE: Use Message directly for the return type annotation.
def fetch_recent_emails(max_results: int = DEFAULT_MAX_RESULTS, label: str = DEFAULT_LABEL) -> List[Message]:
    mail = connect_imap()
    try:
        status, _ = mail.select(label)
        if status != "OK":
            print(f"ERROR: could not open mailbox {label}")
            return []

        # Pull latest emails. Keep the demo simple.
        status, data = mail.search(None, "ALL")
        if status != "OK" or not data or not data[0]:
            return []

        ids = data[0].split()
        ids = ids[-max_results:]

        msgs = []
        for email_id in ids:
            status, msg_data = mail.fetch(email_id, "(RFC822)")
            if status != "OK":
                continue
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    msgs.append(msg)
        return msgs
    finally:
        try:
            mail.close()
        except Exception:
            pass
        mail.logout()


# --------------------------------------------------
# 6. Commands
# --------------------------------------------------

def cmd_reset() -> None:
    save_json(DB_FILE, {"events": []})
    save_json(STATE_FILE, {"seen_email_ids": []})
    print(json.dumps({"ok": True, "reset": True}, indent=2))


def cmd_show() -> None:
    db = ensure_db()
    print(json.dumps(db, indent=2, ensure_ascii=False))


def cmd_fetch(max_results: int, label: str) -> None:
    db = ensure_db()
    state = ensure_state()
    seen = set(state["seen_email_ids"])

    msgs = fetch_recent_emails(max_results=max_results, label=label)

    added = []
    skipped = 0

    for msg in msgs:
        subject = decode_mime_words(msg.get("Subject", ""))
        sender = decode_mime_words(msg.get("From", ""))
        msg_id = decode_mime_words(msg.get("Message-ID", ""))

        if not msg_id:
            skipped += 1
            continue

        if msg_id in seen:
            skipped += 1
            continue

        if not should_keep_email(sender, subject):
            skipped += 1
            continue

        body = extract_text_from_message(msg)
        event = email_to_pal_event(msg, body)

        db["events"].append(event)
        seen.add(msg_id)
        added.append(event)

    state["seen_email_ids"] = sorted(seen)
    save_json(DB_FILE, db)
    save_json(STATE_FILE, state)

    print(json.dumps({
        "ok": True,
        "fetched_messages": len(msgs),
        "added_events": len(added),
        "skipped": skipped,
        "events": added,
    }, indent=2, ensure_ascii=False))


def cmd_loop(interval_sec: int, max_results: int, label: str) -> None:
    print(f"[pal_core_07] polling gmail every {interval_sec}s...")
    while True:
        try:
            cmd_fetch(max_results=max_results, label=label)
        except KeyboardInterrupt:
            print("\n[pal_core_07] stopped")
            break
        except Exception as e:
            print(json.dumps({"ok": False, "error": str(e)}, indent=2))
        time.sleep(interval_sec)


def cmd_serve(port: int, host: str) -> None:
    token = os.getenv("PAL_WEBHOOK_TOKEN", "").strip()
    if not token:
        print("ERROR: Missing PAL_WEBHOOK_TOKEN")
        sys.exit(1)

    class N8nIngestHandler(BaseHTTPRequestHandler):
        server_version = "Cursor07N8N/0.1"

        def _send_json(self, status_code: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[webhook] {self.address_string()} - {fmt % args}")

        def do_POST(self) -> None:
            if self.path != "/ingest":
                self._send_json(404, {"ok": False, "error": "not_found"})
                return

            auth_header = self.headers.get("Authorization", "")
            expected = f"Bearer {token}"
            if not auth_header or not hashlib.sha256(auth_header.encode()).digest() == hashlib.sha256(expected.encode()).digest():
                self._send_json(401, {"ok": False, "error": "unauthorized"})
                return

            raw_len = self.headers.get("Content-Length", "0").strip()
            if not raw_len.isdigit():
                self._send_json(400, {"ok": False, "error": "invalid_content_length"})
                return
            length = int(raw_len)
            if length <= 0 or length > MAX_WEBHOOK_BODY_BYTES:
                self._send_json(413, {"ok": False, "error": "payload_too_large"})
                return

            raw_body = self.rfile.read(length)
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except Exception:
                self._send_json(400, {"ok": False, "error": "invalid_json"})
                return

            if not isinstance(payload, dict):
                self._send_json(400, {"ok": False, "error": "payload_must_be_object"})
                return

            if not payload.get("title") and not payload.get("text"):
                self._send_json(400, {"ok": False, "error": "title_or_text_required"})
                return

            event = webhook_payload_to_event(payload)
            added, reason = add_event_if_new(event)
            self._send_json(200, {"ok": True, "added_events": 1 if added else 0, "reason": reason, "event": event})

    server = ThreadingHTTPServer((host, port), N8nIngestHandler)
    print(f"[cursor_07] n8n webhook server listening on http://{host}:{port}/ingest")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[cursor_07] webhook server stopped")
    finally:
        server.server_close()


# --------------------------------------------------
# 7. Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cursor 07 - Gmail + n8n webhook ingest")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("reset")
    sub.add_parser("show")

    p_fetch = sub.add_parser("fetch")
    p_fetch.add_argument("--max_results", type=int, default=DEFAULT_MAX_RESULTS)
    p_fetch.add_argument("--label", type=str, default=DEFAULT_LABEL)

    p_loop = sub.add_parser("loop")
    p_loop.add_argument("--interval", type=int, default=60)
    p_loop.add_argument("--max_results", type=int, default=DEFAULT_MAX_RESULTS)
    p_loop.add_argument("--label", type=str, default=DEFAULT_LABEL)

    p_serve = sub.add_parser("serve")
    p_serve.add_argument("--port", type=int, default=DEFAULT_WEBHOOK_PORT)
    p_serve.add_argument("--host", type=str, default=DEFAULT_WEBHOOK_HOST)

    args = parser.parse_args()

    if args.cmd == "reset":
        cmd_reset()
    elif args.cmd == "show":
        cmd_show()
    elif args.cmd == "fetch":
        cmd_fetch(max_results=args.max_results, label=args.label)
    elif args.cmd == "loop":
        cmd_loop(interval_sec=args.interval, max_results=args.max_results, label=args.label)
    elif args.cmd == "serve":
        cmd_serve(port=args.port, host=args.host)


if __name__ == "__main__":
    main()
