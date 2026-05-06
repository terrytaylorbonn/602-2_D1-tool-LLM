# jobradar_unified.py — Greenhouse + Gmail LinkedIn alerts + FastAPI + OAuth
#
# Steps 1–2: Greenhouse public API → MongoDB + LLM scoring
# Step 3:    FastAPI job API
# Step 4:    Render (uvicorn jobradar_unified:app)
# Step 5:    Google OAuth for /jobs etc.
# Step 6:    Gmail IMAP → LinkedIn job-alert emails → same MongoDB collection
# Step 7:    Digest email (preview + send)
# Step 7e:   Added Resend transport option. Resend requires a verified domain for "from"
#            (not @gmail.com / @googlemail.com — add your own domain in Resend). Marked "## S7e".
# Step 7f:   Daily digest as a new Google Doc in Drive (OAuth + Docs/Drive scopes). Marked "## S7f".
#            Enable Google Drive API + Google Docs API on the GCP project for GOOGLE_CLIENT_*; POST /digest/google-doc.
#
# CLI (ingest only):
#   python jobradar_unified.py
#   Runs Greenhouse ingest, then Gmail ingest if GMAIL_ADDRESS + GMAIL_APP_PASSWORD are set.
#
# API server:
#   uvicorn jobradar_unified:app --host 0.0.0.0 --port 8000
#   Docs: http://127.0.0.1:8000/docs

import imaplib
import email
import re
import json
import os
import smtplib
import requests
from datetime import datetime, timezone, timedelta
from email.header import decode_header
from email.utils import parsedate_to_datetime
from email.message import EmailMessage

from bson import ObjectId
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from pymongo import MongoClient, errors
from authlib.integrations.starlette_client import OAuth
from starlette.middleware.sessions import SessionMiddleware

# --- CONFIG ---
load_dotenv()

# ## S7 helper: parse int envs safely (blank/invalid -> default)
def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name, str(default)) or "").strip()
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"WARNING: {name}='{raw}' is not an int. Using default {default}.")
        return default


MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "jobradar")
COLLECTION = os.getenv("COLLECTION", "jobs")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
JOB_LIMIT = max(0, _env_int("JOB_LIMIT", 10))

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")

# Gmail IMAP (optional — only needed for CLI Gmail ingest)
GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = (os.getenv("GMAIL_APP_PASSWORD") or "").replace(" ", "")
GMAIL_LABEL = os.getenv("GMAIL_LABEL", "job-alerts")
IMAP_SERVER = "imap.gmail.com"
DELETE_AFTER_DAYS = max(0, _env_int("DELETE_AFTER_DAYS", 90))
GMAIL_MAX_EMAILS = max(0, _env_int("GMAIL_MAX_EMAILS", 2))

# ## S7 config: digest email settings
DIGEST_EMAIL_TO = os.getenv("DIGEST_EMAIL_TO")
DIGEST_EMAIL_FROM = os.getenv("DIGEST_EMAIL_FROM") or GMAIL_ADDRESS
DIGEST_EMAIL_APP_PASSWORD = (
    (os.getenv("DIGEST_EMAIL_APP_PASSWORD") or GMAIL_APP_PASSWORD or "").replace(" ", "")
)
# ## S7e Resend config: choose sender transport and HTTP API settings
DIGEST_SENDER = (os.getenv("DIGEST_SENDER", "smtp") or "smtp").strip().lower()
RESEND_API_KEY = os.getenv("RESEND_API_KEY")
RESEND_API_URL = os.getenv("RESEND_API_URL", "https://api.resend.com/emails")
DIGEST_SMTP_HOST = os.getenv("DIGEST_SMTP_HOST", "smtp.gmail.com")
DIGEST_SMTP_PORT = max(1, _env_int("DIGEST_SMTP_PORT", 587))
DIGEST_MAX_JOBS = max(1, _env_int("DIGEST_MAX_JOBS", 15))
DIGEST_MIN_SCORE = max(0, _env_int("DIGEST_MIN_SCORE", 0))

COMPANIES = [
    "anthropic",
    "mongodb",
]

CV_TEXT = """
Experienced Python developer. Skills: FastAPI, MongoDB, LLM integration,
REST APIs, cloud deployment (Render), Docker basics, OpenAI/Anthropic APIs,
agentic AI pipelines, RAG, MCP, tool calling.
Background: drones, robotics AI, full-stack development.
"""

OPENAI_COST_PER_1K_INPUT = 0.00015
OPENAI_COST_PER_1K_OUTPUT = 0.0006
ANTHROPIC_COST_PER_1K_INPUT = 0.00025
ANTHROPIC_COST_PER_1K_OUTPUT = 0.00125
cost_tracker = {"total": 0.0, "calls": 0}

if not MONGO_URI:
    raise ValueError("MONGO_URI not found in .env")
if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
    print("WARNING: GOOGLE_CLIENT_ID or GOOGLE_CLIENT_SECRET not set. Auth will fail.")
if not OPENAI_API_KEY and not ANTHROPIC_API_KEY:
    print("WARNING: No LLM API key found. Jobs will be inserted without scoring.")

# --- MONGODB ---
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION]
collection.create_index("job_id", unique=True)

# --- FASTAPI + OAUTH ---
app = FastAPI(title="JobRadar API", version="0.6-unified")
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

oauth = OAuth()
oauth.register(
    name="google",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)


# --- GREENHOUSE ---
def normalize_job(job: dict, company: str) -> dict:
    location_raw = job.get("location", {})
    if isinstance(location_raw, dict):
        location = location_raw.get("name", "Unknown")
    else:
        location = str(location_raw)

    job_id = f"gh_{company}_{job.get('id', '')}"

    return {
        "job_id": job_id,
        "source": "greenhouse",
        "company": company,
        "role": job.get("title", "Unknown"),
        "location": location,
        "remote": "remote" in location.lower(),
        "url": job.get("absolute_url", ""),
        "posted_date": job.get("updated_at", ""),
        "seen_date": datetime.now(timezone.utc).isoformat(),
        "llm_score": None,
        "llm_notes": None,
        "llm_match": None,
        "llm_gaps": None,
        "llm_provider": None,
        "status": "new",
    }


def build_prompt(role: str, company: str, location: str) -> str:
    return f"""You are a job match evaluator.

CV summary:
{CV_TEXT}

Job posting:
Company: {company}
Role: {role}
Location: {location}

Return JSON only, no preamble, no markdown:
{{
  "score": <0-100>,
  "match": ["reason1", "reason2"],
  "gaps": ["gap1", "gap2"],
  "one_line": "plain English summary of fit"
}}"""


def score_with_llm(role: str, company: str, location: str) -> tuple[dict, str] | None:
    prompt = build_prompt(role, company, location)

    if OPENAI_API_KEY:
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 300,
                    "temperature": 0,
                },
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()
            text = data["choices"][0]["message"]["content"].strip()
            parsed = json.loads(text)
            usage = data.get("usage", {})
            cost = (
                usage.get("prompt_tokens", 0) / 1000 * OPENAI_COST_PER_1K_INPUT
                + usage.get("completion_tokens", 0) / 1000 * OPENAI_COST_PER_1K_OUTPUT
            )
            cost_tracker["total"] += cost
            cost_tracker["calls"] += 1
            return parsed, "openai"
        except Exception as e:
            print(f"    [openai] scoring failed: {e} — trying Anthropic")

    if ANTHROPIC_API_KEY:
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 300,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()
            text = data["content"][0]["text"].strip()
            parsed = json.loads(text)
            usage = data.get("usage", {})
            cost = (
                usage.get("input_tokens", 0) / 1000 * ANTHROPIC_COST_PER_1K_INPUT
                + usage.get("output_tokens", 0) / 1000 * ANTHROPIC_COST_PER_1K_OUTPUT
            )
            cost_tracker["total"] += cost
            cost_tracker["calls"] += 1
            return parsed, "anthropic"
        except Exception as e:
            print(f"    [anthropic] scoring failed: {e}")

    return None


def ingest_company(company: str) -> tuple[int, int]:
    url = f"https://boards-api.greenhouse.io/v1/boards/{company}/jobs"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"  [{company}] HTTP error: {e}")
        return 0, 0
    except requests.exceptions.RequestException as e:
        print(f"  [{company}] Request failed: {e}")
        return 0, 0

    data = response.json()
    jobs = data.get("jobs", [])

    if not jobs:
        print(f"  [{company}] No jobs found (check board token)")
        return 0, 0

    if JOB_LIMIT > 0:
        jobs = jobs[:JOB_LIMIT]

    inserted = 0
    skipped = 0

    for job in jobs:
        normalized = normalize_job(job, company)

        if OPENAI_API_KEY or ANTHROPIC_API_KEY:
            result = score_with_llm(
                normalized["role"],
                normalized["company"],
                normalized["location"],
            )
            if result:
                scored, provider = result
                normalized["llm_score"] = scored.get("score")
                normalized["llm_notes"] = scored.get("one_line")
                normalized["llm_match"] = scored.get("match")
                normalized["llm_gaps"] = scored.get("gaps")
                normalized["llm_provider"] = provider

        try:
            collection.insert_one(normalized)
            inserted += 1
            score_str = f"score={normalized['llm_score']}" if normalized["llm_score"] else "no score"
            print(f"    + {normalized['role'][:50]} [{score_str}]")
        except errors.DuplicateKeyError:
            skipped += 1

    return inserted, skipped


# --- GMAIL / LINKEDIN EMAIL (Step 6) ---

def decode_str(value: str | bytes, charset: str | None) -> str:
    if isinstance(value, bytes):
        return value.decode(charset or "utf-8", errors="replace")
    return value


def get_email_body(msg: email.message.Message) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            cd = str(part.get("Content-Disposition", ""))
            if ct == "text/plain" and "attachment" not in cd:
                charset = part.get_content_charset() or "utf-8"
                return part.get_payload(decode=True).decode(charset, errors="replace")
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                charset = part.get_content_charset() or "utf-8"
                raw_html = part.get_payload(decode=True).decode(charset, errors="replace")
                return re.sub(r"<[^>]+>", " ", raw_html)
    else:
        charset = msg.get_content_charset() or "utf-8"
        return msg.get_payload(decode=True).decode(charset, errors="replace")
    return ""


def parse_linkedin_email(body: str, subject: str) -> dict | None:
    url_match = re.search(
        r"https?://(?:www\.)?linkedin\.com/(?:comm/)?jobs/view/(\d+)/?",
        body,
    )
    if not url_match:
        return None

    url = url_match.group(0).split("?")[0]
    job_id_raw = url_match.group(1)

    lines = [l.strip() for l in body.splitlines()]
    lines = [l for l in lines if l]

    role = "Unknown"
    company = "Unknown"
    location = "Unknown"

    for i, line in enumerate(lines):
        if " · " in line or " • " in line:
            sep = " · " if " · " in line else " • "
            parts = line.split(sep, 1)
            if len(parts) == 2:
                company = parts[0].strip()
                location = parts[1].strip()
                if i > 0:
                    role = lines[i - 1].strip()
                break

    if role == "Unknown" and subject:
        subj_match = re.match(r"^(.+?)\s+job(?:\s+at\s+.+)?$", subject, re.IGNORECASE)
        if subj_match:
            role = subj_match.group(1).strip()

    remote = "remote" in location.lower()

    return {
        "role": role,
        "company": company,
        "location": location,
        "remote": remote,
        "url": url,
        "job_id_raw": job_id_raw,
    }


def build_gmail_linkedin_record(parsed: dict, message_id: str, received_date: str) -> dict:
    job_id = f"gmail_li_{parsed['job_id_raw']}"
    return {
        "job_id": job_id,
        "source": "gmail_linkedin",
        "company": parsed["company"],
        "role": parsed["role"],
        "location": parsed["location"],
        "remote": parsed["remote"],
        "url": parsed["url"],
        "posted_date": received_date,
        "seen_date": datetime.now(timezone.utc).isoformat(),
        "gmail_msg_id": message_id,
        "llm_score": None,
        "llm_notes": None,
        "llm_match": None,
        "llm_gaps": None,
        "llm_provider": None,
        "status": "new",
    }


def cleanup_old_jobs(days: int) -> int:
    if days <= 0:
        return 0
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    result = collection.delete_many({"seen_date": {"$lt": cutoff}})
    return result.deleted_count


# ## S7 helper: render digest body lines from job docs
def build_digest_lines(jobs: list[dict]) -> str:
    lines = []
    for idx, job in enumerate(jobs, 1):
        score = job.get("llm_score")
        score_str = "n/a" if score is None else str(score)
        lines.extend(
            [
                f"{idx}. {job.get('role', 'Unknown role')}",
                f"   Company: {job.get('company', 'Unknown')}",
                f"   Location: {job.get('location', 'Unknown')} | Score: {score_str} | Source: {job.get('source', 'n/a')}",
                f"   URL: {job.get('url', '')}",
                "",
            ]
        )
    return "\n".join(lines).strip()


def _digest_email_from_is_plausible(addr: str) -> bool:
    """Catch truncated .env values like name@ with no domain (dotenv / # comment issues)."""
    a = (addr or "").strip()
    if "@" not in a:
        return False
    _, _, domain = a.partition("@")
    if not domain or "." not in domain:
        return False
    return True


# ## S7 helper: digest email sender (SMTP by default)
def send_digest_email(to_addr: str, subject: str, body: str) -> None:
    # ## S7e Resend switch: allow SMTP (local) or Resend HTTPS (Render-friendly)
    sender = DIGEST_SENDER
    if sender not in {"smtp", "resend"}:
        raise RuntimeError("DIGEST_SENDER must be 'smtp' or 'resend'.")

    if not DIGEST_EMAIL_FROM:
        raise RuntimeError("Missing digest sender config. Set DIGEST_EMAIL_FROM.")
    from_addr = DIGEST_EMAIL_FROM.strip()
    if not _digest_email_from_is_plausible(from_addr):
        raise RuntimeError(
            "DIGEST_EMAIL_FROM must be a full address (e.g. name@example.com). "
            "Yours looks truncated — check .env for a broken line or inline # comment on the same line."
        )

    # ## S7e Resend send path
    if sender == "resend":
        if not RESEND_API_KEY:
            raise RuntimeError(
                "Missing Resend config. Set RESEND_API_KEY (or use DIGEST_SENDER=smtp)."
            )
        # Resend rejects some clients without User-Agent with 403 (e.g. code 1010).
        response = requests.post(
            RESEND_API_URL,
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type": "application/json",
                "User-Agent": "JobRadar/1.0 (digest; requests)",
            },
            json={
                "from": from_addr,
                "to": [to_addr],
                "subject": subject,
                "text": body,
            },
            timeout=20,
        )
        response.raise_for_status()
        return

    if not DIGEST_EMAIL_APP_PASSWORD:
        raise RuntimeError(
            "Missing SMTP digest sender config. Set DIGEST_EMAIL_APP_PASSWORD "
            "(or use DIGEST_SENDER=resend)."
        )

    msg = EmailMessage()
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP(DIGEST_SMTP_HOST, DIGEST_SMTP_PORT, timeout=20) as server:
        server.starttls()
        server.login(from_addr, DIGEST_EMAIL_APP_PASSWORD)
        server.send_message(msg)


def ingest_gmail() -> tuple[int, int, int]:
    """Requires GMAIL_ADDRESS and GMAIL_APP_PASSWORD in env."""
    if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
        print("  Gmail ingest skipped: set GMAIL_ADDRESS and GMAIL_APP_PASSWORD in .env")
        return 0, 0, 0

    inserted = 0
    skipped_dup = 0
    skipped_no_job = 0

    print(f"  Connecting to {IMAP_SERVER} as {GMAIL_ADDRESS} …")
    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
    print("  Logged in.")

    status, folders = mail.list()
    folder_names = [f.decode() if isinstance(f, bytes) else f for f in (folders or [])]

    status, data = mail.select(f'"{GMAIL_LABEL}"')
    if status != "OK":
        print(f"\n  ERROR: Label '{GMAIL_LABEL}' not found.")
        print("  Available folders:")
        for f in folder_names:
            print(f"    {f}")
        mail.logout()
        return 0, 0, 0

    print(f"  Folder '{GMAIL_LABEL}' selected — {data[0].decode()} messages total.")

    status, msg_nums = mail.search(None, "UNSEEN")
    if status != "OK" or not msg_nums[0]:
        print("  No new (UNSEEN) messages.")
        mail.logout()
        return 0, 0, 0

    ids = msg_nums[0].split()
    total_unseen = len(ids)
    if GMAIL_MAX_EMAILS > 0:
        ids = ids[:GMAIL_MAX_EMAILS]
    print(f"  Found {total_unseen} UNSEEN message(s) — processing {len(ids)}.")

    for num in ids:
        status, msg_data = mail.fetch(num, "(RFC822)")
        if status != "OK":
            continue

        raw = msg_data[0][1]
        msg = email.message_from_bytes(raw)
        message_id = msg.get("Message-ID", "").strip()

        subject_parts = decode_header(msg.get("Subject", ""))
        subject = "".join(decode_str(part, enc) for part, enc in subject_parts)

        date_str = msg.get("Date", "")
        try:
            received_date = parsedate_to_datetime(date_str).isoformat()
        except Exception:
            received_date = datetime.now(timezone.utc).isoformat()

        body = get_email_body(msg)
        parsed = parse_linkedin_email(body, subject)

        if not parsed:
            print(f"    - [{subject[:50]}] — not a job listing, skipping")
            skipped_no_job += 1
            mail.store(num, "+FLAGS", "\\Seen")
            continue

        record = build_gmail_linkedin_record(parsed, message_id, received_date)

        if OPENAI_API_KEY or ANTHROPIC_API_KEY:
            result = score_with_llm(record["role"], record["company"], record["location"])
            if result:
                scored, provider = result
                record["llm_score"] = scored.get("score")
                record["llm_notes"] = scored.get("one_line")
                record["llm_match"] = scored.get("match")
                record["llm_gaps"] = scored.get("gaps")
                record["llm_provider"] = provider

        try:
            collection.insert_one(record)
            inserted += 1
            score_str = f"score={record['llm_score']}" if record["llm_score"] else "no score"
            print(f"    + {record['role'][:45]} @ {record['company'][:25]} [{score_str}]")
        except errors.DuplicateKeyError:
            skipped_dup += 1
            print(f"    = duplicate: {record['job_id']}")

        mail.store(num, "+FLAGS", "\\Seen")

    mail.logout()
    return inserted, skipped_dup, skipped_no_job


# --- CLI MAIN ---
def main():
    print(f"\nJobRadar — unified ingest (Greenhouse + optional Gmail)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if OPENAI_API_KEY:
        print("LLM: OpenAI (primary)")
    elif ANTHROPIC_API_KEY:
        print("LLM: Anthropic (fallback)")
    else:
        print("LLM: none — scoring disabled")

    print("-" * 45)

    if DELETE_AFTER_DAYS > 0:
        deleted = cleanup_old_jobs(DELETE_AFTER_DAYS)
        if deleted:
            print(f"  Cleaned up {deleted} job(s) older than {DELETE_AFTER_DAYS} days.")

    print(f"\n  [Greenhouse — {len(COMPANIES)} boards]")
    total_inserted = 0
    total_skipped = 0
    for company in COMPANIES:
        print(f"\n  [{company}]")
        inserted, skipped = ingest_company(company)
        print(f"  [{company}] inserted: {inserted}, skipped: {skipped}")
        total_inserted += inserted
        total_skipped += skipped

    print("-" * 45)
    print(f"Greenhouse inserted: {total_inserted}, duplicates: {total_skipped}")

    print(f"\n  [Gmail → label '{GMAIL_LABEL}']")
    if GMAIL_MAX_EMAILS:
        print(f"  Max emails this run: {GMAIL_MAX_EMAILS}")
    else:
        print("  Max emails this run: unlimited")
    gi, gd, gn = ingest_gmail()
    print(f"  Gmail inserted: {gi}, duplicates: {gd}, non-job skipped: {gn}")

    print("-" * 45)
    print(f"Total jobs in DB: {collection.count_documents({})}")

    if cost_tracker["calls"] > 0:
        print(f"LLM calls: {cost_tracker['calls']}")
        print(f"LLM cost:  ${cost_tracker['total']:.5f}")
        print(f"Cost/job:  ${cost_tracker['total']/cost_tracker['calls']:.5f}")

    print("Done.\n")


# --- API HELPERS & ROUTES ---
def fix_id(doc: dict) -> dict:
    doc["_id"] = str(doc["_id"])
    return doc


def require_auth(request: Request) -> dict:
    user = request.session.get("user")
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated. Please login at /login",
        )
    return user


def _whitelist_email_key(addr: str) -> str:
    a = (addr or "").strip().lower()
    if not a or "@" not in a:
        return a
    local, _, domain = a.rpartition("@")
    if domain == "googlemail.com":
        domain = "gmail.com"
    if domain == "gmail.com":
        return f"{local.replace('.', '')}@gmail.com"
    return a


ALLOWED_EMAILS = {
    _whitelist_email_key(e)
    for e in os.getenv("ALLOWED_EMAILS", "").split(",")
    if e.strip()
}


@app.get("/login")
async def login(request: Request):
    redirect_uri = request.url_for("auth_callback")
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get("/auth/callback")
async def auth_callback(request: Request):
    token = await oauth.google.authorize_access_token(request)
    user = token.get("userinfo")
    if not user:
        raise HTTPException(status_code=400, detail="Failed to get user info from Google")

    email = (user.get("email") or "").strip()
    if ALLOWED_EMAILS and _whitelist_email_key(email) not in ALLOWED_EMAILS:
        raise HTTPException(status_code=403, detail=f"Access denied: {email} not authorized")

    request.session["user"] = {
        "email": email,
        "name": user.get("name"),
        "picture": user.get("picture"),
    }
    return RedirectResponse(url="/me")


@app.get("/me")
def me(request: Request):
    user = request.session.get("user")
    if not user:
        return {"logged_in": False, "message": "Visit /login to authenticate"}
    return {"logged_in": True, "user": user}


@app.get("/logout")
def logout(
    request: Request,
    redirect: bool = Query(
        False,
        description="If true, 303 redirect to /me so the browser shows logged_in false",
    ),
):
    request.session.clear()
    if redirect:
        return RedirectResponse(url="/me", status_code=303)
    return {"message": "Logged out. Visit /login to authenticate again.", "login": "/login"}


@app.post("/logout")
def logout_post(request: Request):
    request.session.clear()
    return {"message": "Logged out. Visit /login to authenticate again.", "login": "/login"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "db": DB_NAME,
        "jobs": collection.count_documents({}),
    }


@app.get("/jobs")
def list_jobs(
    request: Request,
    min_score: int = Query(0, description="Minimum LLM score"),
    max_score: int = Query(100, description="Maximum LLM score"),
    company: str = Query(None, description="Filter by company"),
    remote: bool = Query(None, description="Remote only"),
    status: str = Query(None, description="new | reviewed | applied | rejected"),
    source: str = Query(None, description="greenhouse | gmail_linkedin"),
    limit: int = Query(50, description="Max results"),
    skip: int = Query(0, description="Pagination offset"),
    user: dict = Depends(require_auth),
):
    query = {"llm_score": {"$gte": min_score, "$lte": max_score}}
    if company:
        query["company"] = company
    if remote is not None:
        query["remote"] = remote
    if status:
        query["status"] = status
    if source:
        query["source"] = source

    docs = list(collection.find(query).sort("llm_score", -1).skip(skip).limit(limit))
    return [fix_id(d) for d in docs]


@app.get("/jobs/{id}")
def get_job(id: str, user: dict = Depends(require_auth)):
    try:
        doc = collection.find_one({"_id": ObjectId(id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ID format")
    if not doc:
        raise HTTPException(status_code=404, detail="Job not found")
    return fix_id(doc)


@app.put("/jobs/{id}/status")
def update_status(
    id: str,
    status: str = Query(..., description="new | reviewed | applied | rejected"),
    user: dict = Depends(require_auth),
):
    valid = {"new", "reviewed", "applied", "rejected"}
    if status not in valid:
        raise HTTPException(status_code=400, detail=f"Status must be one of {valid}")
    try:
        result = collection.update_one(
            {"_id": ObjectId(id)},
            {"$set": {"status": status}},
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ID format")
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"updated": id, "status": status}


@app.get("/summary")
def summary(user: dict = Depends(require_auth)):
    total = collection.count_documents({})
    scored = collection.count_documents({"llm_score": {"$ne": None}})
    pipeline = [
        {"$match": {"llm_score": {"$ne": None}}},
        {
            "$group": {
                "_id": None,
                "avg_score": {"$avg": "$llm_score"},
                "max_score": {"$max": "$llm_score"},
                "min_score": {"$min": "$llm_score"},
            }
        },
    ]
    agg = list(collection.aggregate(pipeline))
    stats = agg[0] if agg else {}
    stats.pop("_id", None)
    return {
        "total_jobs": total,
        "scored_jobs": scored,
        "by_status": {
            s: collection.count_documents({"status": s})
            for s in ["new", "reviewed", "applied", "rejected"]
        },
        "by_company": {
            c: collection.count_documents({"company": c})
            for c in COMPANIES
        },
        "score_stats": stats,
    }


@app.post("/digest/send")
def send_digest(
    limit: int = Query(DIGEST_MAX_JOBS, ge=1, le=100),
    min_score: int = Query(DIGEST_MIN_SCORE, ge=0, le=100),
    only_new: bool = Query(True, description="If true, include only jobs with status='new'"),
    mark_sent: bool = Query(True, description="If true, set digest_sent_at on included jobs"),
    user: dict = Depends(require_auth),
):
    to_addr = (DIGEST_EMAIL_TO or user.get("email") or "").strip()
    if not to_addr:
        raise HTTPException(status_code=400, detail="No recipient found. Set DIGEST_EMAIL_TO.")

    query: dict = {"llm_score": {"$gte": min_score}}
    if only_new:
        query["status"] = "new"

    jobs = list(collection.find(query).sort("llm_score", -1).limit(limit))
    if not jobs:
        return {"sent": False, "reason": "No matching jobs for digest.", "count": 0}

    digest_body = (
        f"JobRadar Daily Digest\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Filters: min_score>={min_score}, only_new={only_new}\n"
        f"Count: {len(jobs)}\n\n"
        f"{build_digest_lines(jobs)}\n"
    )
    subject = f"JobRadar Digest — {len(jobs)} jobs"

    try:
        send_digest_email(to_addr=to_addr, subject=subject, body=digest_body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Digest send failed: {e}")

    if mark_sent:
        now_iso = datetime.now(timezone.utc).isoformat()
        ids = [job["_id"] for job in jobs]
        collection.update_many({"_id": {"$in": ids}}, {"$set": {"digest_sent_at": now_iso}})

    return {
        "sent": True,
        "to": to_addr,
        "count": len(jobs),
        "subject": subject,
    }


# ## S7 route: preview digest content without sending email
@app.get("/digest/preview")
def preview_digest(
    limit: int = Query(DIGEST_MAX_JOBS, ge=1, le=100),
    min_score: int = Query(DIGEST_MIN_SCORE, ge=0, le=100),
    only_new: bool = Query(True, description="If true, include only jobs with status='new'"),
    user: dict = Depends(require_auth),
):
    query: dict = {"llm_score": {"$gte": min_score}}
    if only_new:
        query["status"] = "new"

    jobs = list(collection.find(query).sort("llm_score", -1).limit(limit))
    if not jobs:
        return {"count": 0, "preview": "No matching jobs for digest.", "filters": {"limit": limit, "min_score": min_score, "only_new": only_new}}

    preview_text = (
        f"JobRadar Daily Digest (Preview)\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Filters: min_score>={min_score}, only_new={only_new}\n"
        f"Count: {len(jobs)}\n\n"
        f"{build_digest_lines(jobs)}\n"
    )
    return {
        "count": len(jobs),
        "filters": {"limit": limit, "min_score": min_score, "only_new": only_new},
        "preview": preview_text,
    }


if __name__ == "__main__":
    main()
