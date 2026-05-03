# jobradar_01_greenhouse_ingest.py STEP3

# Step 1 of 6: Fetch jobs from Greenhouse public API → normalize → dedup → insert into MongoDB
# Step 2 of 6: LLM scoring (OpenAI or Anthropic, whichever key is available)
# Step 3 of 6: FastAPI routes to query jobs from MongoDB
# Run ingest:
#   .venv\Scripts\activate
#   python -m pip install -r requirements.txt
#   python jobradar_01_greenhouse_ingest.py
# Run API:                                                        ## Step 3 FastAPI: new run command
#   uvicorn jobradar_01_greenhouse_ingest:app --reload
#   then open http://127.0.0.1:8000/docs for Swagger UI

import requests
from datetime import datetime, timezone
from pymongo import MongoClient, errors
from dotenv import load_dotenv
from bson import ObjectId                                         ## Step 3 FastAPI: new import
from fastapi import FastAPI, Query, HTTPException                 ## Step 3 FastAPI: new import
import os
import json

# --- CONFIG ---
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "jobradar")
COLLECTION = os.getenv("COLLECTION", "jobs")

## Step 2 LLM: load API keys — script will use whichever is available
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
## Default 10 jobs/board if JOB_LIMIT unset (safe for dev). Set JOB_LIMIT=0 for no cap (full ingest).
JOB_LIMIT = max(0, int(os.getenv("JOB_LIMIT", "10")))

if not MONGO_URI:
    raise ValueError("MONGO_URI not found in .env")

## Step 2 LLM: warn if no LLM key found — scoring will be skipped
if not OPENAI_API_KEY and not ANTHROPIC_API_KEY:
    print("WARNING: No LLM API key found. Jobs will be inserted without scoring.")

# Step 1: removed non-working companies — only anthropic and mongodb confirmed working
COMPANIES = [
    "anthropic",
    "mongodb",
]

## Step 2 LLM: paste your CV here (or load from file)
CV_TEXT = """
Experienced Python developer. Skills: FastAPI, MongoDB, LLM integration,
REST APIs, cloud deployment (Render), Docker basics, OpenAI/Anthropic APIs,
agentic AI pipelines, RAG, MCP, tool calling.
Background: drones, robotics AI, full-stack development.
"""
## Step 2 LLM: optionally load CV from file instead of hardcoding above
## Uncomment to use:
# CV_PATH = os.getenv("CV_PATH", "cv.txt")
# if os.path.exists(CV_PATH):
#     with open(CV_PATH) as f:
#         CV_TEXT = f.read()

## Step 2 LLM: cost per 1K tokens (approximate, check current pricing)
OPENAI_COST_PER_1K_INPUT  = 0.00015
OPENAI_COST_PER_1K_OUTPUT = 0.0006
ANTHROPIC_COST_PER_1K_INPUT  = 0.00025
ANTHROPIC_COST_PER_1K_OUTPUT = 0.00125

## Step 2 LLM: running cost tracker
cost_tracker = {"total": 0.0, "calls": 0}

# --- CONNECT ---
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION]

collection.create_index("job_id", unique=True)

## Step 3 FastAPI: app instance
app = FastAPI(title="JobRadar API", version="0.3")


# --- NORMALIZE ---
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


## Step 2 LLM: scoring prompt
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


## Step 2 LLM: try OpenAI first, then Anthropic
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


# --- INGEST ---
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


# --- MAIN --- (ingest, run directly)
def main():
    print(f"\nJobRadar — Greenhouse Ingest + LLM Scoring")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Companies: {len(COMPANIES)}")

    if OPENAI_API_KEY:
        print(f"LLM: OpenAI (primary)")
    elif ANTHROPIC_API_KEY:
        print(f"LLM: Anthropic (fallback)")
    else:
        print(f"LLM: none — scoring disabled")

    print("-" * 40)

    total_inserted = 0
    total_skipped = 0

    for company in COMPANIES:
        print(f"\n  [{company}]")
        inserted, skipped = ingest_company(company)
        print(f"  [{company}] inserted: {inserted}, skipped: {skipped}")
        total_inserted += inserted
        total_skipped += skipped

    print("-" * 40)
    print(f"Total inserted: {total_inserted}")
    print(f"Total skipped (duplicates): {total_skipped}")
    print(f"Total in DB: {collection.count_documents({})}")

    if cost_tracker["calls"] > 0:
        print(f"LLM calls: {cost_tracker['calls']}")
        print(f"LLM cost:  ${cost_tracker['total']:.5f}")
        print(f"Cost/job:  ${cost_tracker['total']/cost_tracker['calls']:.5f}")

    print(f"Done.\n")


## Step 3 FastAPI: helper — convert MongoDB ObjectId to string for JSON
def fix_id(doc: dict) -> dict:
    doc["_id"] = str(doc["_id"])
    return doc


## Step 3 FastAPI: GET /health — liveness check
@app.get("/health")
def health():
    return {
        "status": "ok",
        "db": DB_NAME,
        "jobs": collection.count_documents({}),
    }


## Step 3 FastAPI: GET /jobs — list jobs with filters
@app.get("/jobs")
def list_jobs(
    min_score: int  = Query(0,    description="Minimum LLM score"),
    max_score: int  = Query(100,  description="Maximum LLM score"),
    company:   str  = Query(None, description="Filter by company"),
    remote:    bool = Query(None, description="Remote only"),
    status:    str  = Query(None, description="new | reviewed | applied | rejected"),
    source:    str  = Query(None, description="greenhouse | remotive | gmail"),
    limit:     int  = Query(50,   description="Max results"),
    skip:      int  = Query(0,    description="Pagination offset"),
):
    query = {
        "llm_score": {"$gte": min_score, "$lte": max_score}
    }
    if company: query["company"] = company
    if remote is not None: query["remote"] = remote
    if status:  query["status"] = status
    if source:  query["source"] = source

    docs = list(collection.find(query).sort("llm_score", -1).skip(skip).limit(limit))
    return [fix_id(d) for d in docs]


## Step 3 FastAPI: GET /jobs/{id} — single job by MongoDB _id
@app.get("/jobs/{id}")
def get_job(id: str):
    try:
        doc = collection.find_one({"_id": ObjectId(id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ID format")
    if not doc:
        raise HTTPException(status_code=404, detail="Job not found")
    return fix_id(doc)


## Step 3 FastAPI: PUT /jobs/{id}/status — update job status
@app.put("/jobs/{id}/status")
def update_status(id: str, status: str = Query(..., description="new | reviewed | applied | rejected")):
    valid = {"new", "reviewed", "applied", "rejected"}
    if status not in valid:
        raise HTTPException(status_code=400, detail=f"Status must be one of {valid}")
    try:
        result = collection.update_one(
            {"_id": ObjectId(id)},
            {"$set": {"status": status}}
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ID format")
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"updated": id, "status": status}


## Step 3 FastAPI: GET /summary — score distribution and counts
@app.get("/summary")
def summary():
    total = collection.count_documents({})
    scored = collection.count_documents({"llm_score": {"$ne": None}})
    pipeline = [
        {"$match": {"llm_score": {"$ne": None}}},
        {"$group": {
            "_id": None,
            "avg_score": {"$avg": "$llm_score"},
            "max_score": {"$max": "$llm_score"},
            "min_score": {"$min": "$llm_score"},
        }}
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


if __name__ == "__main__":
    main()