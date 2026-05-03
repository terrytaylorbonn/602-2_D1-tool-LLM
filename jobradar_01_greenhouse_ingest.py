# jobradar_01_greenhouse_ingest.py

# Step 1 of 6: Fetch jobs from Greenhouse public API → normalize → dedup → insert into MongoDB
# No LLM, no FastAPI, no email — just the pipeline core
# Run (same interpreter that pip used — see below):
#   .venv\\Scripts\\activate   # Git Bash/PowerShell: activate venv first
#   python -m pip install -r requirements.txt
#   python jobradar_01_greenhouse_ingest.py
# On Windows, `py -3 …` often points at *another* Python than venv pip → ModuleNotFoundError.

import requests
from datetime import datetime, timezone
from pymongo import MongoClient, errors
from dotenv import load_dotenv
import os

# --- CONFIG ---
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "jobradar")        # default if not in .env
COLLECTION = os.getenv("COLLECTION", "jobs")       # default if not in .env

if not MONGO_URI:
    raise ValueError("MONGO_URI not found in .env")



# Add/remove companies as needed
# These must match the Greenhouse "board token" (usually company name lowercase)
# Verify at: https://boards-api.greenhouse.io/v1/boards/{company}/jobs
COMPANIES = [
    "anthropic",
    "palantir",
    "mongodb",
    "openai",
    "scale-ai",
    "huggingface",
]

# --- CONNECT ---
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION]

# Unique index on job_id to enforce dedup at DB level
collection.create_index("job_id", unique=True)


# --- NORMALIZE ---
def normalize_job(job: dict, company: str) -> dict:
    """
    Map raw Greenhouse job fields to canonical JobRadar schema.
    """
    # Greenhouse location can be a dict or string depending on version
    location_raw = job.get("location", {})
    if isinstance(location_raw, dict):
        location = location_raw.get("name", "Unknown")
    else:
        location = str(location_raw)

    # Stable unique ID: greenhouse numeric job id + company
    job_id = f"gh_{company}_{job.get('id', '')}"

    return {
        "job_id": job_id,
        "source": "greenhouse",
        "company": company,
        "role": job.get("title", "Unknown"),
        "location": location,
        "remote": "remote" in location.lower(),
        "url": job.get("absolute_url", ""),
        "posted_date": job.get("updated_at", ""),  # Greenhouse uses updated_at
        "seen_date": datetime.now(timezone.utc).isoformat(),
        "llm_score": None,      # populated in Step 2
        "llm_notes": None,      # populated in Step 2
        "status": "new",
    }


# --- INGEST ---
def ingest_company(company: str) -> tuple[int, int]:
    """
    Fetch all jobs for one company, insert new ones, skip duplicates.
    Returns (inserted, skipped).
    """
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

    inserted = 0
    skipped = 0

    for job in jobs:
        normalized = normalize_job(job, company)
        try:
            collection.insert_one(normalized)
            inserted += 1
        except errors.DuplicateKeyError:
            skipped += 1

    return inserted, skipped


# --- MAIN ---
def main():
    print(f"\nJobRadar — Greenhouse Ingest")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Companies: {len(COMPANIES)}")
    print("-" * 40)

    total_inserted = 0
    total_skipped = 0

    for company in COMPANIES:
        inserted, skipped = ingest_company(company)
        print(f"  [{company}] inserted: {inserted}, skipped: {skipped}")
        total_inserted += inserted
        total_skipped += skipped

    print("-" * 40)
    print(f"Total inserted: {total_inserted}")
    print(f"Total skipped (duplicates): {total_skipped}")
    print(f"Total in DB: {collection.count_documents({})}")
    print(f"Done.\n")


if __name__ == "__main__":
    main()