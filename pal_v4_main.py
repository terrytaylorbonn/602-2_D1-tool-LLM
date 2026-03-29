# pal_v4_main.py

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import os

# import your existing logic
#from pal_v4 import run_plan   # <-- adjust this
from pal_v4 import run_plan, run_ingest

app = FastAPI()

# --- API AUTH 01 ---
API_KEY = os.getenv("PAL_API_KEY")

def check_api_key(x_api_key: str = Header(default="")):
    print(API_KEY)
    print(x_api_key)
    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

class Request(BaseModel):
    prompt: str

# --- API INGEST 02 ---
class EventRequest(BaseModel):
    entity: str
    event_type: str
    location: str
    status: str
    note: str
    timestamp: str | None = None


@app.get("/")
def root():
    return {"status": "ok"}

# --- API FIX 03 + AUTH ---
@app.post("/run")
def run(req: Request, x_api_key: str = Header(default="")):
    check_api_key(x_api_key)
    return run_plan(req.prompt)

# --- API INGEST 03 ---
@app.post("/ingest")
def ingest(req: EventRequest, x_api_key: str = Header(default="")):
    check_api_key(x_api_key)
    return run_ingest(req.model_dump(exclude_none=True))
