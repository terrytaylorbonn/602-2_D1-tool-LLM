# pal_v4_main.py

from fastapi import FastAPI
from pydantic import BaseModel

# import your existing logic
from pal_v4 import run_plan   # <-- adjust this

app = FastAPI()

class Request(BaseModel):
    prompt: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/run")
def run(req: Request):
    # replace this with your PAL function
#    result = f"PAL received: {req.prompt}"
    result = run_plan(req.prompt)
    # return {"result (run_plan)": result}
    return result