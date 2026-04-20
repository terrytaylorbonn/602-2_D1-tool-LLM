# main.py

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"ok": True, "message": "hello from Render!"}

from fastapi import Request

@app.post("/github-webhook")
async def github_webhook(request: Request):
    data = await request.json()
    print("EVENT:", request.headers.get("X-GitHub-Event"))
    print("PAYLOAD_KEYS:", list(data.keys())[:10])
    return {"ok222": True}