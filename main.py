# main.py

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"ok": True, "message": "hello from Render!"}

#-------------------------------------

import hmac, hashlib
from fastapi import Request, HTTPException

SECRET = "webhooksecret"

@app.post("/github-webhook")
async def github_webhook(request: Request):
    body = await request.body()
    sig = request.headers.get("X-Hub-Signature-256", "")
    expected = "sha256=" + hmac.new(
        SECRET.encode(),
        body,
        hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(sig, expected):
        raise HTTPException(status_code=403, detail="bad signature")

    data = await request.json()
    print("EVENT(3):", request.headers.get("X-GitHub-Event"))
    print("PAYLOAD_KEYS:", list(data.keys())[:10])
    return {"ok": True}



# from fastapi import Request

# @app.post("/github-webhook")
# async def github_webhook(request: Request):
#     data = await request.json()
#     print("EVENT:", request.headers.get("X-GitHub-Event"))
#     print("PAYLOAD_KEYS:", list(data.keys())[:10])
#     return {"ok222": True}