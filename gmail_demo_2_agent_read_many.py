# gmail_demo_2_agent_read_many.py

import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()


SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# -----------------------------
# Gmail auth
# -----------------------------
def get_gmail_service():
    creds = None

    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open("token.json", "w") as f:
            f.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


# -----------------------------
# Gmail read
# -----------------------------
def get_header(headers, name):
    for h in headers:
        if h["name"].lower() == name.lower():
            return h["value"]
    return ""


def read_emails(service, query="newer_than:7d", max_results=10):
    res = service.users().messages().list(
        userId="me", q=query, maxResults=max_results
    ).execute()

    msgs = res.get("messages", [])
    results = []

    for m in msgs:
        msg = service.users().messages().get(
            userId="me",
            id=m["id"],
            format="metadata",
            metadataHeaders=["From", "Subject", "Date"],
        ).execute()

        headers = msg["payload"]["headers"]

        results.append({
            "subject": get_header(headers, "Subject"),
            "from": get_header(headers, "From"),
            "date": get_header(headers, "Date"),
            "snippet": msg.get("snippet", "")
        })

    return results


# -----------------------------
# OpenAI summarize
# -----------------------------
def summarize_emails(emails):
#    client = OpenAI()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    text = "\n".join([
        f"From: {e['from']}\nSubject: {e['subject']}\nSnippet: {e['snippet']}\n"
        for e in emails
    ])

    prompt = f"""
Summarize these emails briefly:

{text}
"""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return resp.choices[0].message.content


# -----------------------------
# Main agent
# -----------------------------
def main():
    service = get_gmail_service()

    emails = read_emails(service)

    print("\n--- RAW EMAILS ---\n")
    for e in emails:
        print(e["subject"], "|", e["from"])

    summary = summarize_emails(emails)

    print("\n--- AI SUMMARY ---\n")
    print(summary)


if __name__ == "__main__":
    main()
