#  gmail_demo_1_read_search.py

from __future__ import annotations

import os
from typing import List, Dict, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Demo 1: read/search only
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def get_gmail_service():
    creds: Optional[Credentials] = None

    # token.json stores the user's access + refresh tokens after first login
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    # If no valid creds, run local OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json",
                SCOPES,
            )
            creds = flow.run_local_server(port=0)

        with open("token.json", "w", encoding="utf-8") as token_file:
            token_file.write(creds.to_json())

    service = build("gmail", "v1", credentials=creds)
    return service


def get_header(headers: List[Dict[str, str]], name: str) -> str:
    for h in headers:
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


def search_messages(service, query: str, max_results: int = 10):
    result = (
        service.users()
        .messages()
        .list(userId="me", q=query, maxResults=max_results)
        .execute()
    )
    return result.get("messages", [])


def read_message_metadata(service, msg_id: str):
    msg = (
        service.users()
        .messages()
        .get(
            userId="me",
            id=msg_id,
            format="metadata",
            metadataHeaders=["From", "Subject", "Date"],
        )
        .execute()
    )
    return msg


def main():
    service = get_gmail_service()

    # Change this to whatever you want
    # Examples:
    # query = "newer_than:30d"
    # query = 'from:amazon newer_than:30d'
    # query = 'label:inbox is:unread'
    query = "newer_than:30d"

    messages = search_messages(service, query=query, max_results=10)

    print(f"query: {query}")
    print(f"matches: {len(messages)}")
    print("-" * 80)

    if not messages:
        print("No messages found.")
        return

    for i, item in enumerate(messages, start=1):
        msg = read_message_metadata(service, item["id"])
        payload = msg.get("payload", {})
        headers = payload.get("headers", [])

        subject = get_header(headers, "Subject")
        sender = get_header(headers, "From")
        date = get_header(headers, "Date")
        snippet = msg.get("snippet", "")

        print(f"[{i}] Subject: {subject}")
        print(f"    From:    {sender}")
        print(f"    Date:    {date}")
        print(f"    Snippet: {snippet}")
        print("-" * 80)


if __name__ == "__main__":
    main()

