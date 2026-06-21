## mattermost_01_test.py
import requests

SERVER = "http://localhost:8065"
TOKEN = "61t,,,,,,,,,,,,,,,,,,,,,,,,,,,8q9oo"

headers = {
    "Authorization": f"Bearer {TOKEN}"
}

r = requests.get(
    f"{SERVER}/api/v4/users/me",
    headers=headers
)

print(r.status_code)
print(r.text)
