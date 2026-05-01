import os
import requests

API_KEY = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY") or "").strip()
if not API_KEY:
    raise SystemExit("Missing GEMINI_API_KEY / GOOGLE_API_KEY / GOOGLE_GENERATIVE_AI_API_KEY")

MODEL = (os.getenv("LLM_MODEL") or "gemini-1.5-flash").strip()

url_v1 = f"https://generativelanguage.googleapis.com/v1/models/{MODEL}:generateContent?key={API_KEY}"

data = {
    "contents": [
        {
            "parts": [
                {"text": "Hello, test message"}
            ]
        }
    ]
}

resp = requests.post(url_v1, json=data, timeout=30)
print("status:", resp.status_code)
print("body:", resp.text)

if resp.status_code == 404:
    url_v1beta = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"
    resp2 = requests.post(url_v1beta, json=data, timeout=30)
    print("\n--- fallback v1beta ---")
    print("status:", resp2.status_code)
    print("body:", resp2.text)
