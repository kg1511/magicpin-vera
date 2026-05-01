#!/usr/bin/env python3
"""Tiny Anthropic /v1/messages smoke test (requests-based).

This matches the exact minimal shape used in Anthropic docs-style examples.

Usage (PowerShell):
  $env:ANTHROPIC_API_KEY = '<key>'
  $env:LLM_MODEL = 'claude-sonnet-4-6'
  c:/Users/gupta/Downloads/magicpin-ai-challenge/.venv/Scripts/python.exe .\anthropic_smoke_test.py
"""

import os
import requests


api_key = os.environ["ANTHROPIC_API_KEY"].strip()
model = (os.environ.get("LLM_MODEL") or "claude-sonnet-4-6").strip()

resp = requests.post(
    "https://api.anthropic.com/v1/messages",
    headers={
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    json={
        "model": model,
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "Hello"}],
    },
    timeout=30,
)

print("status:", resp.status_code)
print("body:", resp.text)
