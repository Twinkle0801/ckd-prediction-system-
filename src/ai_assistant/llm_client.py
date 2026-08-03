# src/ai_assistant/llm_client.py
"""
Thin wrapper around the Anthropic client so the rest of the AI layer
(router, tools) never imports the SDK directly -- if the provider ever
changes, only this file needs to change.

MOCK MODE: if ANTHROPIC_API_KEY is missing OR the real API call fails
(e.g. no credits yet), simple_completion() falls back to a clearly-labeled
canned response instead of crashing. This lets the rest of the AI layer
(router, guardrails, tool dispatch) be built and tested with zero API
cost -- real calls only happen once credits/billing are set up, whenever
that happens.
"""

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

MODEL_NAME = "claude-sonnet-4-5"


def get_client():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


def simple_completion(prompt: str, max_tokens: int = 200) -> str:
    """
    Minimal single-turn call. Falls back to a mock response if no API key
    is set, or if the real call fails for any reason (e.g. billing/credits) --
    prefixed with [MOCK] so it's never mistaken for a real model response.
    """
    client = get_client()

    if client is None:
        return "[MOCK] No API key configured -- this is a placeholder response."

    try:
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except anthropic.APIError as e:
        return f"[MOCK] Real API call failed ({type(e).__name__}) -- placeholder response used instead."

def completion_with_usage(prompt: str, max_tokens: int = 200) -> dict:
    """
    Same as simple_completion, but also returns token usage for cost
    tracking. Used by call_rag_tool so Day 18 cost logging reflects real
    API usage instead of estimates.
    """
    client = get_client()

    if client is None:
        return {
            "text": "[MOCK] No API key configured -- this is a placeholder response.",
            "input_tokens": 0,
            "output_tokens": 0,
        }

    try:
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return {
            "text": response.content[0].text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
    except anthropic.APIError as e:
        return {
            "text": f"[MOCK] Real API call failed ({type(e).__name__}) -- placeholder response used instead.",
            "input_tokens": 0,
            "output_tokens": 0,
        }