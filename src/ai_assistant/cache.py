"""
Day 18: exact-match cache for repeated RAG questions.
In-memory only -- resets on process restart. Fine for a portfolio
project; swap for Redis/disk if you want persistence across restarts.
"""

import hashlib

_query_cache: dict[str, dict] = {}


def get_cache_key(question: str) -> str:
    return hashlib.sha256(question.strip().lower().encode()).hexdigest()


def get_cached(question: str):
    return _query_cache.get(get_cache_key(question))


def set_cached(question: str, result: dict):
    _query_cache[get_cache_key(question)] = result