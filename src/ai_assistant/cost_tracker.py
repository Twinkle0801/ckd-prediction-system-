"""
Day 18: per-query cost logging for the RAG tool.
"""

import json
import time
from pathlib import Path

COST_LOG_PATH = Path("logs/rag_cost_log.jsonl")

# claude-sonnet-4-5 pricing -- PLACEHOLDER, confirm exact current rate
INPUT_COST_PER_1M = 3.00
OUTPUT_COST_PER_1M = 15.00


def log_query_cost(question: str, input_tokens: int, output_tokens: int, cache_hit: bool = False):
    cost = (input_tokens / 1_000_000 * INPUT_COST_PER_1M) + \
           (output_tokens / 1_000_000 * OUTPUT_COST_PER_1M)
    entry = {
        "timestamp": time.time(),
        "question": question[:100],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 6),
        "cache_hit": cache_hit,
    }
    COST_LOG_PATH.parent.mkdir(exist_ok=True)
    with open(COST_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")