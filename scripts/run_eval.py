"""
Day 18 Step 7: Run the eval set against the real router and report
pass/fail per case.

Tool-name mapping (eval set label -> actual router output):
  "rag"            -> "rag"
  "refusal"        -> "guardrail_refusal"
  "decline"        -> "none_matched"   (out-of-scope, no keyword match, not blocked)
  "refusal_or_rag" -> "guardrail_refusal" OR "rag" (either acceptable)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
from pathlib import Path
from src.ai_assistant.router import route_message

EVAL_PATH = Path("data/eval/rag_eval_set.jsonl")

TOOL_MAP = {
    "rag": {"rag"},
    "refusal": {"guardrail_refusal"},
    "decline": {"none_matched"},
    "refusal_or_rag": {"guardrail_refusal", "rag"},
}


def run_eval():
    cases = [json.loads(line) for line in EVAL_PATH.read_text().splitlines() if line.strip()]
    results = []

    for case in cases:
        response = route_message(case["question"])
        actual_tool = response["tool"]
        acceptable = TOOL_MAP.get(case["expected_tool"], {case["expected_tool"]})
        passed = actual_tool in acceptable

        results.append({
            "id": case["id"],
            "question": case["question"],
            "expected_tool": case["expected_tool"],
            "actual_tool": actual_tool,
            "passed": passed,
        })

    return results


if __name__ == "__main__":
    results = run_eval()
    failures = [r for r in results if not r["passed"]]

    print(f"{len(results) - len(failures)}/{len(results)} passed\n")

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r['id']}: expected={r['expected_tool']} actual={r['actual_tool']}")

    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  {f['id']}: \"{f['question']}\" -> expected {f['expected_tool']}, got {f['actual_tool']}")