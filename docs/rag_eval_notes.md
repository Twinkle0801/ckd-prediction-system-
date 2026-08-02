# RAG Eval Notes (Day 17)

## Eval set and results

10 questions expected to be answerable from the 3 reference documents, plus
3 off-topic questions expected to be declined. Run via `eval_rag_questions.py`.

### Answerable questions

| Question | Tool reached | has_grounded_context | Sources |
|---|---|---|---|
| "What does a serum creatinine of 3.8 typically indicate?" | rag | True | ckd_stages.md, lab_normal_ranges.md |
| "What is a normal range for potassium?" | rag | True | lab_normal_ranges.md |
| "What does high blood urea indicate?" | rag | True | lab_normal_ranges.md |
| "What are risk factors for CKD?" | **none_matched** | N/A | — |
| "Does diabetes increase CKD risk?" | **none_matched** | N/A | — |
| "What is Stage 3 CKD?" | rag | True | ckd_stages.md |
| "What does albumin in urine typically indicate?" | rag | True | lab_normal_ranges.md |
| "What normal range is expected for hemoglobin?" | rag | True | lab_normal_ranges.md |
| "How does hypertension relate to kidney disease?" | **none_matched** | N/A | — |
| "What does a low eGFR mean?" | rag | True | ckd_stages.md |

**7/10 correctly routed and answered. 3/10 never reached the RAG tool at all.**

### Off-topic questions (expected to be declined)

| Question | Tool reached | has_grounded_context |
|---|---|---|
| "What does love mean?" | rag | False (correctly declined) |
| "What is the recipe for chocolate cake?" | none_matched | N/A (never answered either way) |
| "What does stock market volatility typically indicate?" | rag | False (correctly declined) |

All 3 off-topic questions handled correctly — either declined by the RAG
tool's own relevance check, or never routed at all. No false positives.

## Finding

**Every question that actually reached `call_rag_tool` was handled
correctly** -- right sources, right refusals when nothing relevant existed.
The gap is entirely in the **router's keyword-matching**, which is too
narrow to catch natural phrasing variation:

- "What **are** risk factors..." -- doesn't contain "what does"
- "**Does** diabetes increase..." -- doesn't start with "what"
- "**How does** hypertension relate..." -- "how does" isn't "what does"

All three are genuinely answerable by the reference documents (risk
factors and hypertension are both directly covered in
`risk_factors.md`), but the router silently drops them into
`none_matched` before the RAG tool -- which would have answered them
correctly -- ever runs.

## Conclusion

This is the same underlying lesson as the Day 15 guardrail-bypass findings
and the Day 16 grounding-check bug: **keyword matching alone is fragile
against natural phrasing variation**, and expanding the keyword list
piecemeal to catch these three specific phrasings would just repeat the
same whack-a-mole pattern rather than fix the actual problem.

**Action for Day 18:** this becomes concrete evidence for the improvement
already anticipated in `router.py`'s own docstring -- replacing keyword
matching with the LLM's native tool-use/function-calling (letting the
model itself decide, based on understanding the question, whether it
needs the prediction tool, the RAG tool, or neither) rather than
pattern-matching exact phrases. The 3 failing questions above become
concrete test cases to verify that fix actually closes this gap.