## Grounding check bug found and fixed (Day 16)

Red-teaming the grounding check (`src/ai_assistant/grounding.py`) with 5
adversarial explanation texts surfaced a real bug, not just a theoretical
gap:

**Bug:** the original exclusion filter (`float(n) >= 2`) meant to exempt
harmless small counting numbers (e.g. "3-4 sentences") accidentally
exempted almost all real clinical values too -- SHAP contributions,
creatinine, and potassium values are typically well under 2. This meant
fabricated small decimals (a fake combined SHAP total, a fake reference
range) were never even evaluated by the check.

**Fix:** replaced the magnitude-based exclusion with an exact-match
exclusion list (`{"0.0", "1.0", "2.0", "3.0", "4.0", "5.0"}`), so only
genuine small whole-number counts are exempted, not any number under 2.

**Verification (5 edge cases tested before and after the fix):**

| Case | Before fix | After fix |
|---|---|---|
| Probability phrased as "93%" instead of 0.93 | Correctly flagged | Correctly flagged |
| Casual rounding ("around 15" for 15.4) | Correctly flagged | Correctly flagged |
| LLM's own arithmetic on two real SHAP values (-0.11) | **Missed (bug)** | **Now caught** |
| A real number attributed to the wrong feature | Missed | Still missed (see below) |
| A fabricated generic reference range (0.6-1.3) | **Missed (bug)** | **Now caught** |

**Remaining known limitation:** the check is number-based, not
feature-attribution-based -- if an explanation cites a real number from
the source data but attaches it to the wrong lab value (e.g. "potassium
of 15.4" when 15.4 was actually the hemoglobin value), the check will not
catch it, since the number itself is genuinely present in the allowed set.
Closing this gap would require checking feature-value pairs together, not
just the numbers in isolation -- a candidate improvement for Day 18's
eval work, alongside the intent-classifier idea already logged there.

**Minor cosmetic quirk (not a safety issue):** the number-extraction regex
reads a hyphen in a range like "0.6-1.3" as a minus sign, reporting the
flagged number as -1.3 rather than 1.3. The case is still correctly
flagged as ungrounded either way, so this doesn't affect safety, only the
readability of the reported `ungrounded_numbers` list.