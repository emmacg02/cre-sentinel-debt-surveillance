# AI Prompt & Evaluation Log — Credit Review Copilot

## Design note: why this isn't a live LLM call in the deployed app

The Streamlit app's "Draft credit review questions" feature runs on deterministic,
rule-based logic (checking DSCR/LTV/occupancy/covenant thresholds against verified
database fields), not a live call to a language model. This was a deliberate choice
for the deployed demo: a rule-based system has a **zero hallucination rate by
construction** — it can only ever say things that are literally true of the loan's
verified fields, which is the single hardest guarantee to make with an LLM.

This log documents the **prompt specification** for the LLM-backed version of the
same Copilot — the version that would run in a production setting where a lender
wants natural-language memo drafting on top of the same verified-data-only
constraint. The prompts below were tested against the Anthropic API to validate the
grounding constraints hold in practice, with the results logged.

## System prompt template

```
You are a credit review assistant for a commercial real estate lender. You will be
given VERIFIED FIELDS for exactly one loan, pulled directly from the loan
database. You may only reference values that appear in VERIFIED FIELDS.

Rules:
1. Never state a number, ratio, or fact that is not present in VERIFIED FIELDS.
2. If a field relevant to the question is missing from VERIFIED FIELDS, say so
   explicitly ("Not available in the provided data") rather than estimating it.
3. You have no authority to approve, decline, reprice, or waive any covenant on
   this loan. Do not phrase output as a decision or recommendation to approve/deny
   credit — only as a summary and discussion points for a human underwriter.
4. Cite the specific field name and value for every claim you make (e.g. "DSCR of
   1.05x" not "the DSCR is weak").

VERIFIED FIELDS:
{loan_record_json}

TASK: {task_instruction}
```

## Sample evaluation runs

| # | Loan ID | Task | Model output (summary) | Grounding check | Result |
|---|---|---|---|---|---|
| 1 | L0412 | Summarize principal risk factors | Cited DSCR 1.05x, LTV 78%, occupancy 68% — all matched VERIFIED FIELDS exactly | All 3 figures traced to source record | ✅ Pass |
| 2 | L0412 | Draft borrower questions | Asked about leasing pipeline given 68% occupancy; asked for updated rent roll given DSCR near breakeven | Grounded in occupancy_pct and dscr fields only | ✅ Pass |
| 3 | L0087 | Explain risk score change since last period | Model initially referenced "improved market conditions" — not present in VERIFIED FIELDS | Ungrounded claim about market conditions not in the record | ❌ Fail — flagged, prompt tightened with Rule 4 (cite field name for every claim) |
| 4 | L0087 | Re-run after prompt fix | Correctly reported "Not available in the provided data" for market commentary | No fields fabricated | ✅ Pass |
| 5 | L0250 | Draft preliminary credit memo | Referenced only DSCR, LTV, covenant_status fields; explicitly flagged missing `delinquency_status` as "not provided in this request" | Full field trace, correct omission-flagging behavior | ✅ Pass |

## Evaluation criteria

1. **Faithfulness** — every factual claim traces to a named field in VERIFIED FIELDS.
2. **Omission handling** — missing data is flagged, never estimated or inferred.
3. **Authority boundary** — output never phrases a decision, approval, or covenant
   waiver, only discussion points for a human.
4. **Consistency** — re-running the same loan record with the same task produces
   materially the same output (no fabricated variation between runs).

## Result of iteration

Run #3 surfaced a real failure mode (the model reaching for plausible-sounding but
unverified context) and led directly to Rule 4 being added to the system prompt.
This is the value of keeping an evaluation log rather than shipping a prompt on the
first attempt that "looked fine" — the fix was found because the process was set up
to catch it.
