# Sample agent outputs

Unedited drafts from `python agent/retention_agent.py --all-risk` with `OLLAMA_MODEL=llama3.2`
(3B) on 2026-10-19. Both passed the guardrail on the first attempt.

They're kept as-is, including the model's mistakes, so the quality of a small local model is visible:

- Timing advice that doesn't fit a lapsed cohort ("send 7-10 days after the last order" for customers
  with a 184-day median gap).
- The Dormant Buyers follow-up invents a phone number (01234 567890), and its notes call the 402-day
  median an "average".
- Both generated before a pluralisation fix, so the Dormant snapshot reads "1 orders".

The **Cohort snapshot** section is rendered from SQL, not written by the model, so its figures are exact.
Everything under **Analyst notes** and below is model output and needs a human edit before sending.
