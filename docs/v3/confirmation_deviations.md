# Confirmation run: deviations from the protocol and resource accounting

Appended record for `docs/v3/confirmation_protocol.md` (the protocol itself is not rewritten). Results:
`reports/v3_confirmation/summary.md`. Written 2026-11-01 after auditor note P36.

## Question set
- Declared: 897 questions on 501 videos. Used: **894 questions on 500 videos**. Video `7TljSpTBS9c` (3 questions)
  could not be downloaded (network failure while reading the archive; `data/lvb_fresh/manifest.json`), so it is
  not in `data/lvb_fresh/qa.jsonl` (894 lines) and no system ever answered it.
- Answers that failed (GPU out of memory) count as wrong in the primary analysis: one question
  (`TMe7oXMJoSM_0`) in each of the two 4B speech-text systems. Before 2026-10-30 such questions were skipped; that
  rule was replaced before the final analysis (step 69), and the skipped list is kept in
  `runs/v3_baseline/skipped.jsonl`.

## Answer calls against the declared allowance
The protocol declared 20,000 answer calls in total. The ledgers are per answer model, each with a 20,000-call cap,
so the aggregate allowance was not enforced. Ledger-charged calls (a request already in the cache is not charged):

| Ledger | Calls | Charged seconds | What it paid for |
|---|---:|---:|---|
| `budget_confirm_2b.json` | 7,587 | 9,404 | 2B, fresh set only |
| `budget_confirm_4b.json` | 7,587 | 92,449 | 4B, fresh set only |
| `budget_confirm_api.json` | 10,199 | 146,975 | Nemotron: fresh set **and** the old-test follow-ups |
| **Total** | **25,373** | | exceeds the declared 20,000 |

- Nemotron answers recorded: 8,046 on the fresh set (9 systems × 894), 330 on CG-Bench test and 2 × 1,212 on the
  Video-MMMU test tracks (old-test follow-ups, reported separately). Identical requests share one cached call, so
  answers (10,800) exceed charged calls (10,199). The ledger does not split calls by study; the fresh-set share is
  at most 8,046 calls.
- Physical requests are more than the charged calls: a Nemotron reply without an option letter is asked once more
  with a larger token budget (charged once), and network or rate-limit failures were retried (3,635 retry lines in
  `runs/confirmation_api*.log`). Tokens were not logged per call. This deviation is about resources; it does not
  change any recorded answer.

## Statistics
- p-values are approximate bootstrap-tail estimates, 2 × (tail count + 1) / (10,000 + 1): H3 and H4 ≈ 0.0002, Holm
  ≈ 0.0008. They replace the earlier "<0.0001" (research log step 73).
