# INVALID — do not cite these numbers

This pilot (2026-09-27, 40 LongVideoBench videos) was produced before two bugs
were fixed, found in a later audit:

1. **The answerer received no transcript on 35 of 40 videos.** `rank_bm25`'s IDF
   is negative when a term appears in most documents; 29/40 clips form a single
   retrieval unit, so every unit scored < 0, no retrieval window was found, and
   `pack_excerpt` (which only drew from windows) returned an empty excerpt. As a
   consequence the clean / targeted / control transcript conditions were
   identical from the answerer's point of view, which is why quality is flat
   across conditions in `summary.md`. Fixed in commits "fix(retrieval): use
   non-negative BM25 IDF ..." and "fix(packing): fall back to the whole
   transcript ...".
2. **Evidence extraction** took apostrophes as quote marks (29/440 real
   questions) and ignored all but the first quote.

The files are kept only as a record of what was run. The replacement result is
in `reports/lvb_full/`.
