"""Utility labels from the frozen teacher, under an explicit call budget.

Two Head B schemes, compared at the SAME per-question budget of B = 1 + M*K calls:

  independent   R(T, {})  plus  R(T, {c}) for B-1 candidates  ->  rows with an EMPTY history
  prefix        M selection chains of length K. Each chain adds one candidate at a
                time; each step yields R(S + {c}) - R(S) with S = the chain so far.
                The shared baseline R(T, {}) and identical prefixes are reused via
                the teacher cache, so the real call count is <= B.

The prefix scheme observes FEWER distinct candidates per question (it spends calls
on deeper contexts), so it is not an equally complete supervision of single-frame
value; the pilot measures whether that trade helps.

Head A labels use SEPARATE interventions on an empty reference context
(T0 = no transcript, S0 = no frames):
  text   R({t_i}, {}) - R({}, {})        visual   R({}, {f_i}) - R({}, {})
"""

from __future__ import annotations

import random

from .records import QuestionPool, UtilityRow


def independent_labels(pool: QuestionPool, teacher, budget: int, seed: int = 0) -> list[UtilityRow]:
    """Baseline + (budget - 1) single-frame additions (random subset if the pool is larger)."""
    base = teacher.quality(pool, [])
    cands = list(pool.candidates)
    random.Random(f"{seed}:{pool.qa_id}").shuffle(cands)
    rows = []
    for c in cands[: max(0, budget - 1)]:
        after = teacher.quality(pool, [c])
        rows.append(UtilityRow(pool.qa_id, pool.video_id, c.id, [], base, after, after - base, "independent"))
    return rows


def prefix_labels(pool: QuestionPool, teacher, chains: int = 2, length: int = 4, seed: int = 0,
                  epsilon: float = 0.3) -> list[UtilityRow]:
    """M chains of length K. Chain 0 follows MobileCLIP similarity with epsilon-random
    exploration; later chains are random orders (mix of exploitation and coverage)."""
    rng = random.Random(f"{seed}:{pool.qa_id}")
    base = teacher.quality(pool, [])
    rows = []
    for m in range(chains):
        chosen, prev = [], base
        rest = list(pool.candidates)
        for _ in range(min(length, len(rest))):
            if m == 0 and rng.random() > epsilon:
                c = max(rest, key=lambda x: x.clip_sim)
            else:
                c = rng.choice(rest)
            after = teacher.quality(pool, chosen + [c])
            rows.append(UtilityRow(pool.qa_id, pool.video_id, c.id, [h.id for h in chosen], prev, after,
                                   after - prev, "prefix"))
            chosen.append(c)
            rest.remove(c)
            prev = after
    return rows


def head_a_labels(pool: QuestionPool, teacher, path_a_items) -> list[dict]:
    """Separate text and frame interventions for each Path A hot moment.

    ``path_a_items``: list of (segment, candidate) pairs, the hot-moment line and
    its representative frame. Returns one dict per item with both signed gains.
    """
    base = teacher.quality(pool, [], transcript=[])
    out = []
    for seg, cand in path_a_items:
        r_text = teacher.quality(pool, [], transcript=[seg])
        r_vis = teacher.quality(pool, [cand], transcript=[])
        out.append({"qa_id": pool.qa_id, "video_id": pool.video_id, "segment_id": seg.id,
                    "candidate_id": cand.id, "base": base, "text_gain": r_text - base,
                    "visual_gain": r_vis - base})
    return out
