"""Fourth-frame completion: a deployment-matched label target for frame selection.

Why this exists
---------------
Head B and the residual scorers learn from *single-frame* labels
R(T, {c}) - R(T, {}). Deployment shows the answerer FOUR frames together with
the retained transcript, and the main report found that frames interact with
speech: of 132 frames that lose their value when speech is present, 62 fail
because the transcript disrupts an answer the frame alone got right. A
single-frame label cannot see that interaction, nor redundancy between frames.

This module measures and learns the slot that deployment actually fills:

    anchor S3   = top-3 frames of the strongest simple baseline
                  (zero-shot transcript relevance), chosen without labels
    label(c)    = R(T, S3 + {c})  for EVERY remaining pool candidate c

Because every completion of every question is labelled, the four-frame QA
accuracy of ANY fourth-frame policy is known exactly, offline, with no new
teacher calls: the policy's pick is looked up in the table. This also gives
the oracle (best possible fourth frame), i.e. the true headroom of the slot.

The learned scorer is a pairwise ridge residual anchored on the baseline's own
rank-4 choice: with infinite regularisation it reproduces the baseline exactly.
Its features describe the candidate AND its relation to the already-chosen
frames and the retained transcript (redundancy, novelty). Labels, gold
indices and the three-frame outcome R(T, S3) are never inference features
(R(T, S3) would cost an extra answerer call per query).

Prior work this follows (no novelty claimed for the mechanism): contextual
submodular list prediction (Ross et al., ICML 2013), history-conditioned frame
selection (ReFoCUS, MarKey), and contextual importance labels (FrameOracle).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .records import QuestionPool
from .residual_selector import _cosine, _tokens, feature_names, ranked_indices, raw_features

ANCHOR_SIZE = 3                       # frames fixed by the baseline; the learner fills slot 4
RIDGES = (0.1, 1.0, 10.0, 100.0)      # fixed grid, chosen by inner leave-one-video-out
SET_FEATURES = (
    "max_image_cos_to_anchor",        # visual redundancy with the chosen frames
    "mean_image_cos_to_anchor",
    "min_time_gap_to_anchor",         # seconds to the nearest chosen frame / video duration
    "ocr_novel_vs_anchor",            # share of the candidate's OCR words absent from chosen OCR
    "ocr_novel_vs_anchor_and_speech",  # ...and absent from the retained transcript too
    "near_novel_vs_speech",           # share of its nearby speech absent from the retained transcript
    "question_words_newly_covered",   # question words in its OCR not covered by chosen OCR or speech
    "max_ocr_cos_to_anchor",          # OCR-embedding redundancy with the chosen frames
    "shares_path_with_all_anchor",    # 1 if every chosen frame came from the same path as this one
)


# ---------------------------------------------------------------------------
# Anchor and candidates (no labels involved)
# ---------------------------------------------------------------------------


def relevance_scores(pool: QuestionPool) -> np.ndarray:
    """The strongest simple baseline: zero-shot transcript relevance per candidate."""
    return np.array([c.head_a_visual for c in pool.candidates], dtype=float)


def anchor_and_rest(pool: QuestionPool, size: int = ANCHOR_SIZE) -> tuple[list[int], list[int]]:
    """Baseline order split into the fixed anchor and the completion candidates.

    ``rest`` keeps the baseline's order, so ``rest[0]`` is exactly the frame the
    baseline itself would use as its fourth frame.
    """
    order = ranked_indices(pool, relevance_scores(pool))
    return order[:size], order[size:]


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def completion_feature_names() -> tuple[str, ...]:
    return feature_names(False) + SET_FEATURES


def raw_completion_features(pool: QuestionPool, anchor: list[int], rest: list[int]) -> np.ndarray:
    """One row per completion candidate: its own features + its relation to the anchor."""
    own = raw_features(pool, context=False)
    cands = pool.candidates
    anchor_ocr = set().union(*(_tokens(cands[i].ocr_text) for i in anchor)) if anchor else set()
    speech = _tokens(pool.transcript_text)
    q = _tokens(pool.question)
    covered_q = q & (anchor_ocr | speech)
    duration = max(1.0, pool.duration_s)
    rows = []
    for i in rest:
        c = cands[i]
        ocr, near = _tokens(c.ocr_text), _tokens(c.near_text)
        img = [_cosine(c.emb, cands[j].emb) for j in anchor] or [0.0]
        ocr_cos = [_cosine(c.ocr_emb, cands[j].ocr_emb) for j in anchor] or [0.0]
        gap = min((abs(c.time_s - cands[j].time_s) for j in anchor), default=duration) / duration
        same_path = float(bool(anchor) and all(set(c.paths) & set(cands[j].paths) for j in anchor))
        set_row = [max(img), float(np.mean(img)), gap,
                   len(ocr - anchor_ocr) / max(1, len(ocr)),
                   len(ocr - anchor_ocr - speech) / max(1, len(ocr)),
                   len(near - speech) / max(1, len(near)),
                   len((ocr & q) - covered_q) / max(1, len(q)),
                   max(ocr_cos), same_path]
        rows.append(list(own[i]) + set_row)
    x = np.asarray(rows, dtype=float).reshape(len(rows), len(completion_feature_names()))
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite completion feature")
    return x


def completion_design(pool: QuestionPool, anchor: list[int], rest: list[int], extra: np.ndarray | None = None):
    """Standardise within the question's completion set (label-free).

    ``extra``: optional additional label-free feature columns, one row per
    ``rest`` candidate (e.g. local board features); appended before scaling.

    Returns (z, anchor_score): the residual's anchor score is the baseline's own
    preference, a strictly decreasing function of its rank, so zero weights pick
    ``rest[0]``, the baseline's fourth frame.
    """
    x = raw_completion_features(pool, anchor, rest)
    if extra is not None:
        x = np.hstack([x, np.asarray(extra, dtype=float).reshape(len(rest), -1)])
    if not len(x):
        return x, np.zeros(0)
    scale = x.std(axis=0)
    z = (x - x.mean(axis=0)) / np.where(scale > 1e-8, scale, 1.0)
    base = -np.arange(len(rest), dtype=float)
    base = (base - base.mean()) / max(base.std(), 1e-8)
    return z, base


# ---------------------------------------------------------------------------
# Labels -> pairwise blocks -> ridge
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CompletionQuestion:
    """Everything one question contributes: design, anchor score and exact labels."""

    qa_id: str
    video_id: str
    rest_ids: list[str]
    z: np.ndarray
    base: np.ndarray
    correct: np.ndarray          # R(T, S3 + {c}) per completion candidate, in rest order
    three_frame: float           # R(T, S3): reference only, never a feature


def build_questions(pools: dict, rows: list[dict], extra: dict | None = None) -> list[CompletionQuestion]:
    """Join completion labels to label-free designs, checking the labels are complete.

    ``extra[qa_id]``: optional extra feature rows aligned with the baseline's rest order.
    """
    by: dict[str, dict] = {}
    for r in rows:
        labels = by.setdefault(r["qa_id"], {})
        if r["candidate_id"] in labels:
            raise ValueError(f"{r['qa_id']}: duplicate label for {r['candidate_id']}")
        if r["before"] not in (0.0, 1.0) or r["after"] not in (0.0, 1.0):
            raise ValueError(f"{r['qa_id']}: labels must be binary correctness")
        labels[r["candidate_id"]] = r
    out = []
    for q in sorted(by):
        pool = pools[q]
        anchor, rest = anchor_and_rest(pool)
        ids = [pool.candidates[i].id for i in rest]
        labels = by[q]
        validate_question_rows(pool, list(labels.values()))
        z, base = completion_design(pool, anchor, rest, None if extra is None else extra[q])
        out.append(CompletionQuestion(q, pool.video_id, ids, z, base,
                                      np.array([labels[i]["after"] for i in ids], dtype=float),
                                      float(labels[ids[0]]["before"])))
    return out


def validate_question_rows(pool: QuestionPool, rows: list[dict]) -> None:
    """One question's completion rows: exact coverage, one anchor, one context, one video."""
    anchor, rest = anchor_and_rest(pool)
    ids = [pool.candidates[i].id for i in rest]
    if sorted(r["candidate_id"] for r in rows) != sorted(ids):
        raise ValueError(f"{pool.qa_id}: completion labels do not cover exactly the non-anchor candidates")
    if any(r["anchor_ids"] != [pool.candidates[j].id for j in anchor] for r in rows):
        raise ValueError(f"{pool.qa_id}: labels were produced with a different anchor")
    if len({r["before"] for r in rows}) != 1:
        raise ValueError(f"{pool.qa_id}: rows disagree on the three-frame outcome of the shared anchor")
    if any(r["video_id"] != pool.video_id for r in rows):
        raise ValueError(f"{pool.qa_id}: video id does not match the pool")


def fit_ridge(questions: list[CompletionQuestion], ridge: float | None) -> np.ndarray:
    """Pairwise residual ridge; each question has equal weight; None = baseline fallback.

    Target for a pair (a, b): (y_a - y_b) - (base_a - base_b), over ALL
    within-question pairs. For a correctness tie the target is -(base_a - base_b),
    so tied pairs pull the residual toward flattening the baseline order; the
    ridge and the fallback limit this. (A variant trained only on unequal pairs
    was checked on the same data: 73/197 vs 74/197, so ties are not the bottleneck.)
    """
    d = questions[0].z.shape[1] if questions else len(completion_feature_names())
    if ridge is None:
        return np.zeros(d)
    gram, rhs, n = np.zeros((d, d)), np.zeros(d), 0
    for q in questions:
        i, j = np.triu_indices(len(q.correct), 1)
        if not len(i):
            continue
        dx = q.z[i] - q.z[j]
        dy = (q.correct[i] - q.correct[j]) - (q.base[i] - q.base[j])
        gram += dx.T @ dx / len(i)
        rhs += dx.T @ dy / len(i)
        n += 1
    if not n:
        return np.zeros(d)
    return np.linalg.solve(gram / n + ridge * np.eye(d), rhs / n)


def pick(q: CompletionQuestion, w: np.ndarray) -> int:
    """Index (in rest order) of the chosen fourth frame; ties go to the baseline order."""
    s = q.base + q.z @ w
    return int(min(range(len(s)), key=lambda i: (-s[i], i)))


def _video_mean(questions, w) -> float:
    by: dict[str, list[float]] = {}
    for q in questions:
        by.setdefault(q.video_id, []).append(q.correct[pick(q, w)])
    return float(np.mean([np.mean(v) for v in by.values()]))


def choose_ridge(train: list[CompletionQuestion]) -> tuple[float | None, dict]:
    """Inner leave-one-video-out on the outer training videos only.

    Objective: equal-video mean four-frame correctness of the picked completion.
    Ties prefer the baseline fallback, then stronger regularisation.
    """
    videos = sorted({q.video_id for q in train})
    grid = (None,) + tuple(sorted(RIDGES, reverse=True))
    if len(videos) < 2:
        return None, {"reason": "too_few_training_videos"}
    means = {}
    for ridge in grid:
        vals = []
        for held in videos:
            w = fit_ridge([q for q in train if q.video_id != held], ridge)
            vals.append(_video_mean([q for q in train if q.video_id == held], w))
        means[str(ridge)] = float(np.mean(vals))
    best = grid[0]
    for ridge in grid[1:]:
        if means[str(ridge)] > means[str(best)] + 1e-12:
            best = ridge
    return best, {"inner_mean_by_ridge": means}


def nested_cv(questions: list[CompletionQuestion]) -> tuple[dict[str, int], list[dict]]:
    """Outer leave-one-video-out; returns the picked index per question and fold audits."""
    picks, folds = {}, []
    for held in sorted({q.video_id for q in questions}):
        train = [q for q in questions if q.video_id != held]
        ridge, audit = choose_ridge(train)
        w = fit_ridge(train, ridge)
        for q in questions:
            if q.video_id == held:
                picks[q.qa_id] = pick(q, w)
        folds.append({"held_video": held, "ridge": ridge, "weights": w.tolist(), **audit})
    return picks, folds
