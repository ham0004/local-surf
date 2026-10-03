"""Candidate generation: shared windows -> two parallel proposal paths -> merged pool.

    transcript + question
          │  BM25 over ~20 s units (v1 retrieval) -> top windows
          ├───────────────────────────────┐
          ▼                               ▼
    PATH A: hot moments              PATH B: sparse visual scan
    Head A scores each segment       frames every `scan_step_s` inside the windows,
    inside the windows; the top      one representative per stable slide/shot
    `n_path_a` segments by visual    (perceptual hash), ranked by MobileCLIP
    utility give one frame each      question-frame similarity; keep top `n_path_b`
          │                               │
          └───────────────┬───────────────┘
                          ▼
        merge + de-duplicate (same moment or near-identical pixels)
                          ▼
        QuestionPool: fixed retained transcript + candidate frames with cached features

The retained transcript is v1's BM25 excerpt and is FIXED for every selection
arm, so the pilot compares frame selection only.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..frames import decode_at, hamming_hex, pixel_digest, probe
from ..packing import pack_excerpt
from ..retrieval import bm25_rank, build_windows
from ..schemas import TranscriptSegment
from ..transcript import build_units
from .records import FrameCandidate, QuestionPool


@dataclasses.dataclass
class PoolConfig:
    unit_seconds: float = 20.0
    max_windows: int = 4
    neighbour_expansion: int = 1
    excerpt_words: int = 120
    n_path_a: int = 6            # hot-moment frames proposed by Path A
    n_path_b: int = 6            # visual-scan frames proposed by Path B
    scan_step_s: float = 3.0     # Path B sampling step inside windows
    scan_cap: int = 60           # max frames Path B decodes per question
    stable_hamming: int = 6      # phash distance that starts a new "stable segment"
    dup_hamming: int = 3         # near-identical pixels -> same candidate
    dup_seconds: float = 1.0     # same moment -> same candidate
    max_side: int = 640          # frame size (the answerer's size, so pixels are shared)


# ---------------------------------------------------------------------------
# Building one question's pool
# ---------------------------------------------------------------------------


def build_pool(qa, transcript, video_path: str, encoders, head_a, cfg: PoolConfig = PoolConfig()) -> QuestionPool:
    """Run both paths for one question and return its fixed evidence pool."""
    t = {}
    tic = time.perf_counter()
    segs: list[TranscriptSegment] = transcript.segments
    duration = probe(video_path).duration_s

    # -- shared upstream: BM25 windows + retained transcript ------------------
    units = build_units(segs, cfg.unit_seconds, 5.0)
    query = qa.question + " " + " ".join(qa.options or [])
    windows = build_windows(bm25_rank(query, units, 64), units, cfg.max_windows, cfg.neighbour_expansion, duration)
    retained = pack_excerpt(qa.question, segs, windows, {u.id: u.segment_ids for u in units},
                            max_words=cfg.excerpt_words)
    t["retrieval"] = time.perf_counter() - tic

    # -- Path A: Head A scores segments inside the windows --------------------
    tic = time.perf_counter()
    in_win = [s for s in segs if any(s.end_s > w.start_s and s.start_s < w.end_s for w in windows)]
    ctx = [_with_neighbours(segs, s) for s in in_win]
    scores = head_a.score(qa.question, ctx) if in_win else np.zeros((0, 2))
    order = np.argsort(-scores[:, 1]) if len(in_win) else []
    path_a = [(in_win[i], scores[i]) for i in order[: cfg.n_path_a]]
    a_times = [max(0.0, min(duration - 0.05, s.end_s - 0.3)) for s, _ in path_a]   # end of line: board fullest
    t["head_a"] = time.perf_counter() - tic

    # -- Path B: sparse scan inside the windows -------------------------------
    scan_times = []
    for w in windows:
        x = w.start_s
        while x < w.end_s and len(scan_times) < cfg.scan_cap:
            scan_times.append(min(x, duration - 0.05))
            x += cfg.scan_step_s
    tic = time.perf_counter()
    decoded = decode_at(video_path, sorted(set(a_times + scan_times)), max_side=cfg.max_side,
                        video_id=qa.video_id).frames
    t["decode"] = time.perf_counter() - tic
    by_req = {round(f.requested_s, 3): f for f in decoded}
    a_pairs = [(sa, by_req[round(x, 3)]) for sa, x in zip(path_a, a_times, strict=True) if round(x, 3) in by_req]
    a_frames = [f for _, f in a_pairs]
    scan_frames = sorted((by_req[round(x, 3)] for x in scan_times if round(x, 3) in by_req),
                         key=lambda f: f.decoded_pts_s)
    reps = _stable_representatives(scan_frames, cfg.stable_hamming)

    # -- one MobileCLIP pass per unique frame ---------------------------------
    tic = time.perf_counter()
    uniq = {f.digest: f for f in a_frames + reps}
    embs = dict(zip(uniq, encoders.embed_images([f.image for f in uniq.values()]), strict=True))
    q_emb = encoders.embed_texts([qa.question])[0]
    t["clip"] = time.perf_counter() - tic
    reps_ranked = sorted(reps, key=lambda f: -float(embs[f.digest] @ q_emb))[: cfg.n_path_b]

    # -- merge + de-duplicate ---------------------------------------------------
    cands: list[FrameCandidate] = []
    for (_seg, _sc), f in a_pairs:
        if any(c.digest == f.digest for c in cands):          # two hot lines can map to the same frame
            continue
        cands.append(_cand(qa, len(cands), f, ("A",), embs, q_emb, None))
    for f in reps_ranked:
        dup = next((c for c in cands if abs(c.time_s - f.decoded_pts_s) <= cfg.dup_seconds
                    or hamming_hex(c.phash, f.phash) <= cfg.dup_hamming), None)
        if dup is not None:
            dup.paths = tuple(sorted(set(dup.paths) | {"B"}))
        else:
            cands.append(_cand(qa, len(cands), f, ("B",), embs, q_emb, None))
    # Every candidate gets the Head A scores of its nearest segment (Path B frames too).
    _attach_head_a(cands, in_win, scores)

    # -- per-candidate text features: OCR + nearby transcript -----------------
    tic = time.perf_counter()
    for c in cands:
        c.ocr_text = encoders.ocr(c.image)
        c.near_text = " ".join(s.text for s in segs if abs(0.5 * (s.start_s + s.end_s) - c.time_s) <= 15.0)[:600]
    t["ocr"] = time.perf_counter() - tic
    tic = time.perf_counter()
    if cands:
        ocr_e = encoders.embed_texts([c.ocr_text for c in cands])
        near_e = encoders.embed_texts([c.near_text for c in cands])
        for c, o, n in zip(cands, ocr_e, near_e, strict=True):
            c.ocr_emb, c.near_emb = o, n
    t["text_emb"] = time.perf_counter() - tic

    return QuestionPool(qa_id=qa.qa_id, video_id=qa.video_id, question=qa.question, options=list(qa.options),
                        gold_option_index=qa.gold_option_index, transcript=retained, candidates=cands,
                        duration_s=duration, question_emb=q_emb, timings=t)


def _with_neighbours(segs, s) -> str:
    """Segment text with one neighbouring line on each side (Head A's input context)."""
    i = segs.index(s)
    return " ".join(x.text for x in segs[max(0, i - 1): i + 2])


def _stable_representatives(frames, threshold: int):
    """One frame per run of visually similar frames: the LAST frame of each run
    (on a lecture board, the most complete state of what was written)."""
    reps, run_start, last = [], None, None
    for f in frames:
        if run_start is None:
            run_start = f
        elif hamming_hex(run_start.phash, f.phash) > threshold:
            reps.append(last)
            run_start = f
        last = f
    if last is not None:
        reps.append(last)
    return reps


def _cand(qa, k, frame, paths, embs, q_emb, head_a_scores) -> FrameCandidate:
    e = embs[frame.digest]
    return FrameCandidate(id=f"{qa.qa_id}:f{k:02d}", time_s=round(frame.decoded_pts_s, 3), paths=paths,
                          digest=frame.digest, phash=frame.phash, clip_sim=float(e @ q_emb),
                          emb=e, image=frame.image)


def _attach_head_a(cands, segs, scores) -> None:
    for c in cands:
        if not len(segs):
            continue
        i = int(np.argmin([abs(0.5 * (s.start_s + s.end_s) - c.time_s) for s in segs]))
        c.head_a_text, c.head_a_visual = float(scores[i, 0]), float(scores[i, 1])


# ---------------------------------------------------------------------------
# Persistence: pools are built once, then reused by every stage
# ---------------------------------------------------------------------------


def save_pool(pool: QuestionPool, out_dir: str | Path) -> None:
    """JSON metadata + one .npz of embeddings + lossless PNG frames (so pixel digests stay exact)."""
    d = Path(out_dir) / _safe(pool.qa_id)
    d.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in dataclasses.asdict(pool).items() if k not in ("candidates", "transcript", "question_emb")}
    meta["transcript"] = [dataclasses.asdict(s) for s in pool.transcript]
    meta["candidates"] = [{k: v for k, v in dataclasses.asdict(c).items()
                           if k not in ("emb", "ocr_emb", "near_emb", "image")} for c in pool.candidates]
    (d / "pool.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    np.savez(d / "emb.npz", question=pool.question_emb,
             **{f"{c.id}|{k}": getattr(c, k) for c in pool.candidates for k in ("emb", "ocr_emb", "near_emb")})
    for c in pool.candidates:
        c.image.save(d / f"{_safe(c.id)}.png")


def load_pool(out_dir: str | Path, qa_id: str) -> QuestionPool:
    d = Path(out_dir) / _safe(qa_id)
    meta = json.loads((d / "pool.json").read_text(encoding="utf-8"))
    emb = np.load(d / "emb.npz")
    cands = []
    for cm in meta["candidates"]:
        img = Image.open(d / f"{_safe(cm['id'])}.png").convert("RGB")
        cm["paths"] = tuple(cm["paths"])
        c = FrameCandidate(**cm)
        c.image = img
        c.digest = pixel_digest(img)          # identical to the original (PNG is lossless)
        c.emb, c.ocr_emb, c.near_emb = (emb[f"{c.id}|{k}"] for k in ("emb", "ocr_emb", "near_emb"))
        cands.append(c)
    meta["transcript"] = [TranscriptSegment(**s) for s in meta["transcript"]]
    meta["candidates"] = cands
    pool = QuestionPool(**meta)
    pool.question_emb = emb["question"]
    return pool


def _safe(s: str) -> str:
    return s.replace(":", "_").replace("/", "_")
