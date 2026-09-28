"""How often does a dataset's answer live in its speech? (text only, no video)

For each sampled question we build the usual clean / targeted / control
transcript triple (damage.build_triple) and ask the FROZEN answerer with the
transcript excerpt only - no frames, so no video download is needed. We count
the "important-speech failure" stratum used by the primary endpoint:

    transcript-only is right on CLEAN and wrong on TARGETED damage.

The same code runs on LongVideoBench (our local copy) and on Video-MME
(questions + subtitles from lmms-lab/Video-MME), so the two rates are
directly comparable. This is a feasibility screen, not a result: the targeted
spans for Video-MME come from the noisy "answer_overlap" fallback because the
dataset has no evidence annotations.

    python scripts/speech_stratum_check.py --dataset videomme --n 50 --out runs/stratum/videomme50.json
    python scripts/speech_stratum_check.py --dataset lvb --n 50 --out runs/stratum/lvb50.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from videoqa.answerer import AnswerRequest, answer_quality, make_answerer  # noqa: E402
from videoqa.config import load_config  # noqa: E402
from videoqa.damage import build_triple  # noqa: E402
from videoqa.packing import pack_excerpt  # noqa: E402
from videoqa.retrieval import bm25_rank, build_windows  # noqa: E402
from videoqa.schemas import QAItem, TranscriptCondition  # noqa: E402
from videoqa.transcript import build_units, load_transcript, parse_srt_or_vtt  # noqa: E402

C = TranscriptCondition


def _order(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def load_videomme(raw: Path) -> list[tuple[QAItem, object]]:
    """Video-MME questions that have a subtitle file. Official split is 'test'
    (the benchmark has no other); our experiment split is decided later."""
    import pyarrow.parquet as pq  # noqa: PLC0415 - only this adapter needs it

    rows = pq.read_table(raw / "videomme" / "test-00000-of-00001.parquet").to_pylist()
    zf = zipfile.ZipFile(raw / "subtitle.zip")
    subs = {Path(n).stem: n for n in zf.namelist() if n.endswith(".srt")}
    cache: dict[str, object] = {}
    out = []
    for r in rows:
        vid = r["videoID"]
        if vid not in subs:
            continue
        if vid not in cache:
            cache[vid] = parse_srt_or_vtt(zf.read(subs[vid]).decode("utf-8", "replace"), vid,
                                          source=f"subtitle:videomme/{vid}.srt")
        opts = [re.sub(r"^[A-Z]\.\s*", "", o).strip() for o in r["options"]]
        gi = "ABCDEFGH".index(r["answer"].strip()[0])
        qa = QAItem(qa_id=r["question_id"], video_id=vid, question=r["question"], gold_answer=opts[gi],
                    options=opts, gold_option_index=gi, source_dataset="Video-MME",
                    official_split="test", provenance_and_license="lmms-lab/Video-MME (research use)")
        out.append((qa, cache[vid]))
    return out


def load_lvb(root: Path) -> list[tuple[QAItem, object]]:
    from videoqa.datasets import _qa_from_json  # noqa: PLC0415

    out, cache = [], {}
    for line in (root / "qa.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        qa = _qa_from_json(json.loads(line))
        p = root / "transcripts" / f"{qa.video_id}.json"
        if not p.exists():
            continue
        if qa.video_id not in cache:
            cache[qa.video_id] = load_transcript(p, qa.video_id)
        out.append((qa, cache[qa.video_id]))
    return out


def load_eduvidqa(root: Path) -> list[tuple[QAItem, object]]:
    """Official-train EduVidQA only (test stays locked). Built by
    scripts/fetch_eduvidqa.py; the evidence interval is the question's own
    stated timestamp window, so using it for the excerpt is not gold leakage."""
    from videoqa.datasets import _qa_from_json  # noqa: PLC0415

    out, cache = [], {}
    for line in (root / "qa.jsonl").read_text(encoding="utf-8").splitlines():
        qa = _qa_from_json(json.loads(line))
        p = root / "transcripts" / f"{qa.video_id}.json"
        if qa.official_split != "train" or not p.exists():
            continue
        if qa.video_id not in cache:
            cache[qa.video_id] = load_transcript(p, qa.video_id)
        out.append((qa, cache[qa.video_id]))
    return out


def load_tvqa(raw: Path) -> list[tuple[QAItem, object]]:
    """TVQA validation questions (Vision-CAIR/TVQA-Long mirror) with the
    per-clip subtitles; ``ts`` is TVQA's human-annotated evidence moment."""
    from videoqa.transcript import parse_json_segments  # noqa: PLC0415

    ann = raw / "tvqa-long-annotations"
    subs = {x["vid_name"]: x["sub"] for x in json.loads((ann / "tvqa_preprocessed_subtitles.json").read_text(encoding="utf-8"))}
    out, cache = [], {}
    for show, seasons in json.loads((ann / "tvqa_val_edited.json").read_text(encoding="utf-8")).items():
        for season in seasons.values():
            for ep in season.values():
                for q in ep["questions"]:
                    vid = q["vid_name"]
                    key = vid if vid in subs else next((k for k in (f"{p}_{vid}" for p in ("bbt", "castle", "friends", "grey", "house", "met")) if k in subs), None)
                    if key is None:
                        continue
                    if key not in cache:
                        cache[key] = parse_json_segments(subs[key], key, source="tvqa_subtitles")
                    t0, t1 = (float(x) for x in q["ts"].split("-"))
                    opts = [q[f"a{i}"] for i in range(5)]
                    out.append((QAItem(qa_id=str(q["qid"]), video_id=key, question=q["q"],
                                       gold_answer=opts[q["answer_idx"]], options=opts,
                                       gold_option_index=q["answer_idx"], evidence_intervals_s=[(t0, t1)],
                                       source_dataset="TVQA", official_split="validation",
                                       provenance_and_license="TVQA (via Vision-CAIR/TVQA-Long)"), cache[key]))
    return out


def excerpt_for(qa: QAItem, transcript, cfg: dict):
    """The same text-only excerpt transcript_only gets (acquisition.prepare_text
    + pipeline.answer_with), with duration taken from the last subtitle."""
    tcfg, rcfg = cfg["transcript"], cfg["retrieval"]
    duration = max((s.end_s for s in transcript.segments), default=0.0) + 1.0
    units = build_units(transcript.segments, tcfg["unit_seconds"], tcfg["min_unit_seconds"])
    ranked = bm25_rank(qa.question + " " + " ".join(qa.options or []), units, rcfg["bm25_top_k"])
    windows = build_windows(ranked, units, rcfg["max_windows"], rcfg["neighbour_expansion"], duration)
    # Free-form questions that state a time ("At 3:54, ...") also get the
    # speech around that time, as a user-given anchor (EduVidQA only).
    anchor = []
    if qa.options is None and qa.evidence_intervals_s:
        a0, a1 = qa.evidence_intervals_s[0]
        anchor = [x.id for x in transcript.segments if x.end_s > a0 and x.start_s < a1]
    return pack_excerpt(qa.question, transcript.segments, windows, {u.id: u.segment_ids for u in units},
                        max_words=120, extra_segment_ids=anchor)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("videomme", "lvb", "eduvidqa", "tvqa"), required=True)
    ap.add_argument("--raw", default=None, help="videomme: data/videomme_raw; lvb: data/longvideobench_full")
    ap.add_argument("--config", default="configs/gpu_12gb.yaml")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = load_config(a.config)
    raw = Path(a.raw or {"videomme": "data/videomme_raw", "lvb": "data/longvideobench_full",
                         "eduvidqa": "data/eduvidqa", "tvqa": "data/tvqa_long_raw"}[a.dataset])
    items = {"videomme": load_videomme, "lvb": load_lvb, "eduvidqa": load_eduvidqa, "tvqa": load_tvqa}[a.dataset](raw)
    items.sort(key=lambda x: _order(x[0].qa_id))
    answerer = make_answerer(cfg)

    drops, methods, per_q, seen = collections.Counter(), collections.Counter(), [], set()
    t0 = time.time()
    for qa, tr in items:
        if len(per_q) >= a.n:
            break
        if qa.video_id in seen:   # independent videos -> no clustering needed
            continue
        triple, why = build_triple(qa, tr, seed=0)
        if triple is None:
            drops[why] += 1
            continue
        seen.add(qa.video_id)
        methods[triple.relevance_method] += 1
        correct = {}
        for cond, t in triple.by_condition().items():
            ans, _ = answerer.answer(AnswerRequest(qa.question, qa.options, excerpt_for(qa, t, cfg), []))
            # MCQ: exact option; free-form: token F1 against the reference
            # (a screen only - the final study needs an audited rubric).
            correct[cond.value] = (ans.option_index == qa.gold_option_index if qa.options
                                   else round(answer_quality(ans, qa), 4))
        per_q.append({"qa_id": qa.qa_id, "video_id": qa.video_id, "relevance": triple.relevance_method,
                      "targeted_tokens": triple.records[C.TARGETED_DAMAGE].changed_tokens,
                      "control_tokens": triple.records[C.CONTROL_DAMAGE].changed_tokens,
                      **{f"correct_{k}": v for k, v in correct.items()}})
        print(f"[{len(per_q)}/{a.n}] {qa.qa_id} {correct} {time.time() - t0:.0f}s", flush=True)

    n = len(per_q)
    k = lambda f: sum(1 for r in per_q if f(r))  # noqa: E731
    clean, tgt, ctl = (f"correct_{c.value}" for c in (C.CLEAN, C.TARGETED_DAMAGE, C.CONTROL_DAMAGE))
    summary = {
        "dataset": a.dataset, "n": n, "attempted": n + sum(drops.values()), "drops": dict(drops),
        "relevance_methods": dict(methods),
        "text_only_acc_clean": k(lambda r: r[clean]) / max(n, 1),
        "text_only_acc_targeted": k(lambda r: r[tgt]) / max(n, 1),
        "text_only_acc_control": k(lambda r: r[ctl]) / max(n, 1),
        "stratum_speech_failure": k(lambda r: r[clean] and not r[tgt]),
        "stratum_rate": k(lambda r: r[clean] and not r[tgt]) / max(n, 1),
        "control_failure": k(lambda r: r[clean] and not r[ctl]),
        "seconds": round(time.time() - t0, 1),
        "note": "text-only screen; targeted spans for Video-MME come from answer_overlap (noisy)",
    }
    if items and items[0][0].options is None:            # free-form: F1 drops
        import numpy as np  # noqa: PLC0415

        f = {c: np.array([r[f"correct_{c.value}"] for r in per_q]) for c in (C.CLEAN, C.TARGETED_DAMAGE, C.CONTROL_DAMAGE)}
        for k in ("text_only_acc_clean", "text_only_acc_targeted", "text_only_acc_control"):
            summary.pop(k)                                # accuracy is undefined for free-form
        dt, dc = f[C.CLEAN] - f[C.TARGETED_DAMAGE], f[C.CLEAN] - f[C.CONTROL_DAMAGE]
        summary.update({
            "f1_clean": float(f[C.CLEAN].mean()), "f1_targeted": float(f[C.TARGETED_DAMAGE].mean()),
            "f1_control": float(f[C.CONTROL_DAMAGE].mean()),
            "mean_drop_targeted": float(dt.mean()), "mean_drop_control": float(dc.mean()),
            "stratum_speech_failure": int((dt >= 0.10).sum()), "stratum_rate": float((dt >= 0.10).mean()),
            "control_failure": int((dc >= 0.10).sum()),
            "note": "free-form screen: token F1 vs reference, speech failure = F1 drop >= 0.10 under targeted",
        })
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"summary": summary, "questions": per_q}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
