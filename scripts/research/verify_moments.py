"""Check mined lecture questions with the FROZEN answerer and write qa.jsonl.

For every parsed item from scripts/mine_moments.py:

1. Grounding checks (no model):
   - speech-bearing relations (both / speech_only / conflict): the quoted
     speech_evidence must occur in the window's captions (>= 60% of its
     content words), and the correct option must share a content word with
     the speech (or be a number that occurs in it);
   - the four options must be distinct.
2. Answerer checks (Qwen3-VL-2B, the same frozen answerer as the study):
   none   : question + options only (prior; correct = too easy, flagged)
   speech : + the window's captions, no frame
   frame  : + the end-of-window frame, no captions
   both   : + captions + frame
3. A cleaned caption file per lecture (transcripts/<id>.json) is written so
   the assay loads speech without YouTube's rolling-caption duplication.

Output qa.jsonl uses our QAItem schema, evidence_intervals_s = the window,
evidence_type from the relation, and provenance marking it SYNTHETIC.

    python scripts/research/verify_moments.py --lectures data/mit_lectures --mined runs/mined/pilot
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mine_moments import clean_captions  # noqa: E402
from videoqa.answerer import AnswerRequest, make_answerer  # noqa: E402
from videoqa.config import load_config  # noqa: E402
from videoqa.damage import _content_tokens  # noqa: E402
from videoqa.frames import decode_at  # noqa: E402
from videoqa.schemas import QAItem, TranscriptSegment  # noqa: E402
from videoqa.transcript import load_transcript  # noqa: E402

EVIDENCE_TYPE = {"both": "spoken_and_visual", "speech_only": "spoken_answer", "visual_only": "visual_support",
                 "conflict": "speech_visual_conflict"}
SPEECH_BEARING = ("both", "speech_only", "conflict")


def grounded(item: dict, speech: str) -> tuple[bool, str]:
    opts = [str(o).strip() for o in item["options"]]
    if len({o.lower() for o in opts}) < 4:
        return False, "duplicate_options"
    if item["relation"] in SPEECH_BEARING:
        sp = _content_tokens(speech)
        ev = _content_tokens(item.get("speech_evidence") or "")
        if not ev or len(ev & sp) / len(ev) < 0.6:
            return False, "speech_quote_not_found"
        ans = opts[item["answer_index"]]
        if item["relation"] != "conflict" and not (_content_tokens(ans) & sp):
            return False, "answer_not_in_speech"
    return True, "ok"


def split_of(video_id: str) -> str:
    """By lecture: ~70% train, 15% dev, 15% test (fixed hash)."""
    h = int(hashlib.sha256(video_id.encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else "dev" if h < 85 else "test"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lectures", default="data/mit_lectures")
    ap.add_argument("--mined", required=True)
    ap.add_argument("--config", default="configs/gpu_12gb.yaml")
    ap.add_argument("--qa-out", default=None, help="output qa file (default: <lectures>/qa.jsonl)")
    ap.add_argument("--drop-prior", action="store_true",
                    help="dataset rule: drop questions the answerer gets right with NO evidence (no transcript, "
                         "no frames); they carry no evidence-utility signal. All items go to <qa-out>_all.jsonl")
    a = ap.parse_args()
    root, mined = Path(a.lectures), Path(a.mined)
    cfg = load_config(a.config)
    answerer = make_answerer(cfg)
    max_side = cfg["answerer"].get("frame_max_side")

    # Cleaned captions, one JSON per lecture (preferred over .vtt by datasets.py).
    lines_by_vid = {}
    for vtt in (root / "transcripts").glob("*.vtt"):
        vid = vtt.stem
        lines = clean_captions(load_transcript(vtt, vid))
        lines_by_vid[vid] = lines
        (root / "transcripts" / f"{vid}.json").write_text(
            json.dumps([{"start": s, "end": e, "text": t} for s, e, t in lines]), encoding="utf-8")

    stats, items, audit = collections.Counter(), [], []
    for rec in map(json.loads, (mined / "raw.jsonl").read_text(encoding="utf-8").splitlines()):
        p = rec["parsed"]
        if p is None:
            stats["parse_fail"] += 1
            continue
        stats[f"claimed:{p['relation']}"] += 1
        if p["relation"] == "none":
            continue
        # The relation used downstream is MEASURED by the generator answering
        # from one source at a time (mine_moments.check_answer), not claimed.
        chk = rec.get("generator_checks", {})
        sp_ok = chk.get("speech_only") == p["answer_index"]
        fr_ok = chk.get("frame_only") == p["answer_index"]
        claimed = p["relation"]
        if claimed == "conflict":
            measured = "conflict" if fr_ok else None
        else:
            measured = {(True, True): "both", (True, False): "speech_only",
                        (False, True): "visual_only"}.get((sp_ok, fr_ok))
        if measured is None:
            stats["reject:not_answerable_from_either_source"] += 1
            continue
        p = {**p, "relation": measured, "claimed_relation": claimed}
        ok, why = grounded(p, rec["speech"])
        if not ok:
            stats[f"reject:{why}"] += 1
            continue
        vid, t0, t1 = rec["video_id"], rec["t0"], rec["t1"]
        opts = [str(o).strip() for o in p["options"]]
        segs = [TranscriptSegment(f"w{i:03d}", s, e, t) for i, (s, e, t) in enumerate(lines_by_vid[vid])
                if s < t1 and e > t0]
        frame = decode_at(root / "videos" / f"{vid}.mp4", [t1 - 0.5], max_side=max_side, video_id=vid).frames
        res = {}
        for cond, ex, fr in (("none", [], []), ("speech", segs, []), ("frame", [], frame), ("both", segs, frame)):
            ans, _ = answerer.answer(AnswerRequest(p["question"], opts, ex, fr))
            res[cond] = ans.option_index == p["answer_index"]
        qa = QAItem(qa_id=f"mit:{vid}:{int(t0)}", video_id=vid, question=p["question"], gold_answer=opts[p["answer_index"]],
                    options=opts, gold_option_index=p["answer_index"], evidence_intervals_s=[(t0, t1)],
                    source_dataset="MIT-OCW-mined (SYNTHETIC)", official_split="none",
                    experiment_split=split_of(vid), source_split=split_of(vid), evidence_type=EVIDENCE_TYPE[p["relation"]],
                    provenance_and_license="SYNTHETIC question generated by Qwen3-VL-4B from CC BY-NC-SA 4.0 "
                                           "MIT OpenCourseWare (Gilbert Strang); unaudited")
        items.append(qa)
        row = {"qa_id": qa.qa_id, "video_id": vid, "t0": t0, "t1": t1, "relation": p["relation"],
               "question": p["question"], "options": opts, "answer": opts[p["answer_index"]],
               "claimed_relation": p["claimed_relation"], "fact": p.get("fact"), "speech_evidence": p.get("speech_evidence"),
               "visual_evidence": p.get("visual_evidence"), **{f"correct_{k}": v for k, v in res.items()}}
        audit.append(row)
        stats[f"kept:{p['relation']}"] += 1
        for k, v in res.items():
            stats[f"correct_{k}:{p['relation']}"] += int(v)
        print(f"{qa.qa_id} {p['relation']:11s} {res}", flush=True)

    # Dataset rule (declared before results): optionally drop questions the answerer
    # gets right with NO evidence; every item still goes to <qa-out>_all.jsonl.
    qa_out = Path(a.qa_out) if a.qa_out else root / "qa.jsonl"
    prior_ok = {r["qa_id"] for r in audit if r["correct_none"]}
    kept_items = [qa for qa in items if not (a.drop_prior and qa.qa_id in prior_ok)]
    stats["dropped_prior_answerable"] = len(items) - len(kept_items)
    for path, rows in ((qa_out, kept_items), (qa_out.with_name(qa_out.stem + "_all.jsonl"), items)):
        with open(path, "w", encoding="utf-8") as fh:
            for qa in rows:
                fh.write(json.dumps(dataclasses.asdict(qa)) + "\n")
    (mined / "audit.jsonl").write_text("\n".join(json.dumps(r) for r in audit), encoding="utf-8")
    (mined / "verify_stats.json").write_text(json.dumps(dict(sorted(stats.items())), indent=1), encoding="utf-8")
    print(json.dumps(dict(sorted(stats.items())), indent=1))


if __name__ == "__main__":
    main()
