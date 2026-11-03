"""Third supplement to the lecture-study method freeze (auditor P103): the secondary answer-model evaluation.
Written after the primary hit outcomes were observed (reports/m3av/hits_test.json) and before any answer-model
call on test. Additive: the original freeze and supplements 1-2 stay unchanged and are verified by hash; no model,
pool, selection rule, cohort or primary result changes.

Records: the chain (freeze, supplements 1-2, test input and artifact manifests, the primary report); the
condition roster with exact inputs and text policies; the frozen answerer, prompt and processor settings; the
answer, reuse, failure, ledger, completeness and seal rules; the analysis; the commands; hashes of the QA code and
of the code it runs that the earlier records did not cover; the dev check.

    python scripts/m3av_freeze_supplement3.py write      # docs/v3/m3av_freeze_supplement3.json
    python scripts/m3av_freeze_supplement3.py verify     # whole chain + test inputs, artifacts, primary report, QA code
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import m3av_freeze_supplement as s1  # noqa: E402
import m3av_freeze_supplement2 as s2  # noqa: E402

OUT = Path("docs/v3/m3av_freeze_supplement3.json")
S2_SHA = {"lf": "25826f95204154d9ade82b2e57f7d04ff4aee6d2fcdb2d5edde7a9284f9ee23c",
          "bytes": "543da6299d1da96f2572f398b7fcde7cdda6f8e5a34ad29f7bb6e05b924576f2"}
CHAIN = {"docs/v3/m3av_manifest_test_inputs.json": None, "docs/v3/m3av_manifest_test_artifacts.json": None,
         "reports/m3av/hits_test.json": None, "reports/m3av/hits_test.md": None}
NEW_FILES = ["scripts/m3av_qa.py", "scripts/run_m3av_qa_test.sh", "scripts/m3av_freeze_supplement3.py"]
# code on the QA path that the original freeze did not hash
QA_PATH = ["src/videoqa/answerer_hf.py", "src/videoqa/v2/teacher.py", "src/videoqa/config.py", "src/videoqa/retrieval.py",
           "src/videoqa/schemas.py", "src/videoqa/frames.py", "src/videoqa/datasets.py", "configs/gpu_12gb.yaml",
           "scripts/m3av_freeze.py"]
DEV_CHECK = Path("reports/m3av/qa_dev_check.json")


def lf_and_bytes(path: Path) -> dict:
    return {"lf": s1.sha(path), "bytes": hashlib.sha256(path.read_bytes()).hexdigest()}


def processor_settings() -> dict:
    from transformers import AutoProcessor  # noqa: PLC0415

    from videoqa.config import load_config  # noqa: PLC0415

    a = load_config("configs/gpu_16gb_4b.yaml")["answerer"]
    pr = AutoProcessor.from_pretrained(a["model_id"], revision=a["revision"], cache_dir="cache/hf/hub")
    ip = pr.image_processor.to_dict()
    return {"image_processor": {k: ip.get(k) for k in ("image_processor_type", "size", "min_pixels", "max_pixels",
                                                       "patch_size", "temporal_patch_size", "merge_size", "do_resize",
                                                       "resample", "image_mean", "image_std")},
            "tokenizer": type(pr.tokenizer).__name__,
            "chat_template_sha256": hashlib.sha256((pr.chat_template or "").encode()).hexdigest()}


def roster() -> dict:
    import m3av_qa  # noqa: PLC0415

    text = {"none": "no transcript",
            "bm25": "the pool's BM25 window excerpt (question + options as query; packed at 300 words when the pool "
                    "was built), in time order, cut before the first line that would pass the allowance (the first "
                    "line is always kept)",
            "frames+hybrid": "half the allowance: transcript lines nearest the shown frames (distance 0 inside a "
                             "line, within 15 s, stable order); then lines ranked by reciprocal-rank fusion "
                             "(1/(60 + rank)) of BM25 (question + options) and zero-shot MiniLM-L6 relevance (line "
                             "with one neighbour each side), each added with one neighbour each side, until the "
                             "allowance; shown in time order",
            "bm25full": "every line of the whole transcript ranked by BM25 (question + options, stable on ties), each "
                        "added with one neighbour each side, until the allowance; shown in time order"}
    rule = ("a line is added only if the running word count stays within the allowance, except the first line; the "
            "fill stops at the first line that does not fit (words = whitespace tokens)")
    return {c: {"pool": p, "selector": s, "k": m3av_qa.K, "transcript_policy": tp, "words_allowance": w,
                "transcript": text[tp], "frames": "none" if c in m3av_qa.CONTROLS else
                "the stored selection, at most 4, underfilled sets unchanged, shown in time order"}
            for c, (p, s, tp, w) in m3av_qa.CONDITIONS.items()} | {"_fill_rule": rule}


def stage_write(_a) -> None:
    import m3av_qa  # noqa: PLC0415

    from videoqa.answerer import SYSTEM_PROMPT, build_prompt_text  # noqa: PLC0415
    from videoqa.answerer_hf import HFVLMAnswerer  # noqa: PLC0415
    from videoqa.config import load_config  # noqa: PLC0415
    from videoqa.v2.teacher import answerer_identity  # noqa: PLC0415

    if OUT.exists():
        raise SystemExit(f"{OUT} exists")
    if Path("runs/m3av/qa/test").exists():
        raise SystemExit("test QA outputs exist")
    if subprocess.run(["git", "status", "--porcelain", "--", *NEW_FILES, *QA_PATH, str(DEV_CHECK)],
                      capture_output=True, text=True, check=True).stdout.strip():
        raise SystemExit("commit the code and the dev check first")
    cfg = load_config("configs/gpu_16gb_4b.yaml")
    rec = {"supplements": {"freeze": {"record": s1.FREEZE.as_posix(), "sha256": s1.FREEZE_SHA},
                           "supplement_1": {"record": s1.OUT.as_posix(), "sha256_lf": s2.S1_SHA},
                           "supplement_2": {"record": s2.OUT.as_posix(), "sha256": S2_SHA}},
           "chain": {f: lf_and_bytes(Path(f)) for f in CHAIN},
           "dated": "written after the primary hit outcomes were observed (reports/m3av/hits_test.json, 2026-11-02 "
                    "21:09 UTC) and before any answer-model call on test",
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
           "why": "auditor P103: exact execution, inputs, accounting and analysis of the secondary answer-model "
                  "evaluation declared in docs/v3/m3av_protocol.md",
           "python": platform.python_version(),
           "packages": {p: importlib.metadata.version(p) for p in s1.PACKAGES},
           "roster": roster(),
           "logical_records": f"{len(m3av_qa.CONDITIONS)} conditions x 853 questions = {len(m3av_qa.CONDITIONS) * 853}",
           "answerer": {"identity": answerer_identity(cfg), "config": cfg["answerer"], "k": m3av_qa.K,
                        "k_note": "the config's max_frames is not used; every request holds at most 4 frames",
                        "system_prompt": SYSTEM_PROMPT,
                        "user_message": "for each frame in time order: text 'Frame at {t:.2f}s:' then the image; then "
                                        "the prompt text (build_prompt_text)",
                        "build_prompt_text_sha256": hashlib.sha256(inspect.getsource(build_prompt_text).encode()).hexdigest(),
                        "messages_sha256": hashlib.sha256(inspect.getsource(HFVLMAnswerer._messages).encode()).hexdigest(),
                        "processor": processor_settings(),
                        "frames": "the pool's lossless PNG frames (decoded from the 360p stream, at most 640 px on the "
                                  "long side); no further resizing before the processor",
                        "never_in_a_prompt": "the source slide image, evidence_text, the correct option, reference "
                                             "spans, reviewer decisions"},
           "rules": {"reuse": "an identical request key (answerer identity, prompt text, frame digests and labels) is "
                              "answered once per run; later conditions copy the reply and record reused_from. No "
                              "earlier cache is read",
                     "parse": "explicit abstention: option None, wrong; no letter and no abstention: most likely "
                              "letter, flagged forced",
                     "failure": "CUDA out of memory: a failed record, wrong, never retried; any other error stops the "
                                "run without a record (unattempted, missing); no fallback model, precision or "
                                "resolution",
                     "integrity": "a repeated (condition, question) row or two different replies to one key stop the "
                                  "run; recomputed frames must reproduce the stored primary and S-src hits",
                     "ledger": {"file": "runs/m3av/qa/test/ledger.json", "max_new_calls": m3av_qa.LIMITS["test"][0],
                                "max_answer_seconds": m3av_qa.LIMITS["test"][1],
                                "stop": "a clean stop at the limit is INCOMPLETE, never DONE"},
                     "resume": "rerun the driver; saved records are skipped",
                     "complete": "seal and report need all 12 conditions on all 853 questions and nothing else",
                     "seal": "docs/v3/m3av_qa_test_outputs.json: LF hashes of records and sessions before the report"},
           "analysis": {"accuracy": "per condition on all 853 questions (failed answers count as wrong), and on two "
                                    "masks common to all systems: without q_only-correct, without t_only-correct. A "
                                    "failed control is unknown and that question is left out of that mask; counts "
                                    "reported. The masks depend on Qwen3-VL-4B and are not verified visually "
                                    "necessary subsets",
                        "contrasts": [f"{x} - {y}" for x, y in m3av_qa.CONTRASTS],
                        "statistics": "question-weighted differences, paired bootstrap over lectures (10,000 draws, "
                                      "seed 20261101), 95% intervals, raw p uncorrected; descriptive; not part of the "
                                      "primary Holm family. S1t - S1 adds text only; S1t - S0 would change frames and "
                                      "text together and is not reported as a selection gain",
                        "resources": "per condition: frames, realized transcript words, prompt text tokens (system "
                                     "prompt and chat template included, generated tokens excluded), visual tokens, "
                                     "generated tokens, answer seconds, selection seconds, the pool's build seconds "
                                     "(retrieval, Head A, decode, MobileCLIP), reuse; per session: model load seconds, "
                                     "torch peak allocated and reserved GPU memory, GPU name and memory"},
           "commands": {"file": "scripts/run_m3av_qa_test.sh",
                        "text": Path("scripts/run_m3av_qa_test.sh").read_text(encoding="utf-8")},
           "code": {f: s1.sha(Path(f)) for f in NEW_FILES + QA_PATH},
           "dev_check": {"report": DEV_CHECK.as_posix(), "sha256_lf": s1.sha(DEV_CHECK),
                         **json.loads(DEV_CHECK.read_text(encoding="utf-8"))["summary"]}}
    OUT.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"{OUT.as_posix()}: sha256 {s1.sha(OUT)} (LF)")


def stage_verify(_a) -> None:
    from argparse import Namespace  # noqa: PLC0415

    for part in ("inputs", "artifacts"):            # freeze, supplements 1-2, models, code, test manifest
        s2.stage_verify(Namespace(split="test", part=part))
    rec = json.loads(OUT.read_text(encoding="utf-8"))
    bad = [] if lf_and_bytes(s2.OUT)["lf"] == S2_SHA["lf"] else ["supplement 2 changed"]
    bad += [f"chain {f}" for f, h in rec["chain"].items() if s1.sha(Path(f)) != h["lf"]]
    bad += [f"code {f}" for f, h in rec["code"].items() if s1.sha(Path(f)) != h]
    rec1 = json.loads(s1.OUT.read_text(encoding="utf-8"))
    bad += [f"package {p} {importlib.metadata.version(p)} (recorded {v})" for p, v in rec1["packages"].items()
            if importlib.metadata.version(p) != v]
    if platform.python_version() != rec1["python"]:
        bad.append(f"python {platform.python_version()} (recorded {rec1['python']})")
    if bad:
        raise SystemExit("supplement 3 verification failed:\n" + "\n".join(bad))
    print("freeze verified (freeze, supplements 1-3): models, code, packages, test inputs, artifacts, primary report")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("write", "verify"))
    a = p.parse_args()
    {"write": stage_write, "verify": stage_verify}[a.action](a)


if __name__ == "__main__":
    main()
