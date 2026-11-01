"""Supplement to the lecture-study method freeze (docs/v3/m3av_freeze.json, auditor P71-P73). The original record is
kept unchanged; this adds what it left implicit:

- external models: the revision each name resolves to in the local cache, with SHA-256 of every snapshot file;
- package versions; the resolved PoolConfig, selector and answer-model configuration of every frozen system;
- the files changed after the freeze (test-path wiring, R5 setting guard, Holm reporting), with new hashes;
- the exact primary test commands (scripts/run_m3av_test.sh).

Mutable inputs and artifacts get their own manifests: inputs (split file, cohort, qa.jsonl, transcripts, videos)
when a split is built, artifacts (dense features, R5 ranking, pools, Head B selections) after they are built and
before anything is scored. Text files are hashed with line endings normalised to LF (Git converts them on checkout);
binary files byte for byte. Every writer refuses to overwrite.

    python scripts/m3av_freeze_supplement.py write                                   # docs/v3/m3av_freeze_supplement.json
    python scripts/m3av_freeze_supplement.py manifest --split test --stage inputs    # docs/v3/m3av_manifest_test_inputs.json
    python scripts/m3av_freeze_supplement.py verify [--split test --stage inputs]    # models, code, and a manifest
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from m3av_freeze import SYSTEMS  # noqa: E402
from v3_baseline_cycle import (  # noqa: E402
    DATASETS, FEATURES, HEAD_A_CKPTS, HEAD_A_M3AV_CKPTS, POOLS, RUN, SELECTIONS, VSI_SETTING)

FREEZE, OUT = Path("docs/v3/m3av_freeze.json"), Path("docs/v3/m3av_freeze_supplement.json")
FREEZE_SHA = {"lf": "f054ce41038497d8fd6f5e4703ffd8c6bb5e24fa743209b7dec3e0f1861beb3e",
              "crlf": "bead18c3f6d12c7f297a268ed4ebdd9e0539214d68d8f9dda4209d0c556fb687"}
TEXT = {".py", ".sh", ".yaml", ".yml", ".json", ".jsonl", ".md", ".txt"}
NEW_FILES = ["scripts/m3av_freeze_supplement.py", "scripts/run_m3av_test.sh"]
# name: (cache dir, how the code names it, pinned revision or None = whatever refs/main resolves to)
MODELS = {"cross-encoder/ms-marco-MiniLM-L6-v2": ("cache/hf/hub", "src/videoqa/v2/head_a.py BACKBONE", None),
          "BAAI/bge-reranker-base": ("cache/hf/hub", "scripts/v3_head_a_light.py BGE (no revision argument)", None),
          "apple/MobileCLIP-S2-OpenCLIP": ("cache/open_clip", "open_clip MobileCLIP-S2 datacompdr", None),
          "sentence-transformers/all-mpnet-base-v2": ("cache/hf/hub", "scripts/m3av_vsi.py MPNET",
                                                      "e8c3b32edf5434bc2275fc9bab85f82640a19130"),
          "Qwen/Qwen3-VL-4B-Instruct": ("cache/hf/hub", "configs/gpu_16gb_4b.yaml", "ebb281ec70b05090aa6165b016eac8ec08e71b17")}
PACKAGES = ["torch", "torchvision", "transformers", "tokenizers", "open_clip_torch", "timm", "safetensors",
            "huggingface_hub", "accelerate", "numpy", "opencv-python", "pillow", "av"]
FROZEN_POOLS = sorted({p for p, *_ in SYSTEMS.values()})
S3 = {"stage features": "--dataset {ds} --pool hybrid", "stage select": "--dataset {ds} --pool hybrid --train-pool hybrid "
      "--method ev_set --lam 2.0 --mu 1.0 (k 4, gap 0, a-quota 4: defaults)",
      "checkpoints": [f"runs/v3_head_b/ckpt/ev_hybrid_{s}.pt" for s in range(3)],
      "inputs": "S1's hybrid pool; per-frame MobileCLIP similarities to question and options, speech relevance of "
                "nearby transcript lines (MiniLM-L6 cross-encoder)"}


def sha(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix in TEXT:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def tree(root: Path) -> dict:
    """Aggregate hash of a directory: SHA-256 over the sorted (relative path, file hash) list."""
    files = sorted(p for p in root.rglob("*") if p.is_file())
    h = hashlib.sha256()
    for p in files:
        h.update(f"{p.relative_to(root).as_posix()}\t{sha(p)}\n".encode())
    return {"files": len(files), "sha256": h.hexdigest()}


def models() -> dict:
    out = {}
    for name, (cache, named, pinned) in MODELS.items():
        d = Path(cache) / ("models--" + name.replace("/", "--"))
        main = (d / "refs" / "main").read_text().strip() if (d / "refs" / "main").exists() else None
        rev = pinned or main
        snap = d / "snapshots" / rev
        out[name] = {"loaded_by": named, "revision": rev, "refs_main": main,
                     "files": {p.relative_to(snap).as_posix(): sha(p) for p in sorted(snap.rglob("*")) if p.is_file()}}
    return out


def resolved(split: str) -> dict:
    from videoqa.config import load_config  # noqa: PLC0415
    from videoqa.v2.candidates import PoolConfig  # noqa: PLC0415

    ds = f"m3av_{split}"
    pools = {}
    for pool in FROZEN_POOLS:                         # as stage_pools in v3_baseline_cycle.py builds them
        s = dict(POOLS[pool])
        path_a = s.pop("path_a", "v2")
        if "times_file" in s:
            s["times_file"] = s["times_file"].format(split=DATASETS[ds][1], setting=VSI_SETTING)
        cfg = PoolConfig(**s, excerpt_words=300, features_dir=FEATURES[ds] if pool.startswith(("dense", "hybrid")) else None)
        pools[pool] = {"path_a": path_a, "head_a_checkpoints": {"head": HEAD_A_CKPTS, "head_m3av": HEAD_A_M3AV_CKPTS}.get(path_a),
                       "config": dataclasses.asdict(cfg)}
    return {"dataset": ds, "pools": pools,
            "selectors": {"clip": "top 4 by the candidate's MobileCLIP similarity to the question",
                          "mmropt": "greedy MMR, 0.7 * similarity to 'question Options: ...' - 0.3 * max similarity to "
                                    "a chosen frame (MobileCLIP)",
                          "file:<ds>_hybrid_ev_set": S3},
            "dense_features": "scripts/v3_features.py --step 2: MobileCLIP-S2 image embeddings at 0.5, 2.5, 4.5 ... s, "
                              "stored float16",
            "r5_setting": VSI_SETTING, "k": 4,
            "answer_model_config (secondary QA)": load_config("configs/gpu_16gb_4b.yaml")}


def code_changes() -> dict:
    orig = json.loads(FREEZE.read_text(encoding="utf-8"))["files"]
    changed = {}
    for f, h in orig.items():
        p = Path(f)
        if p.suffix in TEXT:
            raw = hashlib.sha256(p.read_bytes()).hexdigest()
            if h not in (raw, hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()):
                changed[f] = sha(p)
        elif hashlib.sha256(p.read_bytes()).hexdigest() != h:
            raise SystemExit(f"frozen binary changed: {f}")
    return {"changed": changed, "new": {f: sha(Path(f)) for f in NEW_FILES}}


def stage_write(_a) -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists")
    if sha(FREEZE) != FREEZE_SHA["lf"]:
        raise SystemExit("the original freeze record changed")
    tracked = [*NEW_FILES, "scripts/v3_baseline_cycle.py", "scripts/m3av_vsi.py", "scripts/m3av_hits.py"]
    if subprocess.run(["git", "status", "--porcelain", "--", *tracked], capture_output=True, text=True,
                      check=True).stdout.strip():
        raise SystemExit("commit the code first")
    rec = {"supplements": {"record": str(FREEZE), "sha256": FREEZE_SHA, "source_commit": "e9bd1ad",
                           "freeze_commit": "3a2e911"},
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
           "why": "auditor P71-P73: external model resolutions, resolved settings, test-path wiring and reporting; no "
                  "model, setting or selection rule changes",
           "r5_attribution": "R5 is a VSI-inspired MPNet/MobileCLIP baseline, not VSI: boost min(1, 2M) for M > 0.5 "
                             "(paper Eq. 8: min(1, M + 2 max(M - 0.5, 0)); released code min(1, 0.5 + 2 (M - 0.5))), "
                             "Gaussian over the whole video (VSI: line +/- W), raw strings (VSI code: lowercase, "
                             "stopwords removed), MobileCLIP option-aware visual branch instead of VLM + YOLO-World",
           "python": platform.python_version(),
           "packages": {p: importlib.metadata.version(p) for p in PACKAGES},
           "models": models(), "code_after_freeze": code_changes(), "test": resolved("test"),
           "primary_test_commands": {"file": "scripts/run_m3av_test.sh",
                                     "text": Path("scripts/run_m3av_test.sh").read_text(encoding="utf-8")},
           "secondary_qa": "the answer-model commands are recorded in a further supplement before the answer model "
                           "runs on test"}
    OUT.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"{OUT.as_posix()}: sha256 {sha(OUT)}")


def manifest(split: str, stage: str) -> dict:
    ds = f"m3av_{split}"
    root = DATASETS[ds][0]
    if stage == "inputs":
        return {"data/m3av_raw/set_split.json": sha(Path("data/m3av_raw/set_split.json")),
                f"data/m3av/questions_{split}_visible.jsonl": sha(Path(f"data/m3av/questions_{split}_visible.jsonl")),
                f"{root}/qa.jsonl": sha(Path(root, "qa.jsonl")),
                f"{root}/transcripts": tree(Path(root, "transcripts")), f"{root}/videos": tree(Path(root, "videos"))}
    out = {FEATURES[ds]: tree(Path(FEATURES[ds])),
           f"runs/m3av/vsi/{split}_{VSI_SETTING}.json": sha(Path(f"runs/m3av/vsi/{split}_{VSI_SETTING}.json")),
           f"runs/v3_head_b/features_{ds}_hybrid.npy": sha(Path(f"runs/v3_head_b/features_{ds}_hybrid.npy")),
           f"{SELECTIONS.as_posix()}/{ds}_hybrid_ev_set.json": sha(SELECTIONS / f"{ds}_hybrid_ev_set.json")}
    out.update({f"{RUN.as_posix()}/{ds}/{p}/pools": tree(RUN / ds / p / "pools") for p in FROZEN_POOLS})
    return out


def stage_manifest(a) -> None:
    dest = Path(f"docs/v3/m3av_manifest_{a.split}_{a.part}.json")
    if dest.exists():
        raise SystemExit(f"{dest} exists")
    dest.write_text(json.dumps(manifest(a.split, a.part), indent=1), encoding="utf-8")
    print(f"{dest.as_posix()}: sha256 {sha(dest)}")


def stage_verify(a) -> None:
    rec = json.loads(OUT.read_text(encoding="utf-8"))
    bad = [f"model {n}" for n, v in models().items() if v != rec["models"][n]]
    now = code_changes()
    bad += [f"code {f}" for kind in ("changed", "new") for f, h in rec["code_after_freeze"][kind].items()
            if now[kind].get(f) != h]
    bad += [f"code {f} changed again" for f in now["changed"] if f not in rec["code_after_freeze"]["changed"]]
    if a.split:
        want = json.loads(Path(f"docs/v3/m3av_manifest_{a.split}_{a.part}.json").read_text(encoding="utf-8"))
        have = manifest(a.split, a.part)
        bad += [f"{a.part} {k}" for k in want if have.get(k) != want[k]]
    if bad:
        raise SystemExit("freeze verification failed:\n" + "\n".join(bad))
    print("freeze verified: models, code" + (f", {a.split} {a.part}" if a.split else ""))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("write", "manifest", "verify"))
    p.add_argument("--split", choices=("dev", "test"), default=None)
    p.add_argument("--stage", dest="part", choices=("inputs", "artifacts"), default=None)
    a = p.parse_args()
    if a.action == "manifest" and not (a.split and a.part):
        raise SystemExit("manifest needs --split and --stage")
    {"write": stage_write, "manifest": stage_manifest, "verify": stage_verify}[a.action](a)


if __name__ == "__main__":
    main()
