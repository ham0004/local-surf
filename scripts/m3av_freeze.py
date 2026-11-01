"""Method freeze of the lecture study (docs/v3/m3av_protocol.md): every system, setting, checkpoint and code file
that decides which frames and text a system shows, hashed before the test cohort is generated.

Writes docs/v3/m3av_freeze.json (system definitions, pool settings, SHA-256 of every file below, git commit) and
prints its own SHA-256 for the research log. Refuses to overwrite an existing freeze.

    python scripts/m3av_freeze.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from v3_baseline_cycle import HEAD_A_CKPTS, HEAD_A_M3AV_CKPTS, POOLS, VSI_SETTING  # noqa: E402

OUT = Path("docs/v3/m3av_freeze.json")
# name: (pool, selector, transcript words, transcript policy); K = 4 frames for every system
SYSTEMS = {
    "S0": ("v2", "clip", 120, "bm25"),
    "S1": ("hybrid", "mmropt", 0, "none"),
    "S1t": ("hybrid", "mmropt", 600, "frames+hybrid"),
    "S2c": ("hybrid_ahead", "mmropt", 0, "none"),
    "S2m": ("hybrid_ahead_m3av", "mmropt", 0, "none"),
    "S3": ("hybrid", "file:{ds}_hybrid_ev_set (ev_set, lam 2, mu 1, CG-Bench-trained scorer)", 0, "none"),
    "R2": ("uniform4d", "clip", 0, "none"),
    "R3": ("dense4x", "clip", 0, "none"),
    "R4": ("dense4ox", "clip", 0, "none"),
    "R5": ("vsi4x", "clip", 0, "none"),
}
FILES = (["scripts/v3_baseline_cycle.py", "src/videoqa/v2/candidates.py", "src/videoqa/v3/head_a.py",
          "src/videoqa/v2/head_a.py", "src/videoqa/v2/encoders.py", "src/videoqa/answerer.py", "scripts/v3_head_b.py",
          "scripts/m3av_vsi.py", "scripts/m3av_hits.py", "scripts/m3av_dataset.py", "scripts/m3av_head_a_data.py",
          "scripts/m3av_visibility.py", "scripts/m3av_questions.py", "configs/gpu_16gb_4b.yaml"]
         + HEAD_A_CKPTS + HEAD_A_M3AV_CKPTS + [f"runs/v3_head_b/ckpt/ev_hybrid_{s}.pt" for s in range(3)]
         + ["cache/hf/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/"
            "e8c3b32edf5434bc2275fc9bab85f82640a19130/model.safetensors"])


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists: the methods are already frozen")
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *[f for f in FILES if not f.startswith(("runs/", "cache/"))]],
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise SystemExit(f"commit the code first:\n{dirty}")
    rec = {"frozen": "before the M3AV test cohort is generated",
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
           "k": 4, "answer_model": "Qwen3-VL-4B (configs/gpu_16gb_4b.yaml)", "vsi_setting": VSI_SETTING,
           "systems": {k: {"pool": p, "pool_settings": POOLS[p], "selector": s, "words": w, "text_policy": tp}
                       for k, (p, s, w, tp) in SYSTEMS.items()},
           "contrasts": {"L1": "S2m - R5", "L2": "S2m - S1", "correction": "Holm over L1 and L2"},
           "files": {f: sha(f) for f in FILES}}
    OUT.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"{OUT}: sha256 {sha(str(OUT))}")


if __name__ == "__main__":
    main()
