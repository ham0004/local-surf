"""R5 of docs/v3/m3av_protocol.md: a VSI-inspired MPNet/MobileCLIP subtitle + visual frame search, 4 frames.
An adaptation of VSI (arXiv 2508.06869), not a reproduction: the published system is not compared.

Text branch: all-mpnet-base-v2 (revision e8c3b32edf5434bc2275fc9bab85f82640a19130, mean pooling, unit norm) cosine
M between the query (question + options) and each transcript line, on the raw strings; boost B = min(1, 2M) when
M > 0.5, else M; lines with B <= 0.2 are dropped; each kept line spreads B with a Gaussian centred on its midpoint,
sigma = (end - start + 2 W) / 4, W = 2 s, evaluated at every video time; S_text(t) = the maximum over lines.
Visual branch: VSI asks a VLM for target objects and runs YOLO-World; here it is the dense MobileCLIP similarity of
each 2 s frame to the question (as R3) or to the best of question + each option (as R4).
Fusion as in VSI: S = lam * z(S_text) + (1 - lam) * z(S_visual), z over the video. Saved: the best 96 times by S,
at least 4 s apart, best first (the pool keeps the first 4 distinct pictures); the dev measure uses the first 4.
VSI's iterative sampling is not needed: every 2 s frame is scored.

Differences from VSI's text branch (auditor P71; kept as frozen, docs/v3/m3av_freeze_supplement.json):
- boost: paper Eq. 8 is min(1, M + 2 max(M - 0.5, 0)); the released code is min(1, 0.5 + 2 (M - 0.5)); here
  min(1, 2M), so every line with M > 0.5 scores 1 (at M = 0.6: paper 0.8, code 0.7, here 1.0);
- Gaussian support: VSI limits it to the line expanded by W; here it covers the whole video;
- preprocessing: the released code lowercases and removes stopwords; here the raw strings are embedded.

lam and the visual query were chosen on dev only by source-slide interval hit@4 (visible spans, 1 s tolerance),
scoring the first 4 ranked times; test runs only the frozen setting (VSI_SETTING in v3_baseline_cycle.py).

    python scripts/m3av_vsi.py --split dev                              # all settings; runs/m3av/vsi/<split>_<setting>.json
    python scripts/m3av_vsi.py --split test --setting lam0.5_qo_max     # test: the frozen setting only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

OUT = Path("runs/m3av/vsi")
MPNET, MPNET_REV = "sentence-transformers/all-mpnet-base-v2", "e8c3b32edf5434bc2275fc9bab85f82640a19130"
LAMS, VQUERIES = (0.0, 0.25, 0.5, 0.75, 1.0), ("q", "qo_max")
W_S, THETA, GAMMA, TAU_TEXT, GAP_S, K = 2.0, 0.5, 2.0, 0.2, 4.0, 4


class MPNet:
    def __init__(self) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(MPNET, revision=MPNET_REV, cache_dir="cache/hf/hub")
        self.model = AutoModel.from_pretrained(MPNET, revision=MPNET_REV, cache_dir="cache/hf/hub").to("cuda").eval()

    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), 128):
                b = self.tok(texts[i:i + 128], padding=True, truncation=True, max_length=384, return_tensors="pt").to("cuda")
                h = self.model(**b).last_hidden_state
                m = b["attention_mask"][..., None].float()
                e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
                out.append(self.torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 768), np.float32)


def text_score(times: np.ndarray, lines, sims: np.ndarray) -> np.ndarray:
    b = np.where(sims > THETA, np.minimum(1.0, sims * GAMMA), sims)
    s = np.zeros_like(times, dtype=np.float64)
    for (a, e), bi in zip(lines, b, strict=True):
        if bi > TAU_TEXT:
            sigma = (e - a + 2 * W_S) / 4
            s = np.maximum(s, bi * np.exp(-0.5 * ((times - 0.5 * (a + e)) / sigma) ** 2))
    return s


def z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-6)


def ranked(times: np.ndarray, score: np.ndarray, n: int = 96) -> list[float]:
    out: list[float] = []
    for i in np.argsort(-score, kind="stable"):
        if all(abs(float(times[i]) - u) >= GAP_S for u in out):
            out.append(float(times[i]))
            if len(out) == n:
                break
    return out


def main() -> None:
    from videoqa.datasets import load_local_dataset  # noqa: PLC0415
    from videoqa.v2.encoders import FrozenEncoders  # noqa: PLC0415

    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=("dev", "test"))
    p.add_argument("--setting", default=None, help="test: the frozen setting (lam0.5_qo_max)")
    a = p.parse_args()
    if a.split == "test":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from v3_baseline_cycle import VSI_SETTING  # noqa: PLC0415
        if a.setting != VSI_SETTING:
            raise SystemExit(f"test runs only the frozen setting: --setting {VSI_SETTING}")
    root = Path(f"data/m3av_{a.split}")
    its, _ = load_local_dataset(root, splits=(a.split,))
    settings = [f"lam{lam:g}_{v}" for lam in LAMS for v in VQUERIES] if a.setting is None else [a.setting]
    enc, mp = FrozenEncoders(device="cuda", cache_dir="cache/open_clip", use_ocr=False), MPNet()
    picks: dict = {s: {} for s in settings}
    for n, it in enumerate(its):
        qa, segs = it.qa, it.transcript.segments
        f = np.load(root / "features" / f"{qa.video_id}.npz")
        times, emb = f["times"].astype(np.float64), f["emb"].astype(np.float32)
        query = qa.question + " " + " ".join(qa.options)
        e = mp.embed([query] + [s.text for s in segs])
        st = z(text_score(times, [(s.start_s, s.end_s) for s in segs], e[1:] @ e[0]))
        vis = {"q": emb @ enc.embed_texts([qa.question])[0],
               "qo_max": (emb @ enc.embed_texts([qa.question + " " + o for o in qa.options]).T).max(1)}
        for s in settings:
            lam, v = float(s.split("_")[0][3:]), s.split("_", 1)[1]
            picks[s][qa.qa_id] = ranked(times, lam * st + (1 - lam) * z(vis[v]))
        if (n + 1) % 50 == 0:
            print(f"{n + 1}/{len(its)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for s in settings:
        (OUT / f"{a.split}_{s}.json").write_text(json.dumps(picks[s]), encoding="utf-8")
        if a.split == "dev":                       # dev only: the measure that chooses the setting
            hits = [any(x - 1.0 <= t <= y + 1.0 for t in picks[s][it.qa.qa_id][:K] for x, y in it.qa.evidence_intervals_s)
                    for it in its]
            report[s] = {"hit@4": round(100 * float(np.mean(hits)), 1), "n": len(hits)}
            print(s, report[s])
    if report:
        (OUT / "dev_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
