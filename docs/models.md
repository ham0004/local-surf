# Model choices (verified 2026-09-26 via the Hugging Face Hub API)

| Role | Checkpoint | Revision (sha) | License (card) | Repo size | Status in this repo |
|---|---|---|---|---|---|
| Final answerer (frozen) | `Qwen/Qwen3-VL-2B-Instruct` | `89644892e4d85e24eaac8bacfd4f463576704203` | Apache-2.0 | 4.27 GB | **In use** (`answerer_hf.py`), GPU bf16 and CPU fp32 measured |
| Alternative answerer | `Qwen/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` | Apache-2.0 | 4.57 GB | Class available in transformers 5.17; **not yet run** |
| Tiny lower-bound answerer | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | `7b375e1b73b11138ff12fe22c8f2822d8fe03467` | Apache-2.0 | 7.92 GB (repo incl. extra formats) | Not yet run |
| Text controller (Stage 2+) | `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` | Apache-2.0 | 1.52 GB | Not yet used; numeric head comes first (see progress.md) |
| Visual scout (frozen) | open_clip `MobileCLIP-S2` / `datacompdr` (HF: `apple/MobileCLIP-S2-OpenCLIP`) | `8e8a808316aeb7c24d0400e1cf8f74b6937832aa` | **apple-amlr** | 0.80 GB | **In use** on GPU configs; 99.2 M params |

## License caution

* **MobileCLIP weights use Apple's `apple-amlr` license**, not Apache/MIT. It is intended for
  research use; check its terms before any redistribution or commercial use. A permissively
  licensed CLIP (e.g. an open_clip LAION checkpoint) can replace it via `scout.backend` settings.
* Dataset videos have their own terms (see `DATA_FEASIBILITY.md`), independent of model licenses.

## Measured behaviour worth knowing (synthetic lecture, GPU)

* Visual tokens per 640×360 frame for Qwen3-VL-2B: **220** (counted from image placeholder
  tokens in `input_ids`, not estimated).
* Without the relevant slide, Qwen3-VL-2B answered a visual-only question **wrongly with 0.95
  letter-probability**. Answerer confidence is therefore not used as a correctness signal.
* From a single "after" frame the VLM answered a before/after question correctly, whereas the
  fixture double requires two frames. Utility labels are answerer-specific; they must be
  generated with the same frozen answerer that is evaluated.
