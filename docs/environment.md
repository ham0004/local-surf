# Environment report (measured 2026-09-26)

| Item | Value | How measured |
|---|---|---|
| OS | Windows 11 Pro 10.0.26200 | `platform.platform()` |
| CPU | AMD Ryzen 7 7700, 8 cores / 16 threads | `Win32_Processor`, `nproc` |
| RAM | 16.3 GB (15.2 GiB) | `psutil.virtual_memory()` |
| GPU | NVIDIA GeForce RTX 5060 Ti, 16 GB, compute capability 12.0 (Blackwell) | `nvidia-smi`, `torch.cuda.get_device_capability` |
| Driver / CUDA | 596.36 / CUDA 13.2 driver API | `nvidia-smi` |
| Disk | D: 431 GB, ~421 GB free at start | `df -h` |
| Python | CPython 3.12.14 (managed by uv 0.12.9) | `uv python list` |
| ffmpeg CLI | **not installed** — decoding uses PyAV's bundled FFmpeg (libx264 available) | `av.codec.Codec` checks |

## Pinned stack (see `uv.lock` for every transitive version)

| Package | Version | Note |
|---|---|---|
| torch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 | from the PyTorch cu128 index; PyPI Windows wheels are CPU-only and sm_120 needs CUDA ≥ 12.8. Verified: CUDA matmul on the GPU. |
| transformers | 5.17.0 | provides `Qwen3VLForConditionalGeneration`, `Qwen3_5ForConditionalGeneration`, `AutoModelForImageTextToText` |
| open_clip_torch | 3.3.0 | lists MobileCLIP / MobileCLIP2 checkpoints |
| av (PyAV) | 18.1.0 | decoding with true PTS; H.264 encoding for fixtures |
| (BM25) | in-repo | `retrieval.bm25_scores`, non-negative IDF (rank-bm25 removed: its IDF goes negative on tiny corpora) |
| numpy, pillow, pyyaml, psutil, pytest | see `uv.lock` | |

## Setup

```bash
uv sync --extra dev                  # CPU core: tests, fixtures, fixture-double pipeline
uv sync --extra dev --extra models   # + torch cu128, transformers, open_clip, peft
uv run pytest -q                     # 102 tests at the time of writing
```

Model weights are cached under `cache/hf` and `cache/open_clip` (git-ignored). On Windows,
Hugging Face warns that symlinks are unavailable without Developer Mode; the cache still works
but uses more disk.

## Hardware profiles

| Config | Target | Answerer | Scout | Frames | Measured on this machine |
|---|---|---|---|---|---|
| `configs/cpu.yaml` | CI / tests | fixture test double | pixel stats | 3 | only valid on synthetic fixture videos |
| `configs/cpu_vlm.yaml` | any CPU | Qwen3-VL-2B fp32 | pixel stats | 2 | cold 32.1 s, warm 10.5 s / question, 8.8 GB RSS |
| `configs/gpu_8gb.yaml` | 6–8 GB GPU | Qwen3-VL-2B bf16 | MobileCLIP-S2 | 4 | **not measured** (no such device available) |
| `configs/gpu_12gb.yaml` | ≥ 12 GB GPU | Qwen3-VL-2B bf16 | MobileCLIP-S2 | 8 | cold 14.8 s, warm 1.18 s, 5.45 GB peak VRAM allocated (6 frames) |

The 8 GB profile is expected to fit (5.45 GB allocated with 6 frames at 640 px on the 16 GB
card), but that is an inference from one measurement, not a test on an 8 GB device. Peak
"allocated" excludes allocator overhead and the CUDA context (~0.5–1 GB).
