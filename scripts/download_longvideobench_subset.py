"""Fetch a small, real subset of LongVideoBench and convert it to the local layout.

Downloads ONLY what is needed for the requested subset:
  * ``lvb_val.json`` (small, 1,337 questions / 753 videos with gold answers —
    the Hub "test" split has no gold and is not used)
  * ``subtitles.tar`` (small, ~9,345 subtitle files; downloaded once in full,
    it is far smaller than the video archive)
  * from the 161.6 GB, 31-part video tar: ONLY the selected videos, via
    ``videoqa.sources.tar_range`` (HTTP Range requests; see that module).

Selection (deterministic): videos with duration <= --max-duration (default:
no limit) that have at least one T*-category question (T2A/T2O/T2E/T3O/T3E/TOS — these
anchor the question on an exact quoted subtitle span, which is what makes
transcript damage meaningful) and a subtitle file. Short videos keep the
pilot's bandwidth small; the default takes every eligible video (measured:
252 videos, 440 T* questions, 9.2 GB).

Archive names come from each record's ``video_path``, NOT its ``video_id``:
the 135 TikTok-sourced videos have ids like "@recipesbyanne-7141686631676808454"
but are stored as "videos/7141686631676808454.mp4". (An earlier version built
names from ``video_id``, concluded those videos were "absent", and excluded
them; checked against the completed catalog, all 753 val videos are present.)

A local catalog (``<cache>/lvb_video_catalog.json``) holds the archive's full
directory (name -> byte offset/size), built once by
``videoqa.sources.tar_range.build_catalog`` (resumable; ~1-2 h the first time
on this connection). After that, every selection is an exact lookup: no
scanning, and a wanted file that is not in a COMPLETE catalog is truly absent.

When --n-videos caps the selection, videos are taken in a fixed pseudo-random
order (sha256 of the id) rather than sorted by id: sorting would put ids
starting with "-", digits and "@" (one source) first and bias the subset.

Usage:
    uv run python scripts/download_longvideobench_subset.py \\
        --out data/longvideobench --token $HF_TOKEN            # all eligible
    ... --n-videos 40 --max-duration 70                      # a small pilot

Requires: the account behind --token must have accepted the dataset's terms
at https://huggingface.co/datasets/longvideobench/LongVideoBench (gated,
CC-BY-NC-SA-4.0 — non-commercial; recorded in every converted QAItem's
provenance_and_license field).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from huggingface_hub import hf_hub_download  # noqa: E402

from videoqa.schemas import to_jsonable  # noqa: E402
from videoqa.sources.longvideobench import T_STAR_CATEGORIES, convert_item, subtitles_to_transcript  # noqa: E402
from videoqa.sources.tar_range import RemoteMultipartTar, build_catalog, hf_dataset_tar_parts  # noqa: E402

DATASET_ID = "longvideobench/LongVideoBench"
LICENSE_NOTE = "CC-BY-NC-SA-4.0 (non-commercial), longvideobench/LongVideoBench, gated dataset"
# The 31 part filenames as listed on the Hub (fetched sizes via the API, not hard-coded).
PART_NAMES = [f"videos.tar.part.{c}" for c in
             ["aa", "ab", "ac", "ad", "ae", "af", "ag", "ah", "ai", "aj", "ak", "al", "am", "an", "ao", "ap",
              "aq", "ar", "as", "at", "au", "av", "aw", "ax", "ay", "az", "ba", "bb", "bc", "bd", "be"]]


def archive_name(record: dict) -> str:
    """Name of a video inside the archive. Uses ``video_path``, not ``video_id``
    (they differ for all 135 TikTok-sourced videos)."""
    return f"videos/{record['video_path']}"


def select_videos(by_video: dict[str, list[dict]], max_duration: float | None, n_videos: int | None,
                  subtitle_names: set[str], archive_names: set[str]) -> list[str]:
    """Eligible video ids: has a T*-category question, a subtitle file, is in
    the archive, and is within ``max_duration``. Capped selections follow a
    fixed pseudo-random order (sha256 of the id) so they are unbiased and
    reproducible."""
    eligible = []
    for vid, items in by_video.items():
        if max_duration is not None and items[0]["duration"] > max_duration:
            continue
        if not any(i["question_category"] in T_STAR_CATEGORIES for i in items):
            continue
        if f"subtitles/{items[0]['subtitle_path']}" not in subtitle_names:
            continue
        if archive_name(items[0]) not in archive_names:
            continue
        eligible.append(vid)
    eligible.sort(key=lambda v: hashlib.sha256(v.encode("utf-8")).hexdigest())
    return eligible if n_videos is None else eligible[:n_videos]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/longvideobench")
    ap.add_argument("--cache", default="cache/lvb_meta", help="where the small metadata files are cached")
    ap.add_argument("--n-videos", type=int, default=None, help="cap on videos (default: every eligible one)")
    ap.add_argument("--max-duration", type=float, default=None, help="seconds (default: no limit)")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--dry-run", action="store_true", help="select and report, download nothing")
    ap.add_argument("--max-retries", type=int, default=6,
                    help="retries per HTTP range read before giving up (transient timeouts/5xx)")
    a = ap.parse_args()

    if not a.token:
        raise SystemExit("no token: pass --token or set HF_TOKEN (see docs/DATA_FEASIBILITY.md)")

    out = Path(a.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    (out / "transcripts").mkdir(parents=True, exist_ok=True)

    print("Fetching metadata (small: val json, subtitle archive)...", flush=True)
    val_path = hf_hub_download(DATASET_ID, "lvb_val.json", repo_type="dataset", token=a.token, local_dir=a.cache)
    sub_path = hf_hub_download(DATASET_ID, "subtitles.tar", repo_type="dataset", token=a.token, local_dir=a.cache)
    val_data = json.loads(Path(val_path).read_text(encoding="utf-8"))
    subs = tarfile.open(sub_path)
    subtitle_names = set(subs.getnames())
    by_video: dict[str, list[dict]] = {}
    for d in val_data:
        by_video.setdefault(d["video_id"], []).append(d)

    # 1. A COMPLETE directory of the archive, built once and then reused.
    #    build_catalog resumes from wherever a previous scan stopped.
    base, parts = hf_dataset_tar_parts(DATASET_ID, PART_NAMES, token=a.token)
    tar = RemoteMultipartTar(base, parts, token=a.token, max_retries=a.max_retries)
    catalog_path = Path(a.cache) / "lvb_video_catalog.json"
    t_scan = time.perf_counter()
    catalog = build_catalog(tar, catalog_path)
    print(f"Archive catalog: {len(catalog.entries)} entries, complete={catalog.complete} "
          f"({time.perf_counter() - t_scan:.1f}s).", flush=True)

    # 2. Selection is now an exact lookup against that catalog.
    video_ids = select_videos(by_video, a.max_duration, a.n_videos, subtitle_names, set(catalog.entries))
    n_q = sum(sum(r["question_category"] in T_STAR_CATEGORIES for r in by_video[v]) for v in video_ids)
    total_bytes = sum(catalog.entries[archive_name(by_video[v][0])].size for v in video_ids)
    print(f"Selected {len(video_ids)} videos, {n_q} T* questions, {total_bytes / 1e9:.2f} GB "
          f"(n_videos={a.n_videos}, max_duration={a.max_duration}).", flush=True)
    if a.dry_run:
        print(json.dumps(video_ids, indent=2))
        return

    # 3. Transcripts, re-based onto each clip's own timeline (see
    #    subtitles_to_transcript: short clips ship their SOURCE video's subtitles).
    transcripts = {}
    for vid in video_ids:
        rec0 = by_video[vid][0]
        raw = json.loads(subs.extractfile(f"subtitles/{rec0['subtitle_path']}").read().decode("utf-8"))
        t = subtitles_to_transcript(vid, raw, offset_s=rec0["starting_timestamp_for_subtitles"],
                                    clip_duration_s=rec0["duration"])
        transcripts[vid] = t
        (out / "transcripts" / f"{vid}.json").write_text(
            json.dumps([{"start": s.start_s, "end": s.end_s, "text": s.text} for s in t.segments]),
            encoding="utf-8")

    # 4. Download each selected video (skipping any already complete on disk:
    #    same size as the archive entry; extract_to never leaves partial files).
    t0 = time.perf_counter()
    downloaded_bytes = 0
    found: list[str] = []
    download_errors: dict[str, str] = {}
    for k, vid in enumerate(video_ids, 1):
        entry = catalog.entries[archive_name(by_video[vid][0])]
        dest = out / "videos" / f"{vid}.mp4"
        if dest.exists() and dest.stat().st_size == entry.size:
            found.append(vid)
            continue
        try:
            downloaded_bytes += tar.extract_to(entry, dest)
            found.append(vid)
            print(f"  [{k}/{len(video_ids)}] {vid} ({entry.size / 1e6:.1f} MB)", flush=True)
        except Exception as e:  # noqa: BLE001 - keep going; report per-video failures in the manifest
            download_errors[vid] = f"{type(e).__name__}: {e}"
            print(f"  [{k}/{len(video_ids)}] {vid} FAILED: {download_errors[vid]}", flush=True)
    elapsed = time.perf_counter() - t0
    missing = [v for v in video_ids if v not in set(found)]

    # 5. QA rows for videos we actually have, in selection order (deterministic).
    qa_lines, ratios = [], []
    for vid in found:
        for record in by_video[vid]:
            item = convert_item(record, transcripts[vid], LICENSE_NOTE)
            qa_lines.append(json.dumps(to_jsonable(item.qa)))
            if item.evidence_match_ratio is not None:
                ratios.append(item.evidence_match_ratio)
    (out / "qa.jsonl").write_text("\n".join(qa_lines) + "\n", encoding="utf-8")

    manifest = {
        "dataset": DATASET_ID, "license": LICENSE_NOTE,
        "selection": {"n_videos": a.n_videos, "max_duration": a.max_duration, "order": "sha256(video_id)"},
        "n_videos_selected": len(video_ids), "n_videos_available": len(found), "missing_videos": missing,
        "download_errors": download_errors,
        "n_questions": len(qa_lines), "n_with_evidence_quote": len(ratios),
        "mean_evidence_match_ratio": sum(ratios) / len(ratios) if ratios else None,
        "downloaded_video_bytes_this_run": downloaded_bytes, "download_seconds": round(elapsed, 1),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "complete": not missing,
        "catalog_path": str(catalog_path), "catalog_entries": len(catalog.entries),
        "catalog_complete": catalog.complete,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if missing:
        print(f"WARNING: {len(missing)} selected videos could not be downloaded (re-run to retry): {missing}")


if __name__ == "__main__":
    main()
