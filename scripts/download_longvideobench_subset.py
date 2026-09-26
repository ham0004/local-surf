"""Fetch a small, real subset of LongVideoBench and convert it to the local layout.

Downloads ONLY what is needed for the requested subset:
  * ``lvb_val.json`` (small, 1,337 questions / 753 videos with gold answers —
    the Hub "test" split has no gold and is not used)
  * ``subtitles.tar`` (small, ~9,345 subtitle files; downloaded once in full,
    it is far smaller than the video archive)
  * from the 161.6 GB, 31-part video tar: ONLY the selected videos, via
    ``videoqa.sources.tar_range`` (HTTP Range requests; see that module).

Selection (deterministic, sorted by video_id): videos with duration <= 70 s
that have at least one T*-category question (T2A/T2O/T2E/T3O/T3E/TOS — these
anchor the question on an exact quoted subtitle span, which is what makes
transcript damage meaningful) and a subtitle file. Short videos keep the
pilot's bandwidth and label-generation cost small; scaling up later means
raising --n-videos and/or --max-duration.

Video ids starting with "@" (the dataset's TikTok-sourced clips, e.g.
"@healthfood-6867204066108329221") are excluded from selection: a full,
completed scan of every entry in the video archive (3,992 entries, confirmed
via the persisted catalog — see below) never found ANY of them there, so they
are recorded as confirmed absent rather than repeatedly searched for.

A local catalog (``<cache>/lvb_video_catalog.json``) remembers every archive
entry this script has ever seen, by byte offset, so re-running with a wider
selection (more videos, longer max-duration) never re-scans bytes it has
already read — only genuinely new names trigger more scanning.

Usage:
    uv run python scripts/download_longvideobench_subset.py \\
        --out data/longvideobench --n-videos 40 --token $HF_TOKEN

Requires: the account behind --token must have accepted the dataset's terms
at https://huggingface.co/datasets/longvideobench/LongVideoBench (gated,
CC-BY-NC-SA-4.0 — non-commercial; recorded in every converted QAItem's
provenance_and_license field).
"""

from __future__ import annotations

import argparse
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
from videoqa.sources.tar_range import RemoteMultipartTar, hf_dataset_tar_parts, load_catalog, save_catalog  # noqa: E402

DATASET_ID = "longvideobench/LongVideoBench"
LICENSE_NOTE = "CC-BY-NC-SA-4.0 (non-commercial), longvideobench/LongVideoBench, gated dataset"
# The 31 part filenames as listed on the Hub (fetched sizes via the API, not hard-coded).
PART_NAMES = [f"videos.tar.part.{c}" for c in
             ["aa", "ab", "ac", "ad", "ae", "af", "ag", "ah", "ai", "aj", "ak", "al", "am", "an", "ao", "ap",
              "aq", "ar", "as", "at", "au", "av", "aw", "ax", "ay", "az", "ba", "bb", "bc", "bd", "be"]]


def select_videos(val_data: list[dict], max_duration: float, n_videos: int,
                  subtitle_names: set[str], known_absent: frozenset[str] = frozenset()) -> list[str]:
    by_video: dict[str, list[dict]] = {}
    for d in val_data:
        by_video.setdefault(d["video_id"], []).append(d)
    picked = []
    for vid, items in sorted(by_video.items()):
        if vid.startswith("@"):
            continue  # TikTok-sourced ids: confirmed absent from this video archive (see module docstring)
        if vid in known_absent:
            continue
        if items[0]["duration"] > max_duration:
            continue
        if not any(i["question_category"] in T_STAR_CATEGORIES for i in items):
            continue
        if f"subtitles/{items[0]['subtitle_path']}" not in subtitle_names:
            continue
        picked.append(vid)
        if len(picked) >= n_videos:
            break
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/longvideobench")
    ap.add_argument("--cache", default="cache/lvb_meta", help="where the small metadata files are cached")
    ap.add_argument("--n-videos", type=int, default=40)
    ap.add_argument("--max-duration", type=float, default=70.0, help="seconds")
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

    catalog_path = Path(a.cache) / "lvb_video_catalog.json"
    catalog = load_catalog(catalog_path)
    known_absent = frozenset() if not catalog.complete else frozenset(
        vid for d in val_data if (vid := d["video_id"]) and f"videos/{vid}.mp4" not in catalog.entries)
    video_ids = select_videos(val_data, a.max_duration, a.n_videos, subtitle_names, known_absent)
    print(f"Selected {len(video_ids)} videos (target {a.n_videos}, max_duration {a.max_duration}s).", flush=True)
    if a.dry_run:
        print(json.dumps(video_ids, indent=2))
        return

    by_video: dict[str, list[dict]] = {}
    for d in val_data:
        by_video.setdefault(d["video_id"], []).append(d)

    # Extract subtitles for selected videos and build transcripts up front
    # (needed for evidence matching before we even touch the video archive).
    transcripts = {}
    for vid in video_ids:
        sub_name = f"subtitles/{by_video[vid][0]['subtitle_path']}"
        raw = json.loads(subs.extractfile(sub_name).read().decode("utf-8"))
        rec0 = by_video[vid][0]
        t = subtitles_to_transcript(vid, raw, offset_s=rec0["starting_timestamp_for_subtitles"],
                                    clip_duration_s=rec0["duration"])
        transcripts[vid] = t
        (out / "transcripts" / f"{vid}.json").write_text(
            json.dumps([{"start": s.start_s, "end": s.end_s, "text": s.text} for s in t.segments]),
            encoding="utf-8")

    # Resumable: a video already on disk (from a prior, interrupted run) is
    # not re-fetched. We still need the archive scan to know its byte range
    # is only for videos still missing, so `wanted_names` shrinks accordingly
    # but `found` starts pre-populated with what is already there.
    def _already_downloaded(vid: str) -> bool:
        p = out / "videos" / f"{vid}.mp4"
        return p.exists() and p.stat().st_size > 0

    found = {vid for vid in video_ids if _already_downloaded(vid)}
    if found:
        print(f"Resuming: {len(found)}/{len(video_ids)} videos already downloaded.", flush=True)
    wanted_names = {f"videos/{vid}.mp4" for vid in video_ids if vid not in found}

    base, parts = hf_dataset_tar_parts(DATASET_ID, PART_NAMES, token=a.token)
    tar = RemoteMultipartTar(base, parts, token=a.token, max_retries=a.max_retries)
    t0 = time.perf_counter()
    n_scanned = 0
    downloaded_bytes = 0
    scan_error: str | None = None

    def _extract_wanted(name: str, entry) -> None:
        nonlocal downloaded_bytes
        vid = name[len("videos/") : -len(".mp4")]
        downloaded_bytes += tar.extract_to(entry, out / "videos" / f"{vid}.mp4")
        found.add(vid)
        print(f"  [{len(found)}/{len(video_ids)}] {vid}.mp4 ({entry.size / 1e6:.1f} MB)", flush=True)

    # Resolve whatever we can straight from the cached catalog first: zero
    # network requests for names we have already scanned in a previous run.
    resolved_from_cache = 0
    for name in list(wanted_names):
        if name in catalog.entries:
            _extract_wanted(name, catalog.entries[name])
            wanted_names.discard(name)
            resolved_from_cache += 1
    if catalog.entries:
        print(f"Resolved {resolved_from_cache} names from the cached catalog "
             f"({len(catalog.entries)} entries known, complete={catalog.complete}).", flush=True)

    if wanted_names and not catalog.complete:
        print(f"Scanning for {len(wanted_names)} names not yet in the cached catalog, "
             f"resuming at byte {catalog.next_offset:,} of {tar.total_size:,} "
             "(each request may retry on transient network errors)...", flush=True)
        since_save = 0
        try:
            for entry in tar.iter_entries(stop_after=wanted_names, start_offset=catalog.next_offset):
                n_scanned += 1
                catalog.entries[entry.name] = entry
                data_blocks = (entry.size + 511) // 512
                catalog.next_offset = entry.global_offset + 512 + data_blocks * 512
                since_save += 1
                if since_save >= 50:
                    save_catalog(catalog, catalog_path)
                    since_save = 0
                if entry.name in wanted_names:
                    _extract_wanted(entry.name, entry)
                    wanted_names.discard(entry.name)
            # Normal completion with names still unresolved means iter_entries
            # ran off the true end of the archive (stop_after was never
            # satisfied) rather than stopping early - so we now know the FULL
            # contents of the archive, even though not everything we wanted
            # was in it.
            if wanted_names:
                catalog.complete = True
        except Exception as e:  # noqa: BLE001 - deliberately broad: we still want partial results saved
            scan_error = f"{type(e).__name__}: {e}"
            print(f"Archive scan stopped early after exhausting retries: {scan_error}\n"
                 "Writing partial results; re-run the same command to resume "
                 "(the catalog and already-downloaded videos are both kept).", flush=True)
        finally:
            save_catalog(catalog, catalog_path)
    elapsed = time.perf_counter() - t0
    missing = [v for v in video_ids if v not in found]

    # Convert QA items only for videos we actually got.
    qa_lines, ratios = [], []
    for vid in found:
        for record in by_video[vid]:
            item = convert_item(record, transcripts[vid], LICENSE_NOTE)
            qa_lines.append(json.dumps(to_jsonable(item.qa)))
            if item.evidence_match_ratio is not None:
                ratios.append(item.evidence_match_ratio)
    (out / "qa.jsonl").write_text("\n".join(qa_lines) + "\n", encoding="utf-8")

    manifest = {
        "dataset": DATASET_ID, "license": LICENSE_NOTE, "n_videos_requested": len(video_ids),
        "n_videos_downloaded": len(found), "missing_videos": missing,
        "n_questions": len(qa_lines), "n_with_evidence_quote": len(ratios),
        "mean_evidence_match_ratio": sum(ratios) / len(ratios) if ratios else None,
        "n_archive_entries_scanned_this_run": n_scanned, "downloaded_video_bytes": downloaded_bytes,
        "download_seconds": round(elapsed, 1), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scan_error": scan_error, "complete": scan_error is None and not missing,
        "catalog_path": str(catalog_path), "catalog_entries_known": len(catalog.entries),
        "catalog_scan_complete": catalog.complete,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if missing:
        print(f"WARNING: {len(missing)} selected videos were not found in the archive: {missing}")


if __name__ == "__main__":
    main()
