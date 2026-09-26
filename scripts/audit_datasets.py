"""Data-feasibility audit: what do the official dataset sources ACTUALLY expose?

For each source we record, from live API calls (nothing is assumed):
  * repository / dataset-card license and last update
  * top-level file listing (and sizes where the API gives them)
  * whether the Hugging Face dataset is gated (needs an access request)
  * a few REAL annotation rows, when an annotation file is directly fetchable

A code license (e.g. MIT on the repo) does NOT grant rights to third-party
videos; the report keeps those separate.  Video availability (dead YouTube
links etc.) can only be measured by actually attempting downloads, which this
script does not do - see DATA_FEASIBILITY.md for the next step.

Usage:  uv run python scripts/audit_datasets.py --out reports/data_audit.json
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request

GITHUB = {
    "EduVidQA": "sourjyadip/eduvidqa-emnlp25",
    "LongVideoBench": "longvideobench/LongVideoBench",
    "QVHighlights": "jayleicn/moment_detr",
    "NExT-GQA": "doc-doc/NExT-GQA",
    "Video-MME": "MME-Benchmarks/Video-MME",
}
HF_DATASETS = {
    "LongVideoBench": "longvideobench/LongVideoBench",
    "Video-MME": "lmms-lab/Video-MME",
}
# Annotation files we try to sample directly (repo-relative paths; verified by the listing).
SAMPLE_FILES = {
    "QVHighlights": ("jayleicn/moment_detr", "data/highlight_val_release.jsonl"),
}


def _get(url: str, raw: bool = False, max_bytes: int = 200_000):
    req = urllib.request.Request(url, headers={"User-Agent": "videoqa-data-audit"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read(max_bytes)
            return data.decode("utf-8", "replace") if raw else json.loads(data)
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}"}
    except Exception as e:  # network failures are recorded, not hidden
        return {"error": f"{type(e).__name__}: {e}"}


def audit_github(repo: str) -> dict:
    meta = _get(f"https://api.github.com/repos/{repo}")
    if "error" in meta:
        return meta
    listing = _get(f"https://api.github.com/repos/{repo}/contents")
    files = [f"{x['type']}:{x['name']}" for x in listing] if isinstance(listing, list) else listing
    readme = _get(f"https://raw.githubusercontent.com/{repo}/{meta.get('default_branch', 'main')}/README.md", raw=True)
    readme = readme if isinstance(readme, str) else ""
    keywords = [k for k in ("youtube", "google drive", "huggingface", "download", "license", "subtitle", "srt",
                            "transcript", "gated", "request") if k in readme.lower()]
    return {"license": (meta.get("license") or {}).get("spdx_id"), "pushed_at": meta.get("pushed_at"),
            "archived": meta.get("archived"), "default_branch": meta.get("default_branch"),
            "top_level": files, "readme_mentions": keywords}


def audit_hf(dataset: str) -> dict:
    info = _get(f"https://huggingface.co/api/datasets/{dataset}?blobs=true")
    if "error" in info:
        return info
    siblings = info.get("siblings", [])
    total = sum(s.get("size") or 0 for s in siblings)
    exts: dict[str, int] = {}
    for s in siblings:
        ext = s["rfilename"].rsplit(".", 1)[-1] if "." in s["rfilename"] else ""
        exts[ext] = exts.get(ext, 0) + 1
    card = info.get("cardData") or {}
    return {"gated": info.get("gated"), "license": card.get("license"), "sha": info.get("sha"),
            "last_modified": info.get("lastModified"), "n_files": len(siblings),
            "total_gb": round(total / 1e9, 2), "file_types": exts,
            "sample_files": [s["rfilename"] for s in siblings[:15]]}


def sample_rows(repo: str, path: str, branch: str = "main", n: int = 3) -> list | dict:
    text = _get(f"https://raw.githubusercontent.com/{repo}/{branch}/{path}", raw=True)
    if not isinstance(text, str):
        return text
    rows = []
    for line in text.splitlines()[:n]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"unparsed": line[:300]})
    return rows


def csv_summary(url: str, video_col: str, n_rows: int = 2) -> dict:
    """Row count, columns, distinct videos and a couple of real rows from a CSV."""
    import csv
    import io

    text = _get(url, raw=True, max_bytes=50_000_000)
    if not isinstance(text, str):
        return text
    rows = list(csv.DictReader(io.StringIO(text)))
    trim = [{k: (v[:160] + "...") if len(v) > 160 else v for k, v in r.items()} for r in rows[:n_rows]]
    return {"rows": len(rows), "columns": list(rows[0].keys()) if rows else [],
            "distinct_videos": len({r[video_col] for r in rows}), "sample_rows": trim,
            "_video_ids": sorted({r[video_col] for r in rows})}


def youtube_alive(video_ids: list[str], n: int = 30) -> dict:
    """Check availability of up to ``n`` YouTube ids via the public oEmbed
    endpoint (metadata only; nothing is downloaded).  200 = available,
    401/403 = embedding disabled/private, 404 = removed."""
    status: dict[str, int] = {}
    for vid in video_ids[:n]:
        res = _get(f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={vid}&format=json")
        code = "ok" if isinstance(res, dict) and "error" not in res else res.get("error", "?")
        status[code] = status.get(code, 0) + 1
    return {"checked": min(n, len(video_ids)), "status_counts": status}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/data_audit.json")
    a = ap.parse_args()
    report = {"github": {k: audit_github(v) for k, v in GITHUB.items()},
              "huggingface": {k: audit_hf(v) for k, v in HF_DATASETS.items()},
              "samples": {}}
    for name, (repo, path) in SAMPLE_FILES.items():
        branch = report["github"].get(name, {}).get("default_branch", "main") or "main"
        report["samples"][name] = sample_rows(repo, path, branch)

    raw = "https://raw.githubusercontent.com/"
    edu = {split: csv_summary(f"{raw}sourjyadip/eduvidqa-emnlp25/main/data/{split}.csv", col)
           for split, col in (("real_world_test", "id"), ("synthetic_train", "vid_id"), ("synthetic_test", "vid_id"))}
    if "_video_ids" in edu["real_world_test"]:
        edu["real_world_test"]["youtube_oembed"] = youtube_alive(edu["real_world_test"]["_video_ids"])
    nextgqa = {split: csv_summary(f"{raw}doc-doc/NExT-GQA/main/datasets/nextgqa/{split}.csv", "video_id")
               for split in ("train", "val", "test")}
    for d in (edu, nextgqa):
        for v in d.values():
            v.pop("_video_ids", None)
    report["samples"]["EduVidQA"] = edu
    report["samples"]["NExT-GQA"] = nextgqa
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=True)[:6000])


if __name__ == "__main__":
    main()
