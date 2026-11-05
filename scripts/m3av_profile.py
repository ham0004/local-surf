"""Lecture-study resource profile (auditor P125, "resource characterization"; research log step 97).

Two parts. Neither changes a frozen file, a frozen output or any method.

saved (CPU only): time per stage as recorded by the frozen runs themselves.
  - per question and pool system: the stage timings every pool.json stores (BM25 retrieval, Head A, frame
    decoding, MobileCLIP), dev and test;
  - per answer call: the test QA records' answer seconds and token counts, by condition (reused replies excluded);
  - per lecture: the 2 s feature pass, from the run logs (whole seconds);
  - the test QA session: model load and the torch memory peaks.

gpu (dev only, the first N questions in the hash order of m3av_qa --limit): the stages measured again with the
frozen code, outputs written under runs/m3av/profile/ and checked against the saved ones.
  - features (the lectures of those questions), the R5 scoring (all dev, the frozen setting), the pools of S1, S2m
    and R5, then the answer model on S1, S2m, R5, S1t, q_only and t_only;
  - for each stage: wall time, torch peak allocated and reserved, and the process's dedicated (on-card) and shared
    (system RAM) GPU memory from the Windows counters, sampled each second;
  - the answers twice: as frozen, then with the torch allocator capped below the card's free memory. The replies
    must be identical; any out-of-memory under the cap is counted.

    python scripts/m3av_profile.py saved                 # reports/m3av/profile_saved.{json,md}
    python scripts/m3av_profile.py gpu --limit 100       # reports/m3av/profile_gpu.{json,md}; one GPU job at a time
    python scripts/m3av_profile.py report                # re-render profile_gpu.{json,md} from the saved run
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from m3av_freeze import SYSTEMS  # noqa: E402

RUN, OUT, PROF = Path("runs/v3_baseline"), Path("reports/m3av"), Path("runs/m3av/profile")
POOL_KEYS = ("retrieval", "head_a", "decode", "clip")
PROFILE_SYSTEMS = ("S1", "S2m", "R5")
ANSWER_CONDITIONS = ("S1", "S2m", "R5", "S1t", "q_only", "t_only")
FEATURE_LINE = re.compile(r"^\d+/\d+ (\S+): (\d+) frames, ([\d.]+) min video, (\d+) s$")


def stats(x) -> dict:
    x = np.asarray(list(x), float)
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "mean": round(float(x.mean()), 3), "median": round(float(np.median(x)), 3),
            "p95": round(float(np.percentile(x, 95)), 3), "max": round(float(x.max()), 3)}


def pool_timings(split: str, pool: str) -> dict:
    rows = [json.loads((d / "pool.json").read_text(encoding="utf-8"))["timings"]
            for d in sorted((RUN / f"m3av_{split}" / pool / "pools").iterdir()) if (d / "pool.json").exists()]
    out = {k: stats(r.get(k, 0.0) for r in rows) for k in POOL_KEYS}
    out["total"] = stats(sum(r.get(k, 0.0) for k in POOL_KEYS) for r in rows)
    return out


def feature_log(path: Path) -> dict:
    rows = [FEATURE_LINE.match(x.strip()) for x in path.read_text(encoding="utf-8", errors="replace").splitlines()]
    rows = [m for m in rows if m]
    sec = [int(m.group(4)) for m in rows]
    return {"lectures": len(rows), "seconds_per_lecture": stats(sec), "seconds_per_lecture_after_first": stats(sec[1:]),
            "frames_per_lecture": stats(int(m.group(2)) for m in rows),
            "minutes_per_lecture": stats(float(m.group(3)) for m in rows), "total_seconds": int(sum(sec))}


def stage_saved(_a) -> None:
    rep = {"what": "time per stage recorded by the frozen runs (saved files only; no model run)",
           "pools": {}, "answers_test": {}, "features": {}}
    for split in ("dev", "test"):
        rep["pools"][split] = {k: dict(pool=p, **pool_timings(split, p)) for k, (p, _s, _w, _t) in SYSTEMS.items()
                               if (RUN / f"m3av_{split}" / p / "pools").exists()}
    rec = [json.loads(x) for x in Path("runs/m3av/qa/test/records.jsonl").read_text(encoding="utf-8").splitlines()
           if x.strip()]
    for c in dict.fromkeys(r["condition"] for r in rec):
        rs = [r for r in rec if r["condition"] == c]
        new = [r for r in rs if r["reused_from"] is None and r["failed"] is None]
        rep["answers_test"][c] = {"records": len(rs), "answered_here": len(new), "reused": len(rs) - len(new),
                                  "failed": sum(r["failed"] is not None for r in rs),
                                  "answer_seconds": stats(r["answer_seconds"] for r in new),
                                  "visual_tokens": stats(r["visual_tokens"] for r in new),
                                  "prompt_text_tokens": stats(r["prompt_text_tokens"] for r in new),
                                  "generated_tokens": stats(r["generated_tokens"] for r in new),
                                  "select_seconds": stats(r["select_seconds"] for r in rs),
                                  "text_seconds": stats(r["text_seconds"] for r in rs)}
    rep["features"] = {"dev": feature_log(Path("runs/m3av_dev.log")), "test": feature_log(Path("runs/m3av/test_run.log"))}
    rep["qa_test_sessions"] = [json.loads(x) for x in Path("runs/m3av/qa/test/sessions.jsonl").read_text(
        encoding="utf-8").splitlines() if x.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "profile_saved.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    md = ["# Lecture study: time per stage from the frozen runs' own records",
          "", "Saved files only; no model was run. Seconds. Pools: per question; answers: per new answer call.", ""]
    for split in ("dev", "test"):
        md += [f"## Pool building, {split}", "",
               "| System | pool | retrieval | Head A | decode | MobileCLIP | total (mean) | total (median) | total (p95) |",
               "|---|---|---|---|---|---|---|---|---|"]
        for k, r in rep["pools"][split].items():
            md.append(f"| {k} | {r['pool']} | " + " | ".join(f"{r[x]['mean']:.3f}" for x in POOL_KEYS)
                      + f" | {r['total']['mean']:.2f} | {r['total']['median']:.2f} | {r['total']['p95']:.2f} |")
        md.append("")
    md += ["## Answer model (Qwen3-VL-4B), test", "",
           "| Condition | new calls | reused | seconds (mean) | median | p95 | visual tokens | text tokens |",
           "|---|---|---|---|---|---|---|---|"]
    for c, r in rep["answers_test"].items():
        s = r["answer_seconds"]
        md.append(f"| {c} | {r['answered_here']} | {r['reused']} | {s.get('mean', 0):.2f} | {s.get('median', 0):.2f} | "
                  f"{s.get('p95', 0):.2f} | {r['visual_tokens'].get('mean', 0):.0f} | "
                  f"{r['prompt_text_tokens'].get('mean', 0):.0f} |")
    md += ["", "## Feature pass (MobileCLIP every 2 s, whole lecture)", ""]
    for split, r in rep["features"].items():
        md.append(f"- {split}: {r['lectures']} lectures, {r['frames_per_lecture']['mean']:.0f} frames and "
                  f"{r['minutes_per_lecture']['mean']:.1f} min each; {r['seconds_per_lecture_after_first']['mean']:.1f} s "
                  f"per lecture after the first (first includes loading), {r['total_seconds']} s in all.")
    s = rep["qa_test_sessions"][-1]
    md += ["", "## Test answer session", "",
           f"- model load {s.get('model_load_seconds')} s; torch peak allocated {s.get('torch_peak_allocated_gb')} GiB, "
           f"peak reserved {s.get('torch_peak_reserved_gb')} GiB, card {s.get('gpu_total_gb')} GiB ({s.get('gpu')})."]
    (OUT / "profile_saved.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


# ---------------------------------------------------------------- gpu


class Counters:
    """The Windows per-process GPU memory counters (dedicated = on the card, shared = system RAM), every second."""

    def __init__(self, pid: int, path: Path) -> None:
        q = subprocess.run(["typeperf", "-qx", "GPU Process Memory"], capture_output=True, text=True).stdout
        self.paths = [x.strip() for x in q.splitlines()
                      if f"(pid_{pid}_" in x and x.strip().endswith(("Dedicated Usage", "Shared Usage"))]
        self.file = path
        self.file.unlink(missing_ok=True)
        self.proc = subprocess.Popen(["typeperf", *self.paths, "-si", "1", "-f", "CSV", "-o", str(path), "-y"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) if self.paths else None

    def stop(self) -> None:
        if self.proc:
            self.proc.terminate()
            self.proc.wait()

    def samples(self) -> list[tuple]:
        """(local time, dedicated GiB, shared GiB), summed over the process's adapters."""
        if not self.file.exists():
            return []
        rows = list(csv.reader(self.file.read_text(encoding="utf-8", errors="replace").splitlines()))
        if not rows:
            return []
        head, out = rows[0], []
        ded = [i for i, h in enumerate(head) if h.endswith("Dedicated Usage")]
        sha = [i for i, h in enumerate(head) if h.endswith("Shared Usage")]
        for r in rows[1:]:
            try:
                t = datetime.datetime.strptime(r[0], "%m/%d/%Y %H:%M:%S.%f")
                out.append((t, sum(float(r[i]) for i in ded) / 2**30, sum(float(r[i]) for i in sha) / 2**30))
            except (ValueError, IndexError):
                continue
        return out


class Stage:
    """Wall time and torch memory peaks of one stage; counter peaks are filled in afterwards from the samples."""

    def __init__(self, name: str, log: list) -> None:
        self.name, self.log = name, log

    def __enter__(self):
        import torch  # noqa: PLC0415

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.t0, self.start = time.perf_counter(), datetime.datetime.now()
        return self

    def __exit__(self, *exc):
        import torch  # noqa: PLC0415

        torch.cuda.synchronize()
        self.log.append({"stage": self.name, "start": self.start.isoformat(), "end": datetime.datetime.now().isoformat(),
                         "seconds": round(time.perf_counter() - self.t0, 2),
                         "torch_peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                         "torch_peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 3)})
        return False


def stage_gpu(a) -> None:  # noqa: C901, PLR0915
    import torch  # noqa: PLC0415

    import m3av_vsi  # noqa: PLC0415
    import v3_baseline_cycle as vbc  # noqa: PLC0415
    import v3_features  # noqa: PLC0415
    from m3av_qa import CONDITIONS, CONFIG, K  # noqa: PLC0415
    from videoqa.answerer import AnswerRequest, make_answerer  # noqa: PLC0415
    from videoqa.config import load_config  # noqa: PLC0415
    from videoqa.v2.candidates import load_pool  # noqa: PLC0415
    from videoqa.v2.encoders import FrozenEncoders  # noqa: PLC0415
    from videoqa.v2.teacher import _as_frame  # noqa: PLC0415

    import shutil  # noqa: PLC0415

    ds = "m3av_dev"
    shutil.rmtree(PROF, ignore_errors=True)        # the pool stage skips existing pools; every run starts empty
    PROF.mkdir(parents=True)
    its = vbc.items(ds)[: a.limit]
    qids = [it.qa.qa_id for it in its]
    torch.cuda.init()
    torch.zeros(1, device="cuda")
    free0, total = torch.cuda.mem_get_info()
    ctr = Counters(os.getpid(), PROF / "counters.csv")
    log: list = []
    checks: dict = {}
    rep = {"what": "dev resource profile with the frozen code (outputs under runs/m3av/profile/, checked against the "
                   "saved ones)", "questions": len(qids), "gpu": torch.cuda.get_device_name(0),
           "card_gib": round(total / 2**30, 2), "free_at_start_gib": round(free0 / 2**30, 2),
           "counter_paths": ctr.paths, "start": datetime.datetime.now().isoformat(timespec="seconds")}
    time.sleep(2)
    try:
        # 1. feature pass for the lectures of these questions
        lectures = sorted({it.qa.video_id for it in its})
        enc = FrozenEncoders(device="cuda", cache_dir="cache/open_clip", use_ocr=False)
        diffs = []
        with Stage(f"features ({len(lectures)} lectures)", log):
            for v in lectures:
                t, e, _d = v3_features.video_features(enc, Path(f"data/m3av_dev/videos/{v}.mp4"), 2.0)
                f = np.load(f"data/m3av_dev/features/{v}.npz")
                same_t = len(t) == len(f["times"]) and np.allclose(t, f["times"])
                diffs.append(float(np.abs(e.astype(np.float32) - f["emb"].astype(np.float32)).max()) if same_t else 9.0)
        checks["features_max_abs_diff"] = round(max(diffs), 4)
        del enc
        torch.cuda.empty_cache()
        # 2. R5 scoring, the frozen setting, all dev questions (the script's own loop; written to the profile folder)
        m3av_vsi.OUT = PROF / "vsi"
        argv, sys.argv = sys.argv, ["m3av_vsi.py", "--split", "dev", "--setting", vbc.VSI_SETTING]
        try:
            with Stage("R5 scoring (514 questions)", log):
                m3av_vsi.main()
        finally:
            sys.argv = argv
        torch.cuda.empty_cache()
        new = json.loads((PROF / "vsi" / f"dev_{vbc.VSI_SETTING}.json").read_text(encoding="utf-8"))
        old = json.loads(Path(f"runs/m3av/vsi/dev_{vbc.VSI_SETTING}.json").read_text(encoding="utf-8"))
        checks["vsi_times_identical"] = sum(new[q] == old[q] for q in old) / len(old)
        # 3. pools of S1, S2m and R5 (the frozen stage, on these questions, into the profile folder)
        vbc.RUN = PROF / "pools"
        vbc.items = lambda _ds: its
        for sysname in PROFILE_SYSTEMS:
            pool = SYSTEMS[sysname][0]
            with Stage(f"pools {sysname} ({pool})", log):
                vbc.stage_pools(argparse.Namespace(dataset=ds, pool=pool))
            torch.cuda.empty_cache()
            same = 0
            for q in qids:
                n, o = load_pool(vbc.pool_dir(ds, pool), q), load_pool(RUN / ds / pool / "pools", q)
                same += [round(c.time_s, 3) for c in n.candidates] == [round(c.time_s, 3) for c in o.candidates]
            checks[f"pool_times_identical_{sysname}"] = same / len(qids)
        vbc.RUN = RUN
        # 4. the answer model on the saved (frozen) pools; pass 1 as frozen, pass 2 with the allocator capped
        enc = FrozenEncoders(device="cuda", cache_dir="cache/open_clip", use_ocr=False)
        qo = dict(zip(qids, enc.embed_texts([it.qa.question + " Options: " + ", ".join(it.qa.options) for it in its]),
                      strict=True))
        del enc
        torch.cuda.empty_cache()
        cfg = load_config(CONFIG)
        reqs = []
        for it in its:
            q = it.qa.qa_id
            for cond in ANSWER_CONDITIONS:
                pool, rule, tp, words = CONDITIONS[cond]
                p = load_pool(RUN / ds / pool / "pools", q)
                frames = sorted(vbc.select(p, rule.format(ds=ds), K, qo[q]), key=lambda c: c.time_s)
                ex = vbc.transcript_for(tp, words, p, it.transcript.segments, frames) if words else []
                reqs.append((cond, q, AnswerRequest(p.question, p.options, ex, [_as_frame(c) for c in frames])))
        model = make_answerer(cfg)
        with Stage("answer model load", log):
            model.ensure_loaded()
        rep["weights_resident_gib"] = round(torch.cuda.memory_allocated() / 2**30, 3)
        cap_gib = None
        calls: dict = {}
        for pas in ("frozen", "capped"):
            if pas == "capped":
                torch.cuda.empty_cache()
                free, _ = torch.cuda.mem_get_info()
                cap_gib = round((torch.cuda.memory_reserved() + free) / 2**30 - a.margin, 2)
                torch.cuda.set_per_process_memory_fraction(cap_gib * 2**30 / total)
            rows = []
            with Stage(f"answers, {pas} ({len(reqs)} calls)", log):
                for cond, q, req in reqs:
                    torch.cuda.reset_peak_memory_stats()
                    t0 = time.perf_counter()
                    try:
                        ans, use = model.answer(req)
                        text, err = ans.text, None
                    except RuntimeError as e:
                        if "out of memory" not in str(e):
                            raise
                        torch.cuda.empty_cache()
                        text, use, err = None, None, "out of memory"
                    rows.append({"condition": cond, "qa_id": q, "seconds": round(time.perf_counter() - t0, 3),
                                 "raw": text, "failed": err,
                                 "visual_tokens": use.visual_tokens if use else None,
                                 "text_tokens": use.text_tokens - use.generated_tokens if use else None,
                                 "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                                 "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 3)})
            calls[pas] = rows
        with open(PROF / "answers.jsonl", "w", encoding="utf-8") as fh:
            for pas, rows in calls.items():
                for r in rows:
                    fh.write(json.dumps(dict(r, pass_=pas)) + "\n")
        rep["cap_gib"] = cap_gib
        checks["capped_replies_identical"] = sum(x["raw"] == y["raw"] for x, y in zip(calls["frozen"], calls["capped"],
                                                                                      strict=True)) / len(reqs)
        checks["capped_out_of_memory"] = sum(r["failed"] is not None for r in calls["capped"])
        checks["frozen_out_of_memory"] = sum(r["failed"] is not None for r in calls["frozen"])
        old = {(r["condition"], r["qa_id"]): r["raw"] for r in map(json.loads, Path(
            "runs/m3av/qa/dev/records.jsonl").read_text(encoding="utf-8").splitlines()) if r.get("raw") is not None}
        both = [(r["raw"], old[(r["condition"], r["qa_id"])]) for r in calls["frozen"] if (r["condition"], r["qa_id"]) in old]
        checks["replies_identical_to_dev_check_records"] = {"n": len(both), "identical": sum(x == y for x, y in both)}
        rep["answers"] = {pas: {c: {"seconds": stats(r["seconds"] for r in rows if r["condition"] == c),
                                    "peak_allocated_gb": stats(r["peak_allocated_gb"] for r in rows if r["condition"] == c),
                                    "peak_reserved_gb": stats(r["peak_reserved_gb"] for r in rows if r["condition"] == c),
                                    "text_tokens": stats(r["text_tokens"] for r in rows
                                                         if r["condition"] == c and r["text_tokens"] is not None)}
                                for c in ANSWER_CONDITIONS} for pas, rows in calls.items()}
    finally:
        time.sleep(2)
        ctr.stop()
    samples = ctr.samples()
    for s in log:
        t0, t1 = datetime.datetime.fromisoformat(s["start"]), datetime.datetime.fromisoformat(s["end"])
        inside = [(d, h) for t, d, h in samples if t0 <= t <= t1]
        s["dedicated_peak_gib"] = round(max(d for d, _ in inside), 3) if inside else None
        s["shared_peak_gib"] = round(max(h for _, h in inside), 3) if inside else None
    rep.update({"stages": log, "checks": checks, "end": datetime.datetime.now().isoformat(timespec="seconds")})
    render(rep)


def render(rep: dict) -> None:
    """Write profile_gpu.{json,md}. The per-call peak reset inside an answer pass also resets that stage's own torch
    peak, so an answer stage's torch peaks are the maximum over its calls (answers.jsonl)."""
    calls = [json.loads(x) for x in (PROF / "answers.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    for s in rep["stages"]:
        pas = next((p for p in ("frozen", "capped") if s["stage"].startswith(f"answers, {p}")), None)
        if pas:
            rows = [r for r in calls if r["pass_"] == pas]
            s["torch_peak_allocated_gb"] = max(r["peak_allocated_gb"] for r in rows)
            s["torch_peak_reserved_gb"] = max(r["peak_reserved_gb"] for r in rows)
            s["torch_peaks_from"] = "maximum over the pass's calls"
    (OUT / "profile_gpu.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    checks = rep["checks"]
    md = ["# Lecture study: dev resource profile (frozen code, measured again)", "",
          f"{rep['gpu']}, {rep['card_gib']} GiB, {rep['free_at_start_gib']} GiB free at start (the rest is the desktop and "
          f"other processes). First {rep['questions']} dev questions in hash order; R5 scoring over all 514 dev "
          "questions. Dedicated = process memory on the card; shared = process GPU memory held in system RAM (Windows "
          "counters, 1 s samples). Answer stages: torch peaks are the maximum over their calls.", "",
          "| Stage | seconds | torch peak allocated (GiB) | torch peak reserved (GiB) | dedicated peak (GiB) | shared peak (GiB) |",
          "|---|---|---|---|---|---|"]
    for s in rep["stages"]:
        md.append(f"| {s['stage']} | {s['seconds']} | {s['torch_peak_allocated_gb']} | {s['torch_peak_reserved_gb']} | "
                  f"{s['dedicated_peak_gib']} | {s['shared_peak_gib']} |")
    md += ["", f"Answer weights resident after load: {rep.get('weights_resident_gib')} GiB. Allocator cap in the capped "
           f"pass: {rep.get('cap_gib')} GiB.", "", "## Per answer call", "",
           "| Pass | Condition | seconds (mean) | p95 | peak allocated (max, GiB) | peak reserved (max, GiB) | text tokens |",
           "|---|---|---|---|---|---|---|"]
    for pas, by in rep.get("answers", {}).items():
        for c, r in by.items():
            md.append(f"| {pas} | {c} | {r['seconds']['mean']:.2f} | {r['seconds']['p95']:.2f} | "
                      f"{r['peak_allocated_gb']['max']:.2f} | {r['peak_reserved_gb']['max']:.2f} | "
                      f"{r['text_tokens'].get('mean', 0):.0f} |")
    md += ["", "## Checks", "", *[f"- {k}: {v}" for k, v in checks.items()]]
    (OUT / "profile_gpu.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def stage_report(_a) -> None:
    render(json.loads((OUT / "profile_gpu.json").read_text(encoding="utf-8")))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=("saved", "gpu", "report"))
    p.add_argument("--limit", type=int, default=60, help="gpu: the first N dev questions in hash order")
    p.add_argument("--margin", type=float, default=0.5, help="gpu: GiB left free below the card's free memory")
    a = p.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    {"saved": stage_saved, "gpu": stage_gpu, "report": stage_report}[a.action](a)


if __name__ == "__main__":
    main()
