"""Human-review page data from the sealed test packets (research log step 95). Reads runs/m3av/review_test;
writes items.json and img/ to the folder given as the only argument (outside the repo: test content)."""
import hashlib, json, re, shutil, sys
from pathlib import Path

ROOT = Path(".")
PK = ROOT / "runs/m3av/review_test"
RAW = ROOT / "data/m3av_raw/dataset_v1.0_noaudio"
OUT = Path(sys.argv[1])
(OUT / "img").mkdir(parents=True, exist_ok=True)


def spans(s):
    return [[float(a), float(b)] for a, b in re.findall(r"([\d.]+)-([\d.]+) s", s)]


items = []
for d in sorted(p for p in PK.iterdir() if p.is_dir()):
    t = (d / "packet.md").read_text(encoding="utf-8")
    num, qa_id = re.match(r"# (\d+) (\S+)", t).groups()
    q = re.search(r"\*\*Question\.\*\* (.+)", t).group(1).strip()
    opts, key = [], None
    for m in re.finditer(r"^(\*\*)?([ABCD])\. (.+?)( \(key\)\*\*)?$", t, re.M):
        if m.group(4):
            key = len(opts)
        opts.append(m.group(3))
    assert len(opts) == 4 and key is not None, d
    ev = re.search(r"\*\*Evidence text\.\*\* (.+)", t).group(1).strip()
    seg = spans(re.search(r"\*\*Source slide segment\.\*\* (.+)", t).group(1))[0]
    sp_line = re.search(r"\*\*Proposed visible spans\.\*\* (.+)", t).group(1)
    total = float(re.search(r"\(([\d.]+) s in all\)", sp_line).group(1))
    frames = [float(x) for x in re.findall(r"frame_\d\.jpg at ([\d.]+) s", t)]
    other = spans(re.search(r"\*\*Other valid occurrences of the slide\.\*\* (.+)", t).group(1))
    tr = [dict(a=float(a), b=float(b), text=x.strip())
          for a, b, x in re.findall(r"^- \[([\d.]+)-([\d.]+)\] (.+)$", t, re.M)]
    lecture = qa_id.split(":")[1]
    url = next((RAW / lecture / "raw").glob("*.ytbUrl")).read_text(encoding="utf-8").strip()
    vid = url.split("v=")[-1][:11]
    h = hashlib.sha256(f"human-review:{qa_id}".encode()).hexdigest()
    items.append(dict(packet=int(num), qa_id=qa_id, lecture=lecture, yt=vid, question=q, options=opts, key=key,
                      evidence=ev, segment=seg, spans=spans(sp_line), spans_total=total, frames=frames, other=other,
                      transcript=tr, _h=h))
    for i in range(1, 4):
        shutil.copyfile(d / f"frame_{i}.jpg", OUT / "img" / f"p{int(num):03d}_f{i}.jpg")
    shutil.copyfile(d / "slide.png", OUT / "img" / f"p{int(num):03d}_slide.png")

assert len(items) == 100 and len({i["qa_id"] for i in items}) == 100
items.sort(key=lambda i: i["_h"])          # random order for the human review, fixed by hash
for n, i in enumerate(items, 1):
    i["item"] = n
    del i["_h"]
(OUT / "items.json").write_text(json.dumps(items, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print(len(items), "items;", sum(1 for _ in (OUT / "img").iterdir()), "images")
print("first five:", [(i["item"], i["packet"], i["qa_id"]) for i in items[:5]])
