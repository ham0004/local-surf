"""Per-frame judgments behind the F4 threshold (protocol amendment 2; auditor P52), and the tally computed from them.

Each sheet in reports/m3av/sheets_* shows a saved slide and 5 frames of its segment. The builder judged every frame
by eye: S = the complete saved slide, P = an earlier build stage or scroll position of it (part not yet shown),
D = a different slide, the speaker, or an intro card. Set 1 was picked with the first (grey) curves, sets 2 and 3
with the edge curves, each set excluding segments already seen; frame times in set 1 are the nearest edge-curve
sample to the time printed on the sheet. Rule checked: own edge correlation >= tau and not below the previous or next
segment's saved slide (calibration compares both neighbours; `apply` skips neighbours that are evidence themselves).

    python scripts/m3av_visibility_calibration.py      # reports/m3av/visibility_calibration.{csv,md}
"""

from __future__ import annotations

import collections
import csv
import json
from pathlib import Path

import numpy as np

CURVES, OUT, TAU = Path("runs/m3av/curves_edge"), Path("reports/m3av"), 0.85
# set -> sheet -> five labels, in the order the frames appear on the sheet
LABELS = {
    "grey_set1": {  # sheet times rounded to whole seconds on the sheet; labels per printed time
        "CHI-0ED44_265.45": {265: "D", 267: "P", 269: "P", 270: "P", 272: "S"},
        "CHI-57172_86": {86: "D", 94: "D", 103: "S", 112: "S", 120: "S"},
        "CHI-67576_276.45": {276: "D", 284: "S", 290: "S", 296: "S", 304: "S"},
        "CHI-6CF4B_13": {13: "D", 20: "S", 27: "S", 35: "S", 42: "S"},
        "CHI-67576_76.45": {76: "D", 77: "S", 78: "S", 79: "S"},
        "CHI-72910_240": {240: "D", 244: "S", 248: "S", 251: "S", 255: "S"},
        "CHI-96A54_175.45": {175: "D", 179: "S", 183: "S", 186: "S", 190: "S"},
        "CHI-D36EE_238": {238: "D", 243: "S", 248: "S", 254: "S", 259: "S"},
        "CHI-EBB6C_188": {188: "D", 190: "S", 192: "S", 193: "S", 195: "S"},
        "MLS-792FC_933": {933: "D", 971: "P", 1008: "S", 1045: "S", 1083: "S"},
        "NIH-04FD0_176": {176: "D", 204: "D", 233: "D", 262: "D", 290: "S"},
        "NIH-04FD0_2434.45": {2434: "D", 2479: "D", 2522: "S", 2565: "S", 2609: "S"},
        "NIH-04FD0_290.46": {290: "D", 347: "P", 403: "P", 460: "D", 516: "S"},
        "NIH-7B4FC_3358.5": {3359: "D", 3371: "S", 3383: "S", 3395: "S", 3407: "S"},
        "NIH-EFBA6_1213.45": {1213: "D", 1226: "D", 1240: "S", 1254: "S", 1266: "S"},
        "NIH-F1FA9_1867.45": {1867: "D", 1882: "S", 1896: "S", 1912: "S", 1926: "S"},
        "Ubi-424F7_245.45": {245: "D", 252: "S", 259: "S", 266: "S", 273: "S"},
        "Ubi-432F6_51": {51: "D", 52: "S", 53: "S", 55: "S", 56: "S"},
        "Ubi-77533_37.45": {37: "D", 40: "S", 43: "S", 45: "S", 48: "S"},
        "Ubi-B1B15_218.5": {219: "D", 232: "S", 245: "S", 257: "S", 270: "S"},
    },
    "edge_set2": {
        "CHI-34D3C_145": "DSSSS", "CHI-3672F_23": "DSSSS", "CHI-F48F3_133.45": "DDDSS", "CHI-F48F3_81.45": "DSSSD",
        "IPP-10817_611.45": "DDDSS", "IPP-10817_770.45": "DSSSD", "IPP-3239F_2177.45": "DSSSS",
        "IPP-3239F_755.45": "DSSSS", "IPP-58EFF_19.6": "DSSSS", "IPP-B6228_2882": "DPSSS", "MLS-2A8E1_33.455": "DPPSP",
        "NIH-16CA2_2738.45": "DSSSD", "NIH-4B17B_1977": "DSSSS", "NIH-95F8B_1168.45": "DSSSS",
        "NIH-95F8B_2441.45": "DSSSS", "NIH-9A5A0_2005.45": "DSSSS", "NIH-9ED9E_1835": "DSSSS", "NIH-F1FA9_2177": "DSSSS",
        "Ubi-10CFE_94.45": "DDSSS", "Ubi-53297_83.45": "DSSSS",
    },
    "edge_set3": {
        "CHI-3672F_173": "DSSSS", "CHI-47918_191.45": "DPPPS", "CHI-77EC8_179": "DPPPS", "CHI-77EC8_62": "DSSSS",
        "CHI-98CF8_150.455": "DSSSS", "CHI-A9D9B_136.45": "DSSSS", "CHI-EFC5E_0": "DDSSS", "IPP-3239F_1504.45": "DSSSS",
        "IPP-B6228_2797": "DPPPS", "NIH-04FD0_3093.46": "DPPSS", "NIH-211B0_2848": "DSSSS", "NIH-211B0_3530.45": "DSSSS",
        "NIH-211B0_390": "DPPDS", "NIH-568B7_893.5": "DPPSS", "NIH-77A41_2046": "DSSSS", "NIH-C7721_2624.45": "DSSSS",
        "NIH-E8189_2563": "DSSSS", "NIH-EC45B_1087": "DPPSS", "Ubi-A73B4_109.45": "DSSSS", "Ubi-B1B15_73.45": "DSSSS",
    },
}


def curve_of(sheet: str) -> list:
    lec, st = sheet.rsplit("_", 1)
    cc = json.loads((CURVES / f"{lec}.json").read_text(encoding="utf-8"))
    return next(v for k, v in cc.items() if abs(float(k) - float(st)) < 0.01)


def frames(name: str, sheet: str, labels) -> list[tuple]:
    curve = curve_of(sheet)
    if isinstance(labels, dict):                     # set 1: nearest edge sample to the printed time
        return [(min(curve, key=lambda x: abs(x[0] - t)), lab) for t, lab in labels.items()]
    picks = [curve[round(i)] for i in np.linspace(0, len(curve) - 1, 5)]   # the sheet's own frame choice
    return list(zip(picks, labels, strict=True))


def main() -> None:
    rows = []
    for name, sheets in LABELS.items():
        for sheet, labels in sheets.items():
            for (t, own, prev, nxt), lab in frames(name, sheet, labels):
                nb = max((x for x in (prev, nxt) if x is not None), default=None)
                rows.append({"set": name, "sheet": sheet, "time": t, "own": own, "neighbour": nb, "label": lab,
                             "counted_visible": own >= TAU and (nb is None or own >= nb)})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "visibility_calibration.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tally = collections.Counter((r["set"], r["label"], r["counted_visible"]) for r in rows)
    lines = [f"# F4 calibration tally (tau = {TAU}, neighbour check)", "",
             "Builder's judgments by eye on selected dev frames; not a population error rate (auditor P52). "
             "Set 3 is the only set not used to choose the measure or tau.", "",
             "| Set | Frame shows | Frames | Counted visible |", "|---|---|---:|---:|"]
    names = {"S": "complete saved slide", "P": "build stage / scroll position", "D": "other slide, speaker or intro"}
    for name in LABELS:
        for lab in "DSP":
            n = tally[(name, lab, True)] + tally[(name, lab, False)]
            lines.append(f"| {name} | {names[lab]} | {n} | {tally[(name, lab, True)]} |")
    (OUT / "visibility_calibration.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    for r in rows:
        if r["label"] == "D" and r["counted_visible"] or r["label"] == "S" and not r["counted_visible"]:
            print("error:", r)


if __name__ == "__main__":
    main()
