#!/usr/bin/env python3
"""Collect every result into results.md (paste into the README)."""
import glob
import json
import math
import os


def episodes(dirs):
    return sum(len(glob.glob(os.path.join(d, "episodes", "*.npz"))) for d in dirs)


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def ci(k, n):
    """95% confidence interval (Wilson) in percent."""
    if n == 0:
        return 0, 0
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (c - h), 100 * (c + h)


AUG = [f"data/aug_{s}" for s in range(1, 5)]
SCR = [f"data/scripted_{s}" for s in range(1, 5)]
RUNS = [
    ("act_raw", "ACT", "your demos only, original spot", ["data/raw"]),
    ("act_scripted", "ACT", "scripted motions, NO human data", SCR),
    ("act_scripted_half", "ACT", "scripted motions, NO human data (same size as yours)", SCR[:2]),
    ("act_aug", "ACT", "your demos + augmentation", AUG),
    ("act_aug_s1", "ACT (seed 2)", "your demos + augmentation", AUG),
    ("jepa_aug", "ACT-JEPA", "your demos + augmentation", AUG),
    ("jepa_aug_s1", "ACT-JEPA (seed 2)", "your demos + augmentation", AUG),
]
lines = ["## Policy success on unseen tub/notebook layouts", "",
         "| Policy | Training data | Episodes | 100 layouts | 300 layouts (95% CI) |",
         "|---|---|---|---|---|"]
for run, pol, desc, dirs in RUNS:
    e100 = load(f"runs/{run}/eval.json")
    e300 = load(f"runs/{run}/eval300/eval.json")
    if e100 is None and e300 is None:
        continue
    c100 = f"{e100['success']}/{e100['episodes']}" if e100 else "-"
    if e300:
        lo, hi = ci(e300["success"], e300["episodes"])
        c300 = f"**{100*e300['success']/e300['episodes']:.0f}%** ({lo:.0f}-{hi:.0f}%)"
    else:
        c300 = "-"
    lines.append(f"| {pol} | {desc} | {episodes(dirs)} | {c100} | {c300} |")
lines += ["", "## World-model probe (frozen encoder -> future joint states, held-out episodes)", "",
          "| Policy | RMSE (rad) | ATE (rad) |", "|---|---|---|"]
for run, pol, _, _ in RUNS:
    p = load(f"runs/{run}/probe.json")
    if p:
        lines.append(f"| {pol} | {p['rmse']:.4f} | {p['ate']:.4f} |")
open("results.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
