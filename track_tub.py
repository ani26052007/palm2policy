#!/usr/bin/env python3
"""Track the purple tub's lid in each video and find when it is picked up and put down.

Why: MediaPipe loses your hand while it is wrapped around the tub. While you hold
it, the hand moves with the tub, so the tub's path stands in for the hand's.
The lid's known diameter (7.42 cm) gives the scale (pixels per cm) in every frame,
which also tells us how far the tub is from the camera.

  python track_tub.py data/purple tub_out            # add --overlay for check videos

Saved per clip (OUT_DIR/<name>.npz):
  center   [T,2]  lid centre in pixels, smoothed
  width    [T]    lid width in pixels, smoothed (= 7.42 cm)
  px_per_cm[T]    width / 7.42
  valid    [T]    lid found (or short gap filled)
  moving   [T]    tub is moving (being carried)
  pick, place     first / last frame of the carry
  fps, size
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import medfilt, savgol_filter

LID_D_CM = 7.42
HSV_LO, HSV_HI = (115, 90, 60), (165, 255, 255)   # purple, tuned on your clips
MAX_GAP = 15
MOVE_CM_PER_S = 4.0       # faster than this = being carried
MIN_MOVE_FRAMES = 5


def lid_blobs(bgr, min_area):
    m = cv2.inRange(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), HSV_LO, HSV_HI)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(m)
    return [dict(area=int(st[i, 4]), w=int(st[i, 2]), bbox=st[i, :4].copy(), c=cen[i].copy())
            for i in range(1, n) if st[i, 4] >= min_area]


def fill(x, ok, max_gap):
    T = len(x)
    idx = np.where(ok)[0]
    valid = ok.copy()
    if len(idx) < 2:
        return x, valid
    t = np.arange(T)
    x = x.copy()
    for c in range(x.shape[1]):
        x[:, c] = np.interp(t, idx, x[idx, c])
    gaps = np.diff(idx)
    for a, g in zip(idx[:-1], gaps):
        if 1 < g <= max_gap + 1:
            valid[a + 1:a + g] = True
    return x, valid


def track(path):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    raw, ok, size, prev = [], [], None, None
    while True:
        good, bgr = cap.read()
        if not good:
            break
        if size is None:
            size = (bgr.shape[1], bgr.shape[0])
            min_area = 0.0008 * size[0] * size[1]
        blobs = lid_blobs(bgr, min_area)
        pick = None
        if prev is None:
            cands = [b for b in blobs if b["c"][1] > 0.2 * size[1]]   # ignore the top of frame
            pick = max(cands, key=lambda b: b["area"]) if cands else None
        else:
            lim = 0.12 * size[0] + 0.01 * size[0] * prev[2]   # allowance grows while lost
            cands = [b for b in blobs if np.linalg.norm(b["c"] - prev[0]) < lim]
            pick = max(cands, key=lambda b: b["area"]) if cands else None
        if pick is not None:
            raw.append([*pick["c"], pick["w"]])
            ok.append(True)
            prev = (pick["c"], pick["w"], 0)
        else:
            raw.append([np.nan, np.nan, np.nan])
            ok.append(False)
            if prev is not None:
                prev = (prev[0], prev[1], prev[2] + 1)
    cap.release()
    raw, ok = np.array(raw), np.array(ok)
    if ok.sum() < 5:
        return None
    x, valid = fill(raw, ok, MAX_GAP)
    # fingers over the lid make it look narrower: robust width via median filter
    w = medfilt(x[:, 2], 9)
    w = np.maximum(w, medfilt(x[:, 2], 21) * 0.9)
    win = 9 if len(x) >= 9 else (len(x) // 2) * 2 - 1
    center = savgol_filter(x[:, :2], win, 2, axis=0) if win >= 5 else x[:, :2]
    width = savgol_filter(w, win, 2) if win >= 5 else w
    ppc = width / LID_D_CM
    # carry phase: tub speed in cm/s above threshold for several frames
    speed = np.r_[0, np.linalg.norm(np.diff(center, axis=0), axis=1)] / ppc * fps
    mov = (speed > MOVE_CM_PER_S) & valid
    moving = np.zeros_like(mov)
    run = 0
    for i, m in enumerate(mov):
        run = run + 1 if m else 0
        if run >= MIN_MOVE_FRAMES:
            moving[i - run + 1:i + 1] = True
    idx = np.where(moving)[0]
    pick, place = (int(idx[0]), int(idx[-1])) if len(idx) else (-1, -1)
    return dict(center=center, width=width, px_per_cm=ppc, valid=valid, det=ok,
                moving=moving, pick=pick, place=place, fps=fps, size=size)


def write_overlay(path, r, dest):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    vw = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), r["fps"], r["size"])
    t = 0
    while True:
        good, img = cap.read()
        if not good or t >= len(r["valid"]):
            break
        if r["valid"][t]:
            c = r["center"][t].astype(int)
            half = int(r["width"][t] / 2)
            col = (0, 165, 255) if r["moving"][t] else (0, 220, 0)
            cv2.circle(img, tuple(c), 6, col, -1)
            cv2.line(img, (c[0] - half, c[1]), (c[0] + half, c[1]), col, 3)
        phase = ("approach" if t < r["pick"] else "CARRY" if t <= r["place"] else "release")
        cv2.putText(img, f"{t:4d} {phase}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (255, 255, 255), 3)
        vw.write(img)
        t += 1
    cap.release()
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--overlay", action="store_true")
    args = ap.parse_args()
    root, out = Path(args.videos_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vids = sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".m4v"})
    rows = []
    for p in vids:
        rel = p.relative_to(root)
        dest = out / rel.parent
        dest.mkdir(parents=True, exist_ok=True)
        r = track(p)
        if r is None:
            print(f"{str(rel):40s} lid not found")
            rows.append([str(rel), 0, "0%", -1, -1, "", ""])
            continue
        np.savez(dest / f"{p.stem}.npz", **{k: v for k, v in r.items()})
        if args.overlay:
            od = out / "overlays" / rel.parent
            od.mkdir(parents=True, exist_ok=True)
            write_overlay(p, r, od / f"{p.stem}_tub.mp4")
        T = len(r["valid"])
        found = f"{r['det'].mean():.0%}"
        moved = (np.linalg.norm(r["center"][r["place"]] - r["center"][r["pick"]])
                 / r["px_per_cm"][r["pick"]]) if r["pick"] >= 0 else 0.0
        rows.append([str(rel), T, found, r["pick"], r["place"], f"{moved:.0f}",
                     f"{r['px_per_cm'].mean():.1f}"])
        print(f"{str(rel):40s} frames={T:4d} lid_found={found:>4s} pick={r['pick']:4d} "
              f"place={r['place']:4d} moved~{moved:3.0f}cm px/cm={r['px_per_cm'].mean():.1f}")
    with open(out / "report.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["clip", "frames", "lid_found", "pick", "place", "moved_cm", "px_per_cm"])
        wr.writerows(rows)


if __name__ == "__main__":
    main()
