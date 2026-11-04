#!/usr/bin/env python3
"""Extract the working hand's landmarks from phone videos.

The hand is followed by POSITION from frame to frame, not by MediaPipe's
left/right label (that label flickers on gripping or side-on hands, which
creates fake gaps). The track that moves the most and is seen the longest is
taken as the working hand; an idle resting hand is ignored.

  pip install mediapipe opencv-python numpy scipy
  python extract_hand.py VIDEOS_DIR OUT_DIR --hand right
  python extract_hand.py VIDEOS_DIR OUT_DIR --hand right --overlay   # also write check videos

Saved per clip (OUT_DIR/<subfolder>/<name>.npz):
  lm2d   [T,21,3]  image-normalised landmarks (x,y in 0..1, z relative to wrist), smoothed
  world  [T,21,3]  metric landmarks in metres, origin near the hand centre, smoothed
  valid  [T]       True where a real detection exists or a short gap was interpolated
  det    [T]       True where the hand was really detected
  fps, size        video fps and (width, height) after applying phone rotation
  hand             'right' / 'left' / 'unknown' (from --hand)
  mp_label         MediaPipe's majority label on the tracked hand (for checking mirroring)
"""
import argparse
import csv
import urllib.request
from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision
from scipy.signal import savgol_filter

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
MAX_GAP = 10      # frames; longer gaps are marked invalid, not interpolated
MIN_DET = 0.85    # clip is "usable" if >= 85% of frames have a real detection
# a detection joins a track if its wrist is within this distance (image units, 0..1)
# of the track's last wrist; the allowance grows while the hand is unseen
JUMP0, JUMP_PER_FRAME, JUMP_MAX = 0.12, 0.01, 0.5
CHAINS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
          [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]


def make_landmarker(model_path, conf):
    opts = vision.HandLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=conf,
        min_hand_presence_confidence=conf,
        min_tracking_confidence=conf,
    )
    return vision.HandLandmarker.create_from_options(opts)


def open_video(path):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)   # apply the phone's rotation metadata
    return cap


def collect(path, model_path, conf):
    """All hand detections per frame, regardless of label."""
    cap = open_video(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    lm = make_landmarker(model_path, conf)  # fresh per clip: timestamps must increase
    frames, size, i = [], None, 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if size is None:
            size = (bgr.shape[1], bgr.shape[0])
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        res = lm.detect_for_video(img, int(i * 1000 / fps))
        dets = []
        for k, h in enumerate(res.handedness):
            dets.append(dict(
                label=h[0].category_name,
                lm2d=np.array([[p.x, p.y, p.z] for p in res.hand_landmarks[k]]),
                world=np.array([[p.x, p.y, p.z] for p in res.hand_world_landmarks[k]]),
            ))
        frames.append(dets)
        i += 1
    cap.release()
    lm.close()
    return frames, fps, size


def build_tracks(frames):
    """Greedy nearest-wrist association of detections into tracks."""
    tracks = []
    for t, dets in enumerate(frames):
        pairs = []
        for di, d in enumerate(dets):
            w = d["lm2d"][0, :2]
            for ti, tr in enumerate(tracks):
                thr = min(JUMP0 + JUMP_PER_FRAME * (t - tr["last_t"]), JUMP_MAX)
                dist = float(np.linalg.norm(w - tr["last"]))
                if dist < thr:
                    pairs.append((dist, di, ti))
        used_d, used_t = set(), set()
        for dist, di, ti in sorted(pairs):
            if di in used_d or ti in used_t:
                continue
            tr = tracks[ti]
            tr["t"].append(t)
            tr["dets"].append(dets[di])
            tr["last"] = dets[di]["lm2d"][0, :2]
            tr["last_t"] = t
            used_d.add(di)
            used_t.add(ti)
        for di, d in enumerate(dets):
            if di not in used_d:
                tracks.append(dict(t=[t], dets=[d], last=d["lm2d"][0, :2], last_t=t))
    return tracks


def pick_track(tracks):
    """Working hand = seen often AND moves a lot (an idle hand barely moves)."""
    def score(tr):
        w = np.array([d["lm2d"][0, :2] for d in tr["dets"]])
        extent = float(np.linalg.norm(w.max(0) - w.min(0)))
        return len(tr["t"]) * (0.05 + extent)
    return max(tracks, key=score) if tracks else None


def gap_runs(valid):
    """Lengths of consecutive False runs."""
    runs, n = [], 0
    for v in valid:
        if v:
            if n:
                runs.append(n)
            n = 0
        else:
            n += 1
    if n:
        runs.append(n)
    return runs


def fill_and_smooth(x, det):
    """Interpolate short gaps, mark long ones invalid, smooth with Savitzky-Golay."""
    T = len(x)
    idx = np.where(det)[0]
    valid = det.copy()
    if len(idx) < 2:
        return x, np.zeros(T, bool)
    t = np.arange(T)
    flat = x.reshape(T, -1).copy()
    for c in range(flat.shape[1]):
        flat[:, c] = np.interp(t, idx, flat[idx, c])
    run_start = None
    for i in range(T + 1):
        missing = i < T and not det[i]
        if missing and run_start is None:
            run_start = i
        if not missing and run_start is not None:
            inside = run_start > idx[0] and i - 1 < idx[-1]
            if inside and (i - run_start) <= MAX_GAP:
                valid[run_start:i] = True
            run_start = None
    win = 9 if T >= 9 else (T // 2) * 2 - 1
    if win >= 5:
        flat = savgol_filter(flat, win, 2, axis=0)
    return flat.reshape(x.shape), valid


def process(path, model_path, conf):
    frames, fps, size = collect(path, model_path, conf)
    T = len(frames)
    if T == 0:
        return None
    tr = pick_track(build_tracks(frames))
    L2 = np.full((T, 21, 3), np.nan)
    LW = np.full((T, 21, 3), np.nan)
    det = np.zeros(T, bool)
    if tr is not None:
        for t, d in zip(tr["t"], tr["dets"]):
            L2[t], LW[t], det[t] = d["lm2d"], d["world"], True
        lab, n = Counter(d["label"] for d in tr["dets"]).most_common(1)[0]
        agree = n / len(tr["dets"])
    else:
        lab, agree = "none", 0.0
    L2s, valid = fill_and_smooth(L2, det)
    LWs, _ = fill_and_smooth(LW, det)
    return dict(lm2d=L2s, world=LWs, valid=valid, det=det, fps=fps, size=size,
                mp_label=lab, agree=agree, frames=frames)


def write_overlay(path, r, dest):
    """Check video: green = detected, yellow = interpolated, red = missing, grey = other hand."""
    cap = open_video(path)
    W, H = r["size"]
    vw = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), r["fps"], (W, H))
    t = 0
    while True:
        ok, img = cap.read()
        if not ok or t >= len(r["det"]):
            break
        for d in r["frames"][t]:
            for c in CHAINS:
                pts = (d["lm2d"][c, :2] * [W, H]).astype(int)
                cv2.polylines(img, [pts], False, (150, 150, 150), 1)
        if r["valid"][t]:
            col = (0, 220, 0) if r["det"][t] else (0, 220, 220)
            for c in CHAINS:
                pts = (r["lm2d"][t][c, :2] * [W, H]).astype(int)
                cv2.polylines(img, [pts], False, col, 3)
            status = "tracked" if r["det"][t] else "interpolated"
        else:
            col, status = (0, 0, 255), "MISSING"
        cv2.putText(img, f"{t:4d} {status}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
        vw.write(img)
        t += 1
    cap.release()
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--hand", choices=["right", "left", "unknown"], default="unknown",
                    help="which of your hands did the task (stored with the data)")
    ap.add_argument("--conf", type=float, default=0.3,
                    help="MediaPipe confidence thresholds (lower = fewer dropouts)")
    ap.add_argument("--overlay", action="store_true",
                    help="write a check video per clip to OUT_DIR/overlays/")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "hand_landmarker.task"
    if not model_path.exists():
        print("downloading hand model ...")
        urllib.request.urlretrieve(MODEL_URL, model_path)

    root = Path(args.videos_dir)
    vids = sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".m4v"})
    if not vids:
        raise SystemExit(f"no .mp4/.mov/.m4v files found under {root}")
    rows = []
    for p in vids:
        rel = p.relative_to(root)          # e.g. purple/clip01.mp4
        dest = out / rel.parent            # keeps the purple/ white/ structure
        dest.mkdir(parents=True, exist_ok=True)
        r = process(p, model_path, args.conf)
        if r is None:
            print(f"{str(rel):40s} could not be read")
            rows.append([str(rel), 0, "0%", 0, "", "", "NO"])
            continue
        np.savez(dest / f"{p.stem}.npz", lm2d=r["lm2d"], world=r["world"],
                 valid=r["valid"], det=r["det"], fps=r["fps"], size=np.array(r["size"]),
                 hand=args.hand, mp_label=r["mp_label"])
        if args.overlay:
            od = out / "overlays" / rel.parent
            od.mkdir(parents=True, exist_ok=True)
            write_overlay(p, r, od / f"{p.stem}_overlay.mp4")
        det = r["det"]
        gaps = gap_runs(det)
        rate = float(det.mean())
        longest = max(gaps) if gaps else 0
        usable = rate >= MIN_DET and longest <= MAX_GAP
        lab = f"{r['mp_label']} {r['agree']:.0%}"
        rows.append([str(rel), len(det), f"{rate:.0%}", longest, lab, args.hand,
                     "yes" if usable else "NO"])
        print(f"{str(rel):40s} frames={len(det):4d} detected={rate:4.0%} "
              f"longest_gap={longest:3d} mp_label={lab:10s} usable={'yes' if usable else 'NO'}")

    with open(out / "report.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["clip", "frames", "detected", "longest_gap", "mp_label", "hand", "usable"])
        wr.writerows(rows)
    print(f"\n{sum(r[6] == 'yes' for r in rows)}/{len(rows)} clips usable -> {out/'report.csv'}")


if __name__ == "__main__":
    main()
