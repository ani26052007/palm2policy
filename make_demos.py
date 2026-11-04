#!/usr/bin/env python3
"""Turn each recorded clip into a demo in real table coordinates (metres).

Inputs (from extract_hand.py and track_tub.py):
  HAND_DIR/<clip>.npz   lm2d, world, det, valid, size, fps
  TUB_DIR/<clip>.npz    center, width, valid, pick, place

Camera model: Poco F6 main camera, 78 deg diagonal FOV -> f ~ 988 px for 720x1280 video,
principal point at the image centre.

Table frame (per clip): origin = where the tub's base stood at the start,
x = right (as seen from the camera), y = away from the camera, z = up.

  python make_demos.py hand_out_test tub_out demos
  -> demos/<clip>.npz, demos/summary.csv, demos/overview.png
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import medfilt, savgol_filter

F_PX = 988.0            # focal length in pixels (Poco F6, 720x1280 video)
LID_D = 0.0742          # purple lid diameter (m)
LID_CENTER_H = 0.055    # height of the lid's visible centre above the tub base (m)
NOTEBOOK_H = 0.015      # notebook thickness (m) -- measure and edit
SKIP = {"VID_20261102_005000"}   # lid tracking failed on this clip
FALLBACK_LIFT = 0.045   # m, typical lift of the cleanly tracked clips


def cam_points(tub):
    """Lid centre in camera coordinates (m): X right, Y down, Z forward."""
    W, H = tub["size"]
    c, w = tub["center"], np.asarray(tub["width"], float)
    # Fingers over the lid only ever make it look NARROWER (= falsely far away).
    # Trust a frame's width only if it is close to the local maximum; fill the rest
    # by interpolating 1/width (proportional to depth) from trusted neighbours.
    env = maximum_filter1d(w, size=31, mode="nearest")
    ok = w >= 0.88 * env
    t = np.arange(len(w))
    inv = np.interp(t, t[ok], 1.0 / w[ok]) if ok.sum() >= 2 else 1.0 / w
    if len(inv) >= 9:
        inv = savgol_filter(inv, 9, 2)
    Z = F_PX * LID_D * inv
    return np.stack([(c[:, 0] - W / 2) * Z / F_PX, (c[:, 1] - H / 2) * Z / F_PX, Z], 1)


def rest_positions(P, pick, place):
    s = np.median(P[:max(pick, 1)], 0)
    e = np.median(P[place + 1:], 0) if place + 1 < len(P) else P[-1]
    return s, e


def pitch_from_clip(P, pick, place):
    """Camera pitch (rad, + = looking down) from: end lid is NOTEBOOK_H higher than start lid."""
    s, e = rest_positions(P, pick, place)
    dY, dZ = e[1] - s[1], e[2] - s[2]
    if abs(dZ) < 0.08:
        return None
    R = math.hypot(dY, dZ)
    alpha = math.atan2(dZ, dY)
    x = max(-1.0, min(1.0, -NOTEBOOK_H / R))
    cands = [alpha + math.acos(x), alpha - math.acos(x)]
    cands = [(t + math.pi) % (2 * math.pi) - math.pi for t in cands]
    best = min(cands, key=abs)
    return best if abs(best) < math.radians(45) else None


def to_level(P, th):
    """Camera coords -> level coords (x right, y forward, z up) for pitch th."""
    X, Y, Z = P[..., 0], P[..., 1], P[..., 2]
    return np.stack([X, -Y * math.sin(th) + Z * math.cos(th),
                     -Y * math.cos(th) - Z * math.sin(th)], -1)


def hand_camera_points(hand):
    """Back-project the 21 hand landmarks to camera coordinates using the hand's own size.

    Weak perspective: pixel offsets ~ (f/Z) * metric offsets from MediaPipe's world landmarks.
    """
    W, H = hand["size"]
    px = hand["lm2d"][..., :2] * [W, H]
    wm = hand["world"]
    T = len(px)
    Zh = np.full(T, np.nan)
    for t in range(T):
        if not hand["valid"][t]:
            continue
        a = px[t] - px[t].mean(0)
        b = wm[t, :, :2] - wm[t, :, :2].mean(0)
        den = (b * b).sum()
        if den > 1e-8:
            s = (a * b).sum() / den
            if s > 1:
                Zh[t] = F_PX / s
    ok = ~np.isnan(Zh)
    if ok.sum() >= 9:
        idx = np.where(ok)[0]
        z = medfilt(np.interp(np.arange(T), idx, Zh[idx]), 9)
        Zh = savgol_filter(z, 9, 2) if T >= 9 else z
    Zi = Zh[:, None] + (wm[..., 2] - wm[..., 2].mean(1, keepdims=True))
    P = np.stack([(px[..., 0] - W / 2) * Zi / F_PX, (px[..., 1] - H / 2) * Zi / F_PX, Zi], -1)
    return P, Zh


def palm_frame(K):
    """Wrist position and rotation (columns: x to fingers, y to index side, z = x cross y)."""
    o = K[:, 0]
    x = K[:, 9] - o
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    tmp = K[:, 5] - K[:, 17]
    y = tmp - (tmp * x).sum(-1, keepdims=True) * x
    y /= np.linalg.norm(y, axis=-1, keepdims=True)
    z = np.cross(x, y)
    return o, np.stack([x, y, z], -1)


def build(hand, tub, th, hand_scale):
    pick, place = int(tub["pick"]), int(tub["place"])
    Pt = to_level(cam_points(tub), th)
    s, _ = rest_positions(Pt, pick, place)
    origin = s - np.array([0, 0, LID_CENTER_H])
    tub_base = Pt - np.array([0, 0, LID_CENTER_H]) - origin
    T = len(tub_base)
    end = np.median(tub_base[place + 1:], 0) if place + 1 < T else tub_base[-1]
    # Carry-path sanity: it should stay near the start-end segment and above the table.
    # If fingers hid the lid for too long, replace the carry with a smooth fallback path
    # (same timing, straight-line move, lift of FALLBACK_LIFT) and record that.
    carry = tub_base[pick:place + 1]
    reach = np.linalg.norm(end[:2])
    carry_ok = bool(len(carry) and np.linalg.norm(carry[:, :2], axis=1).max() <= 1.6 * reach + 0.03
                    and carry[:, 2].min() >= -0.02)
    if not carry_ok and place > pick:
        s = np.linspace(0, 1, place - pick + 1)
        mj = 10 * s**3 - 15 * s**4 + 6 * s**5            # minimum-jerk profile
        a, b = tub_base[pick].copy(), end
        a[2] = 0.0
        path = a + (b - a) * mj[:, None]
        path[:, 2] += FALLBACK_LIFT * np.sin(np.pi * s)
        tub_base[pick:place + 1] = path

    Ph, Zh = hand_camera_points(hand)
    # Anchor the hand's depth to the tub's at the grasp (the palm is touching the tub there).
    # MediaPipe's metric hand size is only an average, so its depth can be off per clip.
    Zt = cam_points(tub)[:, 2]
    fr = [i for i in range(max(0, pick - 15), min(len(Zh), pick + 6))
          if hand["det"][i] and not np.isnan(Zh[i])]
    r = float(np.median(Zt[fr] / Zh[fr])) if fr else hand_scale
    if not 0.5 < r < 2.0:
        r = hand_scale
    Ph = Ph * r
    K = to_level(Ph, th) - origin
    valid = hand["valid"] & ~np.isnan(K).any((1, 2))
    wpos, wrot = palm_frame(np.nan_to_num(K))
    det = hand["det"]
    grasp_win = det[max(0, pick - 10):min(T, pick + 11)]
    rel_win = det[place:min(T, place + 31)]
    # palm centre (landmark 9) to tub axis, horizontally, at the frames nearest the grasp
    g = sorted((t for t in range(max(0, pick - 10), min(T, pick + 11)) if det[t]),
               key=lambda t: abs(t - pick))[:5]
    palm_axis = (float(np.median([np.linalg.norm(K[t, 9, :2] - tub_base[t, :2]) for t in g]))
                 if g else float("nan"))
    grasp_frame = g[0] if g else -1
    carry = tub_base[pick:place + 1]
    return dict(
        carry_measured=carry_ok, grasp_frame=grasp_frame,
        fps=float(hand["fps"]), pick=pick, place=place,
        tub_base=tub_base, tub_valid=tub["valid"],
        hand_kp=K, hand_valid=valid, hand_det=det,
        wrist_pos=wpos, wrist_rot=wrot,
        tub_end=end, grasp_seen=bool(grasp_win.any()), release_seen=bool(rel_win.any()),
        palm_to_axis=palm_axis, lift=float(carry[:, 2].max()) if len(carry) else 0.0,
        pitch_deg=math.degrees(th), hand_depth_scale=r,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hand_dir")
    ap.add_argument("tub_dir")
    ap.add_argument("out_dir")
    args = ap.parse_args()
    hd, td, out = Path(args.hand_dir), Path(args.tub_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    clips = []
    for tp in sorted(td.rglob("*.npz")):
        name = tp.stem
        hp = next(hd.rglob(f"{name}.npz"), None)
        if hp is None or name in SKIP:
            continue
        clips.append((name, dict(np.load(hp)), dict(np.load(tp))))

    # 1) one camera pitch for the session: median over clips with a long toward/away move
    ths = [pitch_from_clip(cam_points(t), int(t["pick"]), int(t["place"])) for _, _, t in clips]
    ths = [t for t in ths if t is not None]
    th = float(np.median(ths)) if ths else 0.0
    print(f"camera pitch: {math.degrees(th):.1f} deg (median of {len(ths)} clips)")

    # 2) hand depth scale: at the grasp the palm is beside the tub, so its depth should match
    #    the tub's within a few cm; MediaPipe assumes an average hand size, correct globally.
    ratios = []
    for _, h, t in clips:
        pk = int(t["pick"])
        _, Zh = hand_camera_points(h)
        Zt = cam_points(t)[:, 2]
        fr = [i for i in range(max(0, pk - 5), min(len(Zh), pk + 6))
              if h["det"][i] and not np.isnan(Zh[i])]
        if fr:
            ratios.append(np.median(Zt[fr] / Zh[fr]))
    hand_scale = float(np.median(ratios)) if ratios else 1.0
    print(f"hand depth scale: {hand_scale:.2f} (median tub/hand depth at grasp, {len(ratios)} clips)")

    rows, demos = [], {}
    for name, h, t in clips:
        d = build(h, t, th, hand_scale)
        np.savez(out / f"{name}.npz", **d)
        demos[name] = d
        e = d["tub_end"]
        ok = d["grasp_seen"] and d["palm_to_axis"] < 0.12
        rows.append([name, f"{e[0]*100:.1f}", f"{e[1]*100:.1f}", f"{e[2]*100:.1f}",
                     f"{d['lift']*100:.1f}", f"{d['palm_to_axis']*100:.1f}",
                     d["grasp_seen"], d["release_seen"],
                     "measured" if d["carry_measured"] else "fallback", "yes" if ok else "CHECK"])
        print(f"{name[-6:]} tub end x={e[0]*100:6.1f} y={e[1]*100:6.1f} z={e[2]*100:5.1f} cm | "
              f"lift={d['lift']*100:4.1f} cm | palm-to-tub-axis at grasp="
              f"{d['palm_to_axis']*100:4.1f} cm | carry={'measured' if d['carry_measured'] else 'fallback'} "
              f"release={d['release_seen']} -> {'yes' if ok else 'CHECK'}")
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "end_x_cm", "end_y_cm", "end_z_cm", "lift_cm",
                    "palm_to_axis_cm", "grasp_seen", "release_seen", "carry", "use"])
        w.writerows(rows)
    print(f"\n{sum(r[-1] == 'yes' for r in rows)}/{len(rows)} demos look good -> {out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 5))
        for name, d in demos.items():
            tb, pk, pl = d["tub_base"], d["pick"], d["place"]
            ax[0].plot(tb[pk:pl + 1, 0] * 100, tb[pk:pl + 1, 1] * 100, lw=1)
            ax[0].scatter(tb[pl, 0] * 100, tb[pl, 1] * 100, s=12)
            ax[1].plot(np.arange(pl + 1 - pk), tb[pk:pl + 1, 2] * 100, lw=1)
        ax[0].scatter([0], [0], c="k", marker="x", s=60, label="tub start (all clips)")
        ax[0].set(xlabel="x, right (cm)", ylabel="y, away from camera (cm)",
                  title="Tub carry paths, top view")
        ax[0].axis("equal")
        ax[0].legend()
        ax[1].set(xlabel="frames since pick-up", ylabel="tub base height (cm)",
                  title="Lift profile during carry")
        fig.tight_layout()
        fig.savefig(out / "overview.png", dpi=110)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
