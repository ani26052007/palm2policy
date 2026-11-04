#!/usr/bin/env python3
"""Replay every demo on the robot and report the success rate.

  python sim/replay_all.py demos [--skip VID_..,VID_..]
"""
import argparse, glob, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("demo_dir")
ap.add_argument("--skip", default="VID_20261102_004906,VID_20261102_005039")
args = ap.parse_args()
skip = set(args.skip.split(",")) if args.skip else set()
rows, t0 = [], time.time()
for f in sorted(glob.glob(os.path.join(args.demo_dir, "*.npz"))):
    name = os.path.basename(f)[:-4]
    if name in skip:
        continue
    demo = dict(np.load(f))
    m, d = replay.setup(False)
    P, Rs, F, ph, info = replay.plan(demo)
    replay.place_objects(m, d, info["tub_start"][:2], info["nb_xy"], info["nb_yaw"])
    qarm, qadr, err = replay.ik_trajectory(m, P, Rs, d.qpos.copy())
    replay.run(m, d, qarm, qadr, F, info["fps"])
    ok, det = replay.success(m, d, info["nb_xy"], info["nb_yaw"])
    why = "" if ok else ("tipped over" if det["upright"] < 0.94 else
                         "not on notebook" if not det["on_notebook"] else "wrong height")
    rows.append((name, ok, why, err.max()))
    print(f"{name}  {'SUCCESS' if ok else 'FAIL   '}  {why:16s} IK max err {err.max()*100:.1f} cm", flush=True)
n = sum(r[1] for r in rows)
print(f"\nreplay success: {n}/{len(rows)}  ({time.time()-t0:.0f} s)")
