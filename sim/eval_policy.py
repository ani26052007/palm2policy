#!/usr/bin/env python3
"""Test a trained policy on random tub/notebook layouts it has never seen.

The policy drives the robot on its own (closed loop): every step it sees the
current state and outputs joint targets. Actions use ACT's temporal ensembling
(each step averages the overlapping predictions of recent chunks).

  python sim/eval_policy.py runs/act_aug/policy.pt --episodes 100 [--videos 3]
"""
import argparse
import glob
import json
import math
import os
import sys

import mujoco
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import augment  # noqa: E402
import replay  # noqa: E402
# load ACT from the main folder's train_act.py explicitly (a stray older copy elsewhere
# on the path must not be picked up)
import importlib.util  # noqa: E402
_TA = os.path.join(os.path.dirname(HERE), "train_act.py")
_spec = importlib.util.spec_from_file_location("train_act", _TA)
_train_act = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train_act)
ACT = _train_act.ACT

MAX_STEPS = 330          # 11 s at 30 Hz (demos take 6-9 s)
FPS = 30.0


def load_policy(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])
    model = ACT(cfg.pop("obs_dim"), cfg.pop("act_dim"), **cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    st = {k: np.array(v, dtype=np.float32) for k, v in ck["stats"].items()}
    return model, st


def run_episode(model, st, robot, start, nb_xy, nb_yaw, m_ens=0.01, frames=None, renderer=None):
    robot.reset(start, nb_xy, nb_yaw)
    K = model.chunk
    buf = np.zeros((MAX_STEPS + K, MAX_STEPS, st["act_mean"].shape[0]), np.float32)
    have = np.zeros((MAX_STEPS + K, MAX_STEPS), bool)
    for t in range(MAX_STEPS):
        o = (robot.obs(nb_xy, nb_yaw).astype(np.float32) - st["obs_mean"]) / st["obs_std"]
        with torch.no_grad():
            chunk = model(torch.from_numpy(o)[None])[0][0].numpy()
        chunk = chunk * st["act_std"] + st["act_mean"]
        buf[t:t + K, t] = chunk
        have[t:t + K, t] = True
        idx = np.where(have[t])[0]                 # chunks predicted at steps idx cover step t
        w = np.exp(-m_ens * np.arange(len(idx)))   # oldest prediction weighted most (ACT)
        a = (buf[t, idx] * w[:, None]).sum(0) / w.sum()
        robot.step(a, FPS)
        if renderer is not None and t % 2 == 0:
            renderer.update_scene(robot.d, camera="cam")
            frames.append(renderer.render().copy())
    for _ in range(int(0.5 * FPS)):
        robot.step(a, FPS)
    return replay.success(robot.m, robot.d, nb_xy, nb_yaw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy")
    ap.add_argument("--demos", default="demos")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=12345)      # different from training layouts
    ap.add_argument("--videos", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    model, st = load_policy(args.policy)
    robot = augment.Robot()
    demos = [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(args.demos, "*.npz")))
             if os.path.basename(f)[:-4] not in augment.SKIP]
    rng = np.random.default_rng(args.seed)
    out = args.out or os.path.dirname(args.policy)
    results = []
    renderer = None
    if args.videos:
        m2, _ = replay.setup(True)
        robot.m, robot.d = m2, mujoco.MjData(m2)   # same bodies and joints, plus a camera
        renderer = mujoco.Renderer(m2, 480, 640)
    for i in range(args.episodes):
        demo = demos[rng.integers(len(demos))]
        start, rot = augment.sample_layout(rng, demo)
        c, s = math.cos(rot), math.sin(rot)
        end = start + replay.R90[:2, :2] @ (np.array([[c, -s], [s, c]]) @ demo["tub_end"][:2])
        nb_xy, nb_yaw = replay.notebook_pose(start, end)
        frames = [] if (renderer is not None and i < args.videos) else None
        ok, det = run_episode(model, st, robot, start, nb_xy, nb_yaw, frames=frames,
                              renderer=renderer if frames is not None else None)
        why = "" if ok else ("tipped over" if det["upright"] < 0.94 else
                             "not on notebook" if not det["on_notebook"] else "wrong height")
        results.append(dict(ok=bool(ok), why=why, start=start.tolist(), nb=nb_xy.tolist()))
        print(f"episode {i+1:3d}: {'SUCCESS' if ok else 'FAIL   '} {why}", flush=True)
        if frames:
            import imageio
            imageio.mimsave(os.path.join(out, f"policy_ep{i+1}.mp4"), frames, fps=15)
    n = sum(r["ok"] for r in results)
    fails = {}
    for r in results:
        if not r["ok"]:
            fails[r["why"]] = fails.get(r["why"], 0) + 1
    print(f"\npolicy success: {n}/{len(results)} = {n/len(results):.0%}   failures: {fails}")
    json.dump(dict(success=n, episodes=len(results), failures=fails, results=results),
              open(os.path.join(out, "eval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
