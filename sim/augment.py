#!/usr/bin/env python3
"""Turn your demos into many training episodes by moving the tub and the notebook.

For each of your demos and each random layout:
  * the tub starts at a random spot in the robot's reach,
  * your carry (start -> notebook) is rotated to a random direction,
  * the robot replays your motion there with full physics,
  * only successful episodes are kept (the tub ends upright on the notebook).
Every kept episode records what the robot sensed and what it was told to do,
which is exactly what the policy learns from.

  python sim/augment.py demos data/aug --per-demo 30
  python sim/augment.py demos data/raw --raw          # your demos only, original layout
  python sim/augment.py demos data/scripted --per-demo 30 --scripted   # no human data

Observation (31 numbers per step):
  17 joint angles (right arm 7, waist 3, right hand 7), tub position (3) and up-axis (3),
  notebook centre (2) + direction (cos, sin), 4 touch flags (thumb, index, middle, palm on tub)
Action (17 numbers): target angles for the same 17 joints.
"""
import argparse
import csv
import glob
import math
import os
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay  # noqa: E402
import scene  # noqa: E402

SKIP = {"VID_20261102_004906", "VID_20261102_005039"}
START_X, START_Y = (0.28, 0.42), (-0.20, -0.04)   # tub start region (robot frame, m)
END_X, END_Y = (0.24, 0.48), (-0.34, 0.06)        # notebook must land in here
TOUCH_LINKS = ["right_hand_thumb_2_link", "right_hand_index_1_link",
               "right_hand_middle_1_link", "right_wrist_yaw_link"]
JOINTS = scene.RIGHT_ARM + replay.WAIST + scene.RIGHT_HAND


class Robot:
    """Model + indices used for recording."""

    def __init__(self):
        self.m, self.d = replay.setup(False)
        m = self.m
        self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in JOINTS])
        self.tq = m.jnt_qposadr[m.joint("tub_free").id]
        self.tub_geoms = {m.geom("tub_body").id, m.geom("tub_lid").id}
        self.touch_bodies = [m.body(n).id for n in TOUCH_LINKS]
        act_of_q = {m.jnt_qposadr[m.actuator_trnid[i, 0]]: i for i in range(m.nu)}
        self.act = np.array([act_of_q[a] for a in self.qadr])
        tub_dofs = set(range(m.jnt_dofadr[m.joint("tub_free").id],
                             m.jnt_dofadr[m.joint("tub_free").id] + 6))
        self.robot_dofs = np.array([i for i in range(m.nv) if i not in tub_dofs])
        self.steps = None

    def touch(self):
        m, d = self.m, self.d
        out = np.zeros(len(self.touch_bodies))
        for c in d.contact[:d.ncon]:
            g1, g2 = c.geom1, c.geom2
            if (g1 in self.tub_geoms) == (g2 in self.tub_geoms):
                continue
            b = m.geom_bodyid[g2 if g1 in self.tub_geoms else g1]
            if b in self.touch_bodies:
                out[self.touch_bodies.index(b)] = 1.0
        return out

    def obs(self, nb_xy, nb_yaw):
        d = self.d
        tub_p = d.qpos[self.tq:self.tq + 3]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, d.qpos[self.tq + 3:self.tq + 7])
        up = R.reshape(3, 3)[:, 2]
        return np.concatenate([d.qpos[self.qadr], tub_p, up, nb_xy,
                               [math.cos(nb_yaw), math.sin(nb_yaw)], self.touch()])

    def step(self, action, fps):
        m, d = self.m, self.d
        if self.steps is None:
            self.steps = max(1, int(round(1.0 / fps / m.opt.timestep)))
        d.ctrl[self.act] = action
        for _ in range(self.steps):
            d.qfrc_applied[self.robot_dofs] = d.qfrc_bias[self.robot_dofs]
            mujoco.mj_step(m, d)

    def reset(self, tub_xy, nb_xy, nb_yaw):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        replay.place_objects(m, d, tub_xy, nb_xy, nb_yaw)
        for i in range(m.nu):
            d.ctrl[i] = d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]]
        mujoco.mj_forward(m, d)


def actions_from_plan(robot, P, Rs, F, fps):
    """Joint targets (17) for each step, starting with a move from the rest pose."""
    m, d = robot.m, robot.d
    qarm, qadr, err = replay.ik_trajectory(m, P, Rs, d.qpos.copy())
    q0 = d.qpos[qadr].copy()
    n0 = int(1.5 * fps)
    lead = np.array([q0 * (1 - s) + qarm[0] * s for s in np.linspace(0, 1, n0)])
    arm = np.concatenate([lead, qarm])
    hand = np.concatenate([np.repeat(F[:1], n0, 0), F])
    return np.concatenate([arm, hand], 1), err


def rollout(robot, actions, fps, nb_xy, nb_yaw):
    obs = []
    for a in actions:
        obs.append(robot.obs(nb_xy, nb_yaw))
        robot.step(a, fps)
    for _ in range(int(0.5 * fps)):                     # settle, then judge
        robot.step(actions[-1], fps)
    ok, det = replay.success(robot.m, robot.d, nb_xy, nb_yaw)
    return np.array(obs), ok, det


def scripted_plan(start_xy, end_xy, fps):
    """Baseline with NO human data: straight approach from above, fixed timing,
    straight carry with a 4.5 cm lift, hand fully open / closed."""
    yaw = replay.YAW_RANGE[0]
    Rg = replay.grasp_R(yaw)
    z0 = scene.TABLE_H + scene.TUB_H / 2
    lift_g = np.array([0, 0, replay.GRASP_DZ])
    tub0 = np.array([*start_xy, z0])
    p_grasp = tub0 + lift_g - Rg @ replay.G
    back = p_grasp - Rg[:, 1] * 0.06
    above = back + np.array([0, 0, 0.10])
    P, F = [], []
    def seg(a, b, secs, f):
        for s in np.linspace(0, 1, max(2, int(secs * fps))):
            w = 3 * s**2 - 2 * s**3
            P.append(a * (1 - w) + b * w); F.append(f)
    seg(above, above, 0.3, replay.OPEN)
    seg(above, back, 0.7, replay.OPEN)
    seg(back, back, 0.3, replay.OPEN)
    seg(back, p_grasp, 0.7, replay.OPEN)
    seg(p_grasp, p_grasp, 0.3, replay.OPEN)
    for s in np.linspace(0, 1, int(replay.CLOSE_DWELL * fps)):
        P.append(p_grasp); F.append(replay.CLOSED * s)
    n = int(1.2 * fps)
    s = np.linspace(0, 1, n)
    raw = np.stack([np.linspace(start_xy[0], end_xy[0], n), np.linspace(start_xy[1], end_xy[1], n),
                    z0 + 0.045 * np.sin(np.pi * s)], 1)
    for c in replay.carry_path(raw, tub0, end_xy):
        P.append(c + lift_g - Rg @ replay.G); F.append(replay.CLOSED)
    p_place = P[-1].copy()
    for s in np.linspace(0, 1, int(replay.OPEN_DWELL * fps)):
        P.append(p_place); F.append(replay.CLOSED * (1 - s))
    retreat = p_place - Rg[:, 1] * 0.08 + np.array([0, 0, 0.06])
    seg(p_place, retreat, 0.8, replay.OPEN)
    return np.array(P), np.array([Rg] * len(P)), np.array(F)


def sample_layout(rng, demo):
    for _ in range(50):
        start = np.array([rng.uniform(*START_X), rng.uniform(*START_Y)])
        rot = rng.uniform(-math.pi, math.pi)
        c, s = math.cos(rot), math.sin(rot)
        end_h = np.array([[c, -s], [s, c]]) @ demo["tub_end"][:2]
        end = start + replay.R90[:2, :2] @ end_h
        nb_xy, _ = replay.notebook_pose(start, end)
        if END_X[0] < nb_xy[0] < END_X[1] and END_Y[0] < nb_xy[1] < END_Y[1]:
            return start, rot
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("demo_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--per-demo", type=int, default=30)
    ap.add_argument("--raw", action="store_true", help="your demos only, original layout")
    ap.add_argument("--scripted", action="store_true", help="baseline without human data")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = os.path.join(args.out_dir, "episodes")
    os.makedirs(out, exist_ok=True)
    robot = Robot()
    files = [f for f in sorted(glob.glob(os.path.join(args.demo_dir, "*.npz")))
             if os.path.basename(f)[:-4] not in SKIP]
    rows, n_try, t0 = [], 0, time.time()
    for f in files:
        name = os.path.basename(f)[:-4]
        demo = dict(np.load(f))
        fps = float(demo["fps"])
        jobs = [(replay.TUB_START, 0.0)] if args.raw else []
        while len(jobs) < (1 if args.raw else args.per_demo):
            st, rot = sample_layout(rng, demo)
            if st is None:
                break
            jobs.append((st, rot))
        kept = 0
        for st, rot in jobs:
            n_try += 1
            P, Rs, F, ph, info = replay.plan(demo, tub_start=st, path_rot=rot)
            nb_xy, nb_yaw = info["nb_xy"], info["nb_yaw"]
            if args.scripted:
                P, Rs, F = scripted_plan(st, nb_xy, fps)
            robot.reset(st, nb_xy, nb_yaw)
            acts, err = actions_from_plan(robot, P, Rs, F, fps)
            if err.max() > 0.01:
                continue
            robot.reset(st, nb_xy, nb_yaw)
            obs, ok, det = rollout(robot, acts, fps, nb_xy, nb_yaw)
            if not ok:
                continue
            ep = len(rows)
            np.savez_compressed(os.path.join(out, f"ep_{ep:04d}.npz"), obs=obs.astype(np.float32),
                                act=acts.astype(np.float32), fps=fps, source=name,
                                tub_start=st, nb_xy=nb_xy, nb_yaw=nb_yaw)
            rows.append([ep, name, f"{st[0]:.3f}", f"{st[1]:.3f}", f"{math.degrees(rot):.0f}", len(acts)])
            kept += 1
        print(f"{name}: kept {kept}/{len(jobs)}", flush=True)
    with open(os.path.join(args.out_dir, "index.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["episode", "source_demo", "tub_x", "tub_y", "rotation_deg", "steps"])
        w.writerows(rows)
    print(f"\nkept {len(rows)} successful episodes out of {n_try} tries "
          f"({len(rows)/max(n_try,1):.0%}) in {time.time()-t0:.0f} s -> {args.out_dir}")


if __name__ == "__main__":
    main()
