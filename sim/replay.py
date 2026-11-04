#!/usr/bin/env python3
"""Replay one human demo on the G1 + Dex3 in MuJoCo (full physics) and report success.

Retargeting, in short:
  * Scene: the table frame from make_demos.py is rotated into the robot's frame
    (you faced the camera; the robot faces the table), and the tub's start is
    placed at TUB_START in front of the robot.
  * Wrist:  approach = your palm's path relative to the grasp point;
            grasp    = robot grasp pose around the tub, approach direction taken from your hand;
            carry    = follows the tub's measured path;  release = open, then back off.
  * Fingers: your finger bend (index -> index, middle/ring/pinky -> middle, thumb -> thumb)
    mapped onto Dex3's joint ranges; while holding, the fingers squeeze (torque-limited).
  * Arm + waist joints from inverse kinematics (mink), executed by the robot's
    position actuators; the tub is a free physical object.

  python sim/replay.py demos/VID_20261102_004944.npz [--video out.mp4]
"""
import argparse
import math
import os
import sys

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mink  # noqa: E402
import scene  # noqa: E402

R90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])   # table frame -> robot frame
TUB_START = np.array([0.36, -0.12])                     # tub start on the table, robot frame
G = np.array([0.11, 0.040, 0.0])     # tub axis in the wrist frame when gripped (calibrated in sim)
GRASP_ROLL = 0.0
GRASP_DZ = 0.025                      # grip above the tub's middle so the lower finger clears the table
OPEN = np.zeros(7)
CLOSED = np.array([0.0, 0.72, -1.75, 1.57, 1.75, 1.57, 1.75])
YAW_RANGE = (math.radians(60), math.radians(60))       # reachable approach directions
WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
APPROACH_MAX = 60          # use at most the last 2 s of your approach
CLOSE_DWELL, OPEN_DWELL = 0.5, 0.4   # s
CARRY_SLOW = 1.5      # play the carry 1.5x slower than you did


def grasp_R(yaw, roll=None):
    """Wrist orientation for a side grasp: palm (+y of wrist) faces the tub axis.
    roll tilts the hand about that direction so both Dex3 fingers fit on the short tub."""
    roll = GRASP_ROLL if roll is None else roll
    a = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    z = np.array([0, 0, 1.0])
    R0 = np.stack([np.cross(a, z), a, z], 1)
    c, s = math.cos(roll), math.sin(roll)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    Ra = np.eye(3) * c + s * K + (1 - c) * np.outer(a, a)     # rotation about a
    return Ra @ R0


def flex(K, a, b, c):
    """Bend angle at joint b (0 = straight)."""
    u, v = K[..., b, :] - K[..., a, :], K[..., c, :] - K[..., b, :]
    cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-9)
    return np.arccos(np.clip(cos, -1, 1))


def finger_targets(K):
    """Your finger bend -> Dex3 joint targets (T,7)."""
    idx_mcp, idx_pip = flex(K, 0, 5, 6), flex(K, 5, 6, 7)
    mid_mcp = np.mean([flex(K, 0, 9, 10), flex(K, 0, 13, 14), flex(K, 0, 17, 18)], 0)
    mid_pip = np.mean([flex(K, 9, 10, 11), flex(K, 13, 14, 15), flex(K, 17, 18, 19)], 0)
    th_mcp, th_ip = flex(K, 1, 2, 3), flex(K, 2, 3, 4)
    n = lambda x, full: np.clip(x / full, 0, 1)
    out = np.zeros((len(K), 7))
    out[:, 1] = n(th_mcp, math.radians(50)) * CLOSED[1]
    out[:, 2] = n(th_ip, math.radians(60)) * CLOSED[2]
    out[:, 3] = n(idx_mcp, math.radians(80)) * CLOSED[3]
    out[:, 4] = n(idx_pip, math.radians(90)) * CLOSED[4]
    out[:, 5] = n(mid_mcp, math.radians(80)) * CLOSED[5]
    out[:, 6] = n(mid_pip, math.radians(90)) * CLOSED[6]
    return out


def plan(demo, tub_start=None, path_rot=0.0):
    """Per-step wrist targets (p, R), finger targets and phase labels, at the video's fps.

    tub_start: where the tub starts on the table (robot frame xy); default TUB_START.
    path_rot:  rotate your carry (start -> notebook) about the tub's start by this angle (rad).
               Used by augmentation to put the same motion in new places.
    """
    fps, pick, place = float(demo["fps"]), int(demo["pick"]), int(demo["place"])
    c, s_ = math.cos(path_rot), math.sin(path_rot)
    Rp = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]])
    tub = demo["tub_base"] @ Rp.T
    tub_end_h = demo["tub_end"] @ Rp.T
    K, hv = demo["hand_kp"], demo["hand_valid"]
    start = np.array([*(TUB_START if tub_start is None else tub_start), scene.TABLE_H])
    tub_r = start + tub @ R90.T + np.array([0, 0, scene.TUB_H / 2])   # tub centre, robot frame

    tg = int(demo["grasp_frame"]) if int(demo["grasp_frame"]) >= 0 else pick
    palm = K[:, 9]
    a_h = (tub[tg, :2] @ Rp[:2, :2]) - palm[tg, :2]   # un-rotated tub, your frame
    a_r = R90[:2, :2] @ a_h
    yaw_h = math.atan2(a_r[1], a_r[0])
    yaw = min(max(yaw_h, YAW_RANGE[0]), YAW_RANGE[1])
    Rg = grasp_R(yaw)
    lift_g = np.array([0, 0, GRASP_DZ])
    p_grasp = tub_r[pick] + lift_g - Rg @ G
    fingers_h = finger_targets(np.nan_to_num(K))

    P, Rs, F, ph = [], [], [], []
    # approach: your palm path relative to the grasp point (last <= 2 s, valid frames only),
    # turned about the tub axis so it arrives from the side the robot grasps from, and
    # ending at a stand-off 6 cm outside the tub
    back = p_grasp - Rg[:, 1] * 0.06
    dyaw = yaw - yaw_h
    Rz = np.array([[math.cos(dyaw), -math.sin(dyaw), 0], [math.sin(dyaw), math.cos(dyaw), 0],
                   [0, 0, 1.0]])
    a0 = max(0, tg - APPROACH_MAX)
    # skip frames where the single-camera depth estimate for your hand is implausible
    frames = [t for t in range(a0, tg) if hv[t] and np.isfinite(palm[t]).all()
              and np.linalg.norm(palm[t] - palm[tg]) < 0.35]
    if not np.isfinite(palm[tg]).all():
        frames = []
    axis_xy = tub_r[pick, :2]
    for t in frames:
        q = back + Rz @ R90 @ (palm[t] - palm[tg])
        q[2] = max(q[2], back[2])                       # never dip below the grasp height
        if np.linalg.norm(q[:2] - axis_xy) < 0.12:      # close to the tub: pass above it
            q[2] = max(q[2], scene.TABLE_H + scene.TUB_H + 0.05)
        P.append(q)
        Rs.append(Rg)
        F.append(np.clip(fingers_h[t], np.minimum(0, 0.3 * CLOSED), np.maximum(0, 0.3 * CLOSED)))
        ph.append("approach")
    if not P:
        P.append(back + np.array([0, 0, 0.08])); Rs.append(Rg); F.append(OPEN); ph.append("approach")
    last = P[-1].copy()
    for s in np.linspace(0, 1, int(0.4 * fps)):          # go to the stand-off, open hand
        P.append(last * (1 - s) + back * s); Rs.append(Rg); F.append(OPEN); ph.append("approach")
    for _ in range(int(0.3 * fps)):                      # let the arm settle there
        P.append(back); Rs.append(Rg); F.append(OPEN); ph.append("approach")
    for s in np.linspace(0, 1, int(0.7 * fps)):          # move in around the tub
        w = 3 * s**2 - 2 * s**3
        P.append(back * (1 - w) + p_grasp * w); Rs.append(Rg); F.append(OPEN); ph.append("approach")
    for _ in range(int(0.3 * fps)):                      # settle before closing
        P.append(p_grasp); Rs.append(Rg); F.append(OPEN); ph.append("approach")
    # close
    for s in np.linspace(0, 1, int(CLOSE_DWELL * fps)):
        P.append(p_grasp); Rs.append(Rg); F.append(CLOSED * s); ph.append("grasp")
    # carry: your tub path, smoothed and slowed down, pinned to start on the table and end
    # resting on the notebook, with at least 2.5 cm of lift mid-way
    end_tub = start + tub_end_h @ R90.T
    nb_xy, nb_yaw = notebook_pose(start[:2], end_tub[:2])
    carry = carry_path(tub_r[pick:place + 1], tub_r[pick], nb_xy)
    for c in carry:
        P.append(c + lift_g - Rg @ G); Rs.append(Rg); F.append(CLOSED); ph.append("carry")
    p_place = carry[-1] + lift_g - Rg @ G
    for s in np.linspace(0, 1, int(OPEN_DWELL * fps)):
        P.append(p_place); Rs.append(Rg); F.append(CLOSED * (1 - s)); ph.append("release")
    retreat = p_place - Rg[:, 1] * 0.08 + np.array([0, 0, 0.06])
    for s in np.linspace(0, 1, int(0.8 * fps)):
        P.append(p_place * (1 - s) + retreat * s); Rs.append(Rg); F.append(OPEN); ph.append("retreat")
    info = dict(fps=fps, yaw_human=math.degrees(yaw_h), yaw_used=math.degrees(yaw),
                tub_start=start, tub_end_target=end_tub, nb_xy=nb_xy, nb_yaw=nb_yaw)
    return np.array(P), np.array(Rs), np.array(F), ph, info


def carry_path(raw, start_c, nb_xy):
    """Tub-centre path for the carry (robot frame)."""
    raw = gaussian_filter1d(np.asarray(raw, float), 3, axis=0, mode="nearest")
    n = len(raw)
    s = np.linspace(0, 1, n)
    w = 3 * s**2 - 2 * s**3
    z0 = scene.TABLE_H + scene.TUB_H / 2
    z_end = z0 + scene.NOTEBOOK[2] + 0.004
    raw[:, :2] += (start_c[:2] - raw[0, :2]) * (1 - w)[:, None] + (nb_xy - raw[-1, :2]) * w[:, None]
    raw[:, 2] += -(raw[0, 2] - z0) * (1 - w) - (raw[-1, 2] - z_end) * w
    zl = z0 + (z_end - z0) * w
    raw[:, 2] = np.maximum(raw[:, 2], zl + 0.025 * np.sin(np.pi * s))
    k = max(2, int(n * CARRY_SLOW))
    si = np.linspace(0, n - 1, k)
    return np.stack([np.interp(si, np.arange(n), raw[:, j]) for j in range(3)], 1)


def notebook_pose(start_xy, end_xy):
    """Notebook centred where you put the tub down, short side facing the start.
    Pushed further along the move if it would overlap the tub's start position."""
    v = end_xy - start_xy
    dist = np.linalg.norm(v)
    u = v / dist if dist > 1e-6 else np.array([1.0, 0.0])
    yaw = math.atan2(u[1], u[0])            # notebook's short axis (width) along the move
    need = scene.TUB_BODY_R + scene.NOTEBOOK[1] / 2 + 0.01
    c = end_xy if dist >= need else start_xy + u * need
    return c, yaw


def nb_quat(yaw):
    return np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])


def setup(with_camera=True):
    spec = scene.make_spec()
    if with_camera:
        spec.worldbody.add_camera(name="cam", pos=[1.0, -0.6, 1.15],
                                  xyaxes=[0.569, 0.822, 0, -0.371, 0.257, 0.892])
    m = spec.compile()
    return m, mujoco.MjData(m)


def place_objects(m, d, tub_xy, nb_xy, nb_yaw=0.0):
    tq = m.jnt_qposadr[m.joint("tub_free").id]
    d.qpos[tq:tq + 3] = [tub_xy[0], tub_xy[1], scene.TABLE_H + scene.TUB_H / 2 + 0.001]
    d.qpos[tq + 3:tq + 7] = [1, 0, 0, 0]
    mb = m.body("notebook").mocapid[0]
    d.mocap_pos[mb] = [nb_xy[0], nb_xy[1], scene.TABLE_H + scene.NOTEBOOK[2] / 2]
    d.mocap_quat[mb] = nb_quat(nb_yaw)
    mujoco.mj_forward(m, d)


def ik_trajectory(m, P, Rs, q_init):
    """Arm + waist joint targets for every wrist target (warm-started mink IK)."""
    names = scene.RIGHT_ARM + WAIST
    dofs = np.array([m.jnt_dofadr[m.joint(n).id] for n in names])
    qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in names])
    mask = np.zeros(m.nv, bool)
    mask[dofs] = True
    cfg = mink.Configuration(m)
    cfg.update(q_init.copy())
    task = mink.FrameTask("right_wrist_yaw_link", "body", position_cost=1.0, orientation_cost=0.3)
    post = mink.PostureTask(m, cost=1e-2)
    post.set_target_from_configuration(cfg)
    limits = [mink.ConfigurationLimit(m)]
    out, errs = [], []
    for i, (p, R) in enumerate(zip(P, Rs)):
        task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(R), p))
        for _ in range(60 if i == 0 else 8):
            v = mink.solve_ik(cfg, [task, post], 0.05, "daqp", damping=1e-3, limits=limits)
            v[~mask] = 0
            cfg.integrate_inplace(v, 0.05)
        out.append(cfg.q[qadr].copy())
        errs.append(np.linalg.norm(task.compute_error(cfg)[:3]))
    return np.array(out), qadr, np.array(errs)


def run(m, d, qarm, qadr, F, fps, renderer=None, frames=None, record=None):
    """Execute with physics. Position actuators track the planned joints; gravity is
    compensated on the robot's joints (the real G1's low-level controller does the same)."""
    act_of_q = {m.jnt_qposadr[m.actuator_trnid[i, 0]]: i for i in range(m.nu)}
    arm_act = [act_of_q[a] for a in qadr]
    hand_act = [m.actuator(n).id for n in scene.RIGHT_HAND]
    tub_dofs = set(range(m.jnt_dofadr[m.joint("tub_free").id], m.jnt_dofadr[m.joint("tub_free").id] + 6))
    robot = np.array([i for i in range(m.nv) if i not in tub_dofs])
    for i in range(m.nu):
        d.ctrl[i] = d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]]
    steps = max(1, int(round(1.0 / fps / m.opt.timestep)))

    def step_block(n):
        for _ in range(n):
            d.qfrc_applied[robot] = d.qfrc_bias[robot]
            mujoco.mj_step(m, d)

    q0 = d.qpos[qadr].copy()
    for s in np.linspace(0, 1, int(1.5 * fps)):        # move from rest to the first target
        d.ctrl[arm_act] = q0 * (1 - s) + qarm[0] * s
        d.ctrl[hand_act] = F[0]
        step_block(steps)
    for k in range(len(qarm)):
        d.ctrl[arm_act] = qarm[k]
        d.ctrl[hand_act] = F[k]
        step_block(steps)
        if record is not None:
            record.append(dict(qpos=d.qpos.copy(), qvel=d.qvel.copy(), ctrl_arm=qarm[k].copy(),
                               ctrl_hand=F[k].copy()))
        if renderer is not None and k % 2 == 0:
            renderer.update_scene(d, camera="cam")
            frames.append(renderer.render().copy())
    step_block(int(0.5 / m.opt.timestep))             # let it settle


def success(m, d, nb_xy, nb_yaw=0.0):
    tub = d.body("tub")
    p = tub.xpos.copy()
    up = tub.xmat.reshape(3, 3)[:, 2] @ np.array([0, 0, 1])
    c, s_ = math.cos(-nb_yaw), math.sin(-nb_yaw)
    rel = np.array([[c, -s_], [s_, c]]) @ (p[:2] - nb_xy)
    nbx, nby, nbt = scene.NOTEBOOK[1] / 2, scene.NOTEBOOK[0] / 2, scene.NOTEBOOK[2]
    on_nb = abs(rel[0]) < nbx and abs(rel[1]) < nby
    height_ok = abs(p[2] - (scene.TABLE_H + nbt + scene.TUB_H / 2)) < 0.01
    upright = up > math.cos(math.radians(20))
    return bool(on_nb and height_ok and upright), dict(
        tub=np.round(p, 3), upright=round(float(up), 3), on_notebook=bool(on_nb),
        height_ok=bool(height_ok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("demo")
    ap.add_argument("--video", default=None)
    args = ap.parse_args()
    demo = dict(np.load(args.demo))
    m, d = setup()
    P, Rs, F, ph, info = plan(demo)
    place_objects(m, d, info["tub_start"][:2], info["nb_xy"], info["nb_yaw"])
    qarm, qadr, err = ik_trajectory(m, P, Rs, d.qpos.copy())
    print(f"approach yaw: you {info['yaw_human']:.0f} deg -> robot {info['yaw_used']:.0f} deg | "
          f"steps {len(P)} | IK error median {np.median(err)*100:.1f} cm, max {err.max()*100:.1f} cm")
    renderer, frames = (mujoco.Renderer(m, 480, 640), []) if args.video else (None, None)
    run(m, d, qarm, qadr, F, info["fps"], renderer, frames)
    ok, det = success(m, d, info["nb_xy"], info["nb_yaw"])
    print("SUCCESS" if ok else "FAIL", det)
    if args.video and frames:
        import imageio
        imageio.mimsave(args.video, frames, fps=int(info["fps"] / 2))
        print("video ->", args.video)


if __name__ == "__main__":
    main()
