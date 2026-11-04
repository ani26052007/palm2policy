"""Build the MuJoCo scene: fixed-base Unitree G1 with Dex3 hands, table, purple tub, notebook."""
import os
import mujoco
import numpy as np

G1_SCENE = os.environ.get("G1_SCENE", "third_party/mujoco_menagerie/unitree_g1/scene_with_hands.xml")

TABLE_H = 0.72          # table top height (m)
TABLE_X = 0.45          # table centre, in front of the robot
NOTEBOOK = (0.153, 0.108, 0.015)   # length, width, thickness (m)
# purple tub: body d=7.86 h=4.6, lid d=7.42 h=1.8 (cm)
TUB_BODY_R, TUB_BODY_H = 0.0393, 0.046
TUB_LID_R, TUB_LID_H = 0.0371, 0.018
TUB_H = TUB_BODY_H + TUB_LID_H

RIGHT_ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
             "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
             "right_wrist_yaw_joint"]
RIGHT_HAND = ["right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
              "right_hand_index_0_joint", "right_hand_index_1_joint",
              "right_hand_middle_0_joint", "right_hand_middle_1_joint"]


def make_spec(g1_scene=G1_SCENE):
    spec = mujoco.MjSpec.from_file(g1_scene)
    pelvis = spec.body("pelvis")
    for j in list(pelvis.joints):            # fix the robot in place: no balancing needed
        spec.delete(j)
    wb = spec.worldbody
    table = wb.add_body(name="table", pos=[TABLE_X, 0, TABLE_H / 2])
    table.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.30, 0.45, TABLE_H / 2],
                   rgba=[0.25, 0.25, 0.28, 1], friction=[1, 0.005, 0.0001])
    nb = wb.add_body(name="notebook", mocap=True, pos=[TABLE_X, 0.15, TABLE_H + NOTEBOOK[2] / 2])
    nb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[NOTEBOOK[1] / 2, NOTEBOOK[0] / 2, NOTEBOOK[2] / 2],
                rgba=[0.9, 0.85, 0.2, 1], friction=[1, 0.005, 0.0001])
    tub = wb.add_body(name="tub", pos=[TABLE_X - 0.1, -0.15, TABLE_H + TUB_H / 2])
    tub.add_freejoint(name="tub_free")
    tub.add_geom(name="tub_body", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                 size=[TUB_BODY_R, TUB_BODY_H / 2, 0], pos=[0, 0, -TUB_H / 2 + TUB_BODY_H / 2],
                 mass=0.20, rgba=[0.45, 0.2, 0.65, 1], friction=[1.2, 0.005, 0.0001], condim=4)
    tub.add_geom(name="tub_lid", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                 size=[TUB_LID_R, TUB_LID_H / 2, 0], pos=[0, 0, TUB_H / 2 - TUB_LID_H / 2],
                 mass=0.05, rgba=[0.5, 0.2, 0.75, 1], friction=[1.2, 0.005, 0.0001], condim=4)
    tub.add_site(name="tub_center", size=[0.005, 0, 0])
    return spec


def make_model(g1_scene=G1_SCENE):
    spec = make_spec(g1_scene)
    return spec.compile(), spec
