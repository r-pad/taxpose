import json
import os
from dataclasses import dataclass
from math import floor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pybullet as p
from rpad.core.distributed import NPSeed
from rpad.partnet_mobility_utils.render.pybullet import PMRenderEnv
from scipy.spatial.transform import Rotation as R

TAXPOSE_ROOT = Path(__file__).parent.parent.parent
GOAL_DATA_PATH = TAXPOSE_ROOT / "taxpose" / "datasets" / "pm_data"

GCOND_DSET_PATH = str(TAXPOSE_ROOT / "goalcond-pm-objs-split.json")
RAVENS_ASSETS = TAXPOSE_ROOT / "third_party/ravens/ravens/environments/assets"


def create_dataset_handles(dset, mode, nrep=1):
    envs_all = []
    split = floor(0.8 * len(dset))
    if mode == "train":
        for e in dset[:split]:
            for i in range(nrep):
                envs_all.append(f"{e}_{i}")
        return envs_all
    else:
        for e in dset[split:]:
            for i in range(nrep):
                envs_all.append(f"{e}_{i}")
        return envs_all


def get_dataset_ids(cat: str):
    cat = cat.capitalize()
    split_file = json.load(
        open(os.path.expanduser("~/umpnet/mobility_dataset/split-full.json"))
    )
    res = []
    for mode in split_file:
        if cat in split_file[mode]:
            res += split_file[mode][cat]["train"]
            res += split_file[mode][cat]["test"]
    if "7292" in res:
        res.remove("7292")
    return res


def get_dataset_ids_all(seen_cats, unseen_cats):
    split_file = json.load(open(GCOND_DSET_PATH))
    train_res = []
    val_res = []
    test_res = []
    for cat in seen_cats:
        cat = cat.capitalize()
        for mode in split_file:
            if cat in split_file[mode] and mode == "train":
                train_res += split_file[mode][cat]["train"]
    for cat in seen_cats:
        cat = cat.capitalize()
        for mode in split_file:
            if cat in split_file[mode] and mode == "train":
                val_res += split_file[mode][cat]["test"]
    for cat in unseen_cats:
        cat = cat.capitalize()
        for mode in split_file:
            if cat in split_file[mode] and mode == "test":
                test_res += split_file[mode][cat]["test"]
    if "7292" in train_res:
        train_res.remove("7292")
    return train_res, val_res, test_res


def create_dataset_handles_all(dset, nrep=1, random=False):
    envs_all = []
    if not random:
        for e in dset:
            for i in range(nrep):
                envs_all.append(f"{e}_{i}")
        return envs_all
    else:
        a = np.arange(20 * 21 * 17)
        rand_idx = np.random.choice(a, size=nrep)
        for e in dset:
            for i in rand_idx:
                envs_all.append(f"{e}_{i}")
        return envs_all


def get_random_obj(goal=False):
    if goal:
        return (
            os.path.expanduser(
                "~/discriminative_embeddings/third_party/ravens/ravens/environments/assets/block/block.urdf"
            ),
            4,
        )


@dataclass
class ActionObj:
    name: str
    urdf: str
    scale: Tuple[int, int]  # Appropriate low, hi.

    def random_scale(self, seed: NPSeed = None) -> float:
        low, high = self.scale
        rng = np.random.default_rng(seed)
        scale = rng.uniform(low=low, high=high)
        # print(f"{seed}, {low}, {high} -> {scale}", flush=True)
        return scale


ACTION_OBJS: Dict[str, ActionObj] = {
    "block": ActionObj("block", str(RAVENS_ASSETS / "block/block.urdf"), (2, 4)),
    "bowl": ActionObj("bowl", str(RAVENS_ASSETS / "bowl/bowl.urdf"), (2, 3)),
    "suctiontip": ActionObj(
        "suctiontip", str(RAVENS_ASSETS / "ur5/suction/suction-head.urdf"), (2, 4)
    ),
    "disk0": ActionObj("disk0", str(RAVENS_ASSETS / "hanoi/disk0.urdf"), (2, 4)),
    "disk1": ActionObj("disk1", str(RAVENS_ASSETS / "hanoi/disk1.urdf"), (2, 4)),
    "disk2": ActionObj("disk2", str(RAVENS_ASSETS / "hanoi/disk2.urdf"), (2, 4)),
    "disk3": ActionObj("disk3", str(RAVENS_ASSETS / "hanoi/disk3.urdf"), (2, 4)),
    "slimdisk": ActionObj(
        "slimdisk", str(RAVENS_ASSETS / "hanoi/slimdisk.urdf"), (2, 4)
    ),
    "ell": ActionObj("ell", str(RAVENS_ASSETS / "insertion/ell.urdf"), (2, 3)),
}


def get_category(obj_id):
    split_file = json.load(open(GCOND_DSET_PATH))
    for cat in split_file["train"]:
        if obj_id in split_file["train"][cat]["train"]:
            return cat
        elif obj_id in split_file["train"][cat]["test"]:
            return cat
    for cat in split_file["test"]:
        if obj_id in split_file["test"][cat]["test"]:
            return cat


def get_category_full_dataset(
    obj_id, root=os.path.expanduser("~/partnet-mobility/raw")
):
    split_file = json.load(open(os.path.join(root, obj_id, "meta.json")))
    return split_file["model_cat"]


def get_ids_from_category(cat):
    split_file = json.load(
        open(os.path.expanduser("~/umpnet/mobility_dataset/split-full.json"))
    )
    if cat in split_file["train"]:
        return split_file["train"][cat]["train"] + split_file["train"][cat]["test"]
    elif cat in split_file["test"]:
        return split_file["test"][cat]["test"]


def get_full_and_bottom_points(sim: PMRenderEnv, obj_id):
    """
    This returns the bottom points of object given its obj_id.
    It transforms the object into a canonical pose, where the bottom points are fully visible.
    Then it records the points and creates a mask, and then transforms it back.
    """

    # Step 1: Record the object's current pose.
    start_xyz, start_quat = p.getBasePositionAndOrientation(
        obj_id, physicsClientId=sim.client_id
    )

    # Step 2: Transform the object into its canonical pose
    p.resetBasePositionAndOrientation(
        obj_id,
        posObj=[-1.5, 0, 1.0],
        ornObj=[0, 1, 0, 1],
        physicsClientId=sim.client_id,
    )

    # Step 3: Render
    rgb, depth, seg, P_cam, P_world, P_rgb, pc_seg, segmap = sim.render(True)
    obj_pcd = P_world[pc_seg == obj_id]

    # Step 4: Record the bottom points via thresholding. The bottom points here are in the canonical frame
    cond = np.isclose(obj_pcd[:, 0], obj_pcd[:, 0].min(), atol=5e-3)
    bottom_pcd = obj_pcd[cond]

    # Step 5: Transform the canoncial pose back to the OG pose
    T_can = np.eye(4)
    T_can[:3, :3] = R.from_quat([0, 1, 0, 1]).as_matrix()
    T_can[:3, -1] = [-1.5, 0, 1.0]

    T_og = np.eye(4)
    T_og[:3, :3] = R.from_quat(start_quat).as_matrix()
    T_og[:3, -1] = start_xyz

    T_diff = T_og @ np.linalg.inv(T_can)

    # Bottom points in the OG pose
    bottom_pcd_og_pose = (T_diff[:3, :3] @ (bottom_pcd.T) + T_diff[:3, -1:]).T

    # Step 6: Restore OG pose
    p.resetBasePositionAndOrientation(
        obj_id,
        posObj=start_xyz,
        ornObj=start_quat,
        physicsClientId=sim.client_id,
    )

    # Render w/ postprocessing.
    P_world, pc_seg_obj, rgb = render_input(obj_id, sim)

    # Append
    P_world_with_bottom = np.concatenate([P_world, bottom_pcd_og_pose])
    # Mask for bottom points is 100
    pc_seg_obj_with_bottom = np.concatenate(
        [pc_seg_obj, 100 * np.ones(bottom_pcd_og_pose.shape[0])]
    )
    return P_world_with_bottom, pc_seg_obj_with_bottom, rgb


def render_input_new(action_id, sim: PMRenderEnv):
    rgb, _, _, _, P_world, pc_seg, _ = sim.render(link_seg=False)

    is_obj = pc_seg != -1
    P_world = P_world[is_obj]
    pc_seg = pc_seg[is_obj]

    action_mask = pc_seg == action_id
    return P_world, pc_seg, rgb, action_mask


def render_input(block_id, sim: PMRenderEnv, render_floor=False):
    # Post-processing: Render, add mask. First render visible points, then append transformed bottom points to them.
    rgb, _, _, _, P_world, pc_seg, segmap = sim.render()
    pc_seg_obj = np.ones_like(pc_seg) * -1
    for k, (body, link) in segmap.items():
        if body == sim.obj_id:
            ixs = pc_seg == k
            pc_seg_obj[ixs] = link
        elif body == block_id:
            ixs = pc_seg == k
            pc_seg_obj[ixs] = 99

    is_obj = pc_seg_obj != -1

    P_world = P_world[is_obj]
    pc_seg_obj = pc_seg_obj[is_obj]
    if render_floor:
        x = np.linspace(-0.8, 0.2, 50)
        y = np.linspace(-1, -0.5, 50)
        xv, yv = np.meshgrid(x, y)
        ground_points = np.vstack([xv.ravel(), yv.ravel(), np.zeros_like(xv.ravel())])
        ground_points = np.hstack(
            [
                ground_points,
                np.vstack([xv.ravel(), -yv.ravel(), np.zeros_like(xv.ravel())]),
            ]
        ).T
        P_world = np.vstack([P_world, ground_points])
        pc_seg_obj = np.hstack([pc_seg_obj, np.zeros(ground_points.shape[0])])
    return P_world, pc_seg_obj, rgb


def subsample_pcd(P_world, pc_seg_obj):
    subsample = np.where(pc_seg_obj == 99)[0]
    np.random.shuffle(subsample)
    subsample = subsample[:200]
    subsample_obj = np.where(pc_seg_obj != 99)[0]
    np.random.shuffle(subsample_obj)
    subsample_obj = subsample_obj[:1800]
    pc_seg_obj = np.concatenate([pc_seg_obj[subsample_obj], pc_seg_obj[subsample]])
    P_world = np.concatenate([P_world[subsample_obj], P_world[subsample]])
    while len(P_world) < 2000:
        pc_seg_obj = np.concatenate([pc_seg_obj, pc_seg_obj[-1:]])
        P_world = np.concatenate([P_world, P_world[-1:]])
    return P_world, pc_seg_obj


def has_collisions(action_id, sim: PMRenderEnv):
    p.performCollisionDetection(physicsClientId=sim.client_id)
    collision_counter = len(
        p.getClosestPoints(
            bodyA=action_id,
            bodyB=sim.obj_id,
            distance=0,
            physicsClientId=sim.client_id,
        )
    )
    return collision_counter > 0


def is_action_pose_valid(block_id, sim: PMRenderEnv, n_valid_points=50):
    # Valid only if the block is visible in the rendered image AND there's no collision.
    collision_counter = len(
        p.getClosestPoints(
            bodyA=block_id,
            bodyB=sim.obj_id,
            distance=0,
            physicsClientId=sim.client_id,
        )
    )

    _, pc_seg_obj, _ = render_input(block_id, sim)
    mask = pc_seg_obj == 99
    return sum(mask) >= n_valid_points and collision_counter == 0


def define_goal_location(obs_goal, sim, block_id, snap=False):
    if not snap:
        return obs_goal
    else:
        # Snap the object to a surface first
        p.resetBasePositionAndOrientation(
            block_id,
            posObj=obs_goal,
            ornObj=[0, 0, 0, 1],
            physicsClientId=sim.client_id,
        )
        p.setGravity(0, 0, -10, physicsClientId=sim.client_id)
        for _ in range(1000):
            p.stepSimulation(physicsClientId=sim.client_id)
        goal_pos, rot = p.getBasePositionAndOrientation(
            block_id, physicsClientId=sim.client_id
        )
        p.setGravity(0, 0, 0, physicsClientId=sim.client_id)
        return np.array(goal_pos)


SEEN_CATS = [
    "microwave",
    "dishwasher",
    "chair",
    "oven",
    "fridge",
    "safe",
    "table",
    "drawer",
    "washingmachine",
]
UNSEEN_CATS: List[str] = []
SNAPPED_GOAL_FILE = GOAL_DATA_PATH / "snapped_goals.pkl"
