import json
import os
import pickle

import numpy as np
import pybullet as p
import torch
from chamferdist import ChamferDistance
from rpad.partnet_mobility_utils.render.pybullet import PMRenderEnv
from scipy.spatial.transform import Rotation as R
from torch import nn
from tqdm import tqdm

from taxpose.datasets.pm_placement import (
    ACTION_CLOUD_DIR,
    ACTION_OBJS,
    CATEGORIES,
    GOAL_INF_DSET_PATH,
    SEEN_CATS,
    SEM_CLASS_DSET_PATH,
    SNAPPED_GOAL_FILE,
    UNSEEN_CATS,
    base_from_bottom,
    downsample_pcd_fps,
    find_link_index_to_open,
    get_category,
    get_dataset_ids_all,
    load_action_obj_with_valid_scale,
    render_input,
    render_input_simple,
)
from taxpose.training.pm_baselines.bc_dataset import articulate_specific_joints
from taxpose.training.pm_baselines.flow_model import FlowNet as GoalInfFlowNet
from taxpose.training.pm_baselines.flow_model import (
    FlowNetParams as GoalInfFlowNetParams,
)
from taxpose.training.pm_baselines.test_bc import (
    quaternion_sum,
    randomize_start_pose,
    rigid_transform_3D,
)

"""
This file loads a trained goal inference model and tests the rollout using motion planning in simulation.
"""


def create_test_env(
    pm_root: str,
    obj_id: str,
    full_sem_dset: dict,
    object_dict: dict,
    goal_id: str,
    in_dist=True,
    snap_to_surface=True,
    seed=None,
):
    # This creates the test env for observation.
    obs_env = PMRenderEnv(
        obj_id.split("_")[0],
        os.path.join(pm_root, "raw"),
        camera_pos=[-3, 0, 1.2],
        gui=False,
    )
    obj_id = obj_id.split("_")[0]

    rng = np.random.default_rng(seed)

    partsem = object_dict[f"{obj_id}_{goal_id}"]["partsem"]
    if partsem != "none":
        links_tomove = find_link_index_to_open(
            full_sem_dset, partsem, obj_id, object_dict, goal_id
        )
        articulate_specific_joints(obs_env, links_tomove, 0.9)

    action_id = rng.choice(list(ACTION_OBJS.keys()))
    action_obj = ACTION_OBJS[action_id]
    # Load the object at the original floating goal, with a size that is valid there.
    info = object_dict[f"{obj_id}_{goal_id}"]
    floating_goal = np.array([info["x"], info["y"], info["z"]])
    action_body_id, scale = load_action_obj_with_valid_scale(
        action_obj, floating_goal, obs_env
    )

    # Find the actual desired goal position. In the case where we snap to the
    # goal surface, we need to calculate the position in which to reset (base_pos).
    with open(SNAPPED_GOAL_FILE, "rb") as f:
        snapped_goal_dict = pickle.load(f)
    if snap_to_surface:
        action_goal_pos_pre = snapped_goal_dict[CATEGORIES[obj_id].lower()][
            f"{obj_id}_{goal_id}"
        ]
        gt_goal_pos = base_from_bottom(action_body_id, obs_env, action_goal_pos_pre)
    else:
        gt_goal_pos = floating_goal
    randomize_start_pose(
        action_body_id, obs_env, in_dist=in_dist, goal_pos=gt_goal_pos, seed=rng
    )
    return obs_env, action_body_id, gt_goal_pos, scale, action_id


def augent_with_more_views(
    obs_block_id, obs_env: PMRenderEnv, P_world_full, pc_seg_obj_full
):
    original_cam_pos = obs_env.camera.T_world2cam[:3, -1]
    for pos in [[0, 3, 1.2], [0, -3, 1.2]]:
        obs_env.camera.set_camera_position(pos)
        P_world, pc_seg_obj, rgb = render_input(obs_block_id, obs_env)
        P_world_full = np.concatenate([P_world_full, P_world])
        pc_seg_obj_full = np.concatenate([pc_seg_obj_full, pc_seg_obj])

    # Restore the original pose
    obs_env.camera.set_camera_position(original_cam_pos)
    return P_world_full, pc_seg_obj_full


def get_ids(cat):
    if cat != "All":
        split_file = json.load(
            open(os.path.expanduser("~/umpnet/mobility_dataset/split-full.json"))
        )
        res = []
        for mode in split_file:
            if cat in split_file[mode]:
                res += split_file[mode][cat]["train"]
                res += split_file[mode][cat]["test"]
        return res
    else:
        _, val_res, unseen_res = get_dataset_ids_all(SEEN_CATS, UNSEEN_CATS)
        return val_res + unseen_res


def load_bc_model(method: str, exp_name: str):
    d = os.path.join(
        os.getcwd(),
        f"checkpoints/{method}/{exp_name}/",
    )
    ckpt = os.listdir(d)[0]
    param = GoalInfFlowNetParams()
    net: nn.Module
    net = GoalInfFlowNet.load_from_checkpoint(
        f"{d}/{ckpt}",
        p=param,
    )
    return net


def get_obs_data_from_env(
    action_body_id, env, scale, action_id, full_obj=True, even_downsample=True
):
    """
    Create demonstration
    """

    P_world, pc_seg, rgb, action_mask = render_input_simple(action_body_id, env)
    action_pos, _ = p.getBasePositionAndOrientation(
        action_body_id, physicsClientId=env.client_id
    )

    # We need enough visible points.
    if sum(action_mask) < 1:
        # If we don't find them in obs mode, it's because the random object position has been occluded.
        # In this case, we just need to resample.
        p.removeBody(action_body_id, physicsClientId=env.client_id)

        raise ValueError("the goal we sampled isn't visible! ")

    # Separate out the action and anchor points.
    P_action_world = P_world[action_mask]
    P_anchor_world = P_world[~action_mask]

    # Decide if we want to swap out the action points for the full point cloud.
    if full_obj:
        P_action_world = np.load(ACTION_CLOUD_DIR / f"{action_id}.npy")
        P_action_world *= scale
        P_action_world += action_pos

    P_action_world_full = torch.from_numpy(P_action_world).float()
    P_anchor_world_full = torch.from_numpy(P_anchor_world).float()

    # Now, downsample
    if even_downsample:
        action_ixs = downsample_pcd_fps(P_action_world_full, n=200)
        anchor_ixs = downsample_pcd_fps(P_anchor_world_full, n=1800)
    else:
        action_ixs = torch.randperm(len(P_action_world_full))[:200]
        anchor_ixs = torch.randperm(len(P_anchor_world_full))[:1800]

    # Rebuild the world
    P_action_world = P_action_world_full[action_ixs]
    P_anchor_world = P_anchor_world_full[anchor_ixs]
    P_world_full = np.concatenate([P_action_world_full, P_anchor_world_full], axis=0)
    P_world = np.concatenate([P_action_world, P_anchor_world], axis=0)

    # Regenerate a mask.
    mask_act_full = 99 * torch.ones(len(P_action_world_full)).int()
    mask_anc_full = torch.zeros(len(P_anchor_world_full)).int()
    mask_full = torch.cat([mask_act_full, mask_anc_full])
    mask_act = 99 * torch.ones(len(P_action_world)).int()
    mask_anc = torch.zeros(len(P_anchor_world)).int()
    mask = torch.cat([mask_act, mask_anc])

    # Unload the object from the scene.
    # p.removeBody(action_body_id, physicsClientId=env.client_id)

    return (
        P_world,
        mask.cpu().numpy(),
        P_world_full,
        mask_full.cpu().numpy(),
    )


def get_demo_from_list(
    pm_root: str,
    goal_id_list: list,
    full_sem_dset: dict,
    object_dict: dict,
    snap_to_surface=True,
    full_obj=True,
    even_downsample=True,
    WHICH=None,
    seed=None,
):
    """
    Create demonstration
    """
    # First, create an environment which will generate our source observations.
    rng = np.random.default_rng(seed)
    visible = False
    while not visible:
        demo_id: str = rng.choice(goal_id_list)
        if WHICH is not None:
            demo_id = f"{demo_id.split('_')[0]}_{WHICH}"
            if demo_id not in goal_id_list:
                return None
        obj_id = demo_id.split("_")[0]
        goal_id = demo_id.split("_")[1]

        # Next, check to see if the object needs to be opened in any way.
        partsem = object_dict[f"{obj_id}_{goal_id}"]["partsem"]
        env = PMRenderEnv(
            obj_id.split("_")[0],
            os.path.join(pm_root, "raw"),
            camera_pos=[-3, 0, 1.2],
            gui=False,
        )
        if partsem != "none":
            links_tomove = find_link_index_to_open(
                full_sem_dset, partsem, obj_id, object_dict, goal_id
            )
            articulate_specific_joints(env, links_tomove, 0.9)

        # Select the action object.
        action_id = rng.choice(list(ACTION_OBJS.keys()))
        action_obj = ACTION_OBJS[action_id]

        # Load the object at the original floating goal, with a size that is valid there.
        info = object_dict[f"{obj_id}_{goal_id}"]
        floating_goal = np.array([info["x"], info["y"], info["z"]])
        load_demo_res = load_action_obj_with_valid_scale(action_obj, floating_goal, env)
        if load_demo_res == None:
            continue
        action_body_id, scale = load_demo_res

        # Find the actual desired goal position. In the case where we snap to the
        # goal surface, we need to calculate the position in which to reset (base_pos).
        with open(SNAPPED_GOAL_FILE, "rb") as f:
            snapped_goal_dict = pickle.load(f)
        if snap_to_surface:
            action_goal_pos_pre = snapped_goal_dict[CATEGORIES[obj_id].lower()][
                f"{obj_id}_{goal_id}"
            ]
            action_goal_pos = base_from_bottom(action_body_id, env, action_goal_pos_pre)
        else:
            action_goal_pos = floating_goal

        action_pos = action_goal_pos

        # Place the object at the desired start position.
        p.resetBasePositionAndOrientation(
            action_body_id,
            posObj=action_pos,
            ornObj=[0, 0, 0, 1],
            physicsClientId=env.client_id,
        )

        P_world, pc_seg, rgb, action_mask = render_input_simple(action_body_id, env)

        # We need enough visible points.
        if sum(action_mask) < 1:
            # If we don't find them in obs mode, it's because the random object position has been occluded.
            # In this case, we just need to resample.
            p.removeBody(action_body_id, physicsClientId=env.client_id)

        else:
            visible = True

    # Separate out the action and anchor points.
    P_action_world = P_world[action_mask]
    P_anchor_world = P_world[~action_mask]

    # Decide if we want to swap out the action points for the full point cloud.
    if full_obj:
        P_action_world = np.load(ACTION_CLOUD_DIR / f"{action_id}.npy")
        P_action_world *= scale
        P_action_world += action_pos

    P_action_world_full = torch.from_numpy(P_action_world).float()
    P_anchor_world_full = torch.from_numpy(P_anchor_world).float()

    # Now, downsample
    if even_downsample:
        action_ixs = downsample_pcd_fps(P_action_world_full, n=200)
        anchor_ixs = downsample_pcd_fps(P_anchor_world_full, n=1800)
    else:
        action_ixs = torch.randperm(len(P_action_world_full))[:200]
        anchor_ixs = torch.randperm(len(P_anchor_world_full))[:1800]

    # Rebuild the world
    P_action_world = P_action_world_full[action_ixs]
    P_anchor_world = P_anchor_world_full[anchor_ixs]
    P_world_full = np.concatenate([P_action_world_full, P_anchor_world_full], axis=0)
    P_world = np.concatenate([P_action_world, P_anchor_world], axis=0)

    # Regenerate a mask.
    mask_act_full = 99 * torch.ones(len(P_action_world_full)).int()
    mask_anc_full = torch.zeros(len(P_anchor_world_full)).int()
    mask_full = torch.cat([mask_act_full, mask_anc_full])
    mask_act = 99 * torch.ones(len(P_action_world)).int()
    mask_anc = torch.zeros(len(P_anchor_world)).int()
    mask = torch.cat([mask_act, mask_anc])

    # Unload the object from the scene.
    p.removeBody(action_body_id, physicsClientId=env.client_id)

    return (
        P_world,
        mask.cpu().numpy(),
        P_world_full,
        mask_full.cpu().numpy(),
        rgb,
        demo_id,
        goal_id,
    )


def get_demo(
    pm_root: str,
    goal_id_list: list,
    full_sem_dset: dict,
    object_dict: dict,
    snap_to_surface=True,
    full_obj=True,
    even_downsample=True,
    WHICH=None,
    seed=None,
):
    # This creates the test env for demonstration.
    return get_demo_from_list(
        pm_root,
        goal_id_list,
        full_sem_dset,
        object_dict,
        snap_to_surface,
        full_obj,
        even_downsample,
        WHICH=WHICH,
        seed=seed,
    )


def infer_goal(goalinf_model, P_world, obj_mask, P_world_goal, goal_mask):
    # This infers the goal using the goalinf_modem obs PCD (P_world), and demo PCD (P_world_goal)

    # IS_FLOW: Boolean value indicating if the model is flow-based or not. If TRUE: add pred flow.
    # Otherwise, output goal directly.
    pred_ = goalinf_model.predict(
        torch.from_numpy(P_world).float(),
        torch.from_numpy(obj_mask).float(),
        torch.from_numpy(P_world_goal).float(),
        torch.from_numpy(goal_mask).float(),
    )

    pred_: np.ndarray = pred_.cpu().numpy()
    # Mask out the pred_flow, not necessary if flow model is good
    pred_[~obj_mask] = 0
    inferred_goal: np.ndarray = P_world + pred_
    return inferred_goal


def calculate_chamfer_dist(inferred_goal, gt_goal):
    # This calculates the chamfer distance between inferred and GT goal.
    chamferDist = ChamferDistance()
    source_pcd = torch.from_numpy(inferred_goal[np.newaxis, :]).cuda().float()
    target_pcd = torch.from_numpy(gt_goal[np.newaxis, :]).cuda().float()
    assert len(source_pcd.shape) == 3
    assert len(target_pcd.shape) == 3
    dist = chamferDist(source_pcd, target_pcd).cpu().item()
    return dist


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cat", type=str)
    parser.add_argument("--method", type=str, default="goal_flow")
    parser.add_argument("--model", type=str)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--indist", type=bool, default=True)
    parser.add_argument("--postfix", type=str)
    parser.add_argument("--snap", action="store_true")
    parser.add_argument(
        "--pm-root", type=str, default=os.path.expanduser("~/dataset/partnet-mobility")
    )
    args = parser.parse_args()
    objcat = args.cat
    method = args.method
    postfix = args.postfix
    snap_to_surface = args.snap

    expname = args.model
    start_ind = args.start
    in_dist = args.indist
    pm_root = args.pm_root

    rng = np.random.default_rng(123456)

    # Get which joint to open
    with open(SEM_CLASS_DSET_PATH, "rb") as f:
        full_sem_dset = pickle.load(f)
    with open(
        os.path.join(GOAL_INF_DSET_PATH, f"{objcat}_block_dset_multi.pkl"), "rb"
    ) as f:
        object_dict_meta = pickle.load(f)

    # Get goal inference model
    goalinf_model = load_bc_model(method, expname).cuda()

    # Create result directory
    result_dir = f"./results/pm_baselines/{method}/{expname}/{postfix}"
    if not os.path.exists(result_dir):
        print("Creating result directory for rollouts")
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(f"{result_dir}/vids", exist_ok=True)

    # If we are doing "test" then we sample 10 times per object
    objs = get_ids(objcat.capitalize())
    if "7292" in objs:
        objs.remove("7292")

    result_dict = {}
    mp_result_dict = {}

    trial_len = 10
    trial_start = start_ind % trial_len

    for o in tqdm(objs[start_ind // trial_len :]):
        if objcat == "all":
            object_dict = object_dict_meta[get_category(o.split("_")[0]).lower()]
        else:
            object_dict = object_dict_meta
        # Get demo ids list
        demo_id_list = list(object_dict.keys())
        trial = trial_start
        while trial < trial_len:
            obj_id = f"{o}_{trial}"
            skip = False
            try:
                for ll in open(
                    os.path.join(result_dir, f"rollout_goalinf_res.txt"), "r"
                ).readlines():
                    if obj_id in ll:
                        print("skippin")
                        skip = True
                        break
                if skip:
                    trial += 1
                    continue
            except FileNotFoundError:
                skip = False
            # Obtain Demo data
            demo_res = get_demo(
                pm_root,
                demo_id_list,
                full_sem_dset,
                object_dict,
                snap_to_surface=snap_to_surface,
                WHICH=postfix,
                seed=rng,
            )

            # Get GT goal position

            if demo_res is None:
                print("Invalid demo")
                trial += 1
                continue

            (
                P_world_demo,
                pc_seg_obj_demo,
                P_demo_full,
                pc_demo_full,
                rgb_goal,
                demo_id,
                which_goal,
            ) = demo_res

            if demo_id == "7263_1":
                continue

            # Create obs env
            obs_env, obs_block_id, gt_goal_xyz, scale, action_id = create_test_env(
                pm_root,
                obj_id,
                full_sem_dset,
                object_dict,
                which_goal,
                in_dist=in_dist,
                snap_to_surface=snap_to_surface,
                seed=rng,
            )

            # Log starting position
            start_xyz, start_quat = p.getBasePositionAndOrientation(
                obs_block_id, physicsClientId=obs_env.client_id
            )
            start_xyz = np.array(start_xyz)
            start_rot = R.from_quat(start_quat).as_matrix()
            start_full_pose = np.vstack(
                [
                    np.hstack([start_rot, start_xyz.reshape(3, 1)]),
                    np.array([0, 0, 0, 1]).reshape(1, 4),
                ]
            )

            # Obtain observation data
            P_world, pc_seg_obj, P_world_full, pc_seg_obj_full = get_obs_data_from_env(
                obs_block_id, obs_env, scale, action_id
            )
            inferred_goal = infer_goal(
                goalinf_model,
                P_world,
                pc_seg_obj == 99,
                P_world_demo,
                pc_seg_obj_demo == 99,
            )
            current_xyz = np.array([start_xyz[0], start_xyz[1], start_xyz[2]])
            curr_quat = np.array(
                [start_quat[0], start_quat[1], start_quat[2], start_quat[3]]
            )
            pred_R, pred_t = rigid_transform_3D(
                P_world[pc_seg_obj == 99].T, (inferred_goal[pc_seg_obj == 99]).T
            )
            current_xyz = current_xyz + pred_t.reshape(
                3,
            )
            curr_quat = quaternion_sum(curr_quat, R.from_matrix(pred_R).as_quat())
            curr_quat = R.from_matrix(curr_quat).as_quat()

            start_trans_dist = np.linalg.norm(start_xyz - gt_goal_xyz)
            end_trans_dist = np.linalg.norm(current_xyz - gt_goal_xyz)
            A = R.from_quat([0, 0, 0, 1]).as_matrix()
            B = R.from_quat(curr_quat).as_matrix()

            end_rot_dist = np.arccos((np.trace(A.T @ B) - 1) / 2) * 360 / 2 / np.pi

            result_dict[obj_id] = [
                end_trans_dist,
                end_trans_dist / start_trans_dist,
                end_rot_dist,
            ]

            scene_pose = np.eye(4)
            scene_xyz, scene_rot = p.getBasePositionAndOrientation(
                obs_env.obj_id, physicsClientId=obs_env.client_id
            )
            scene_pose[:3, -1] = np.array(scene_xyz)
            scene_pose[:3, :3] = R.from_quat(np.array(scene_rot)).as_matrix()

            trial += 1
            for i in range(1):
                # Log the result to text file
                goalinf_res_file = open(
                    os.path.join(result_dir, f"rollout_goalinf_res.txt"), "a"
                )
                print(f"{obj_id}: {result_dict[obj_id]}", file=goalinf_res_file)
                goalinf_res_file.close()
            p.disconnect()

    print("Result: \n")
    print(result_dict)
