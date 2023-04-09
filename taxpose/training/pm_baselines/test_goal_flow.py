import os
import pickle
import time
from typing import List

import imageio
import numpy as np
import pybullet as p
import torch
from rpad.partnet_mobility_utils.render.pybullet import PMRenderEnv
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from taxpose.datasets.pm_placement import (
    ACTION_CLOUD_DIR,
    ACTION_OBJS,
    CATEGORIES,
    GOAL_INF_DSET_PATH,
    SEM_CLASS_DSET_PATH,
    SNAPPED_GOAL_FILE,
    base_from_bottom,
    downsample_pcd_fps,
    find_link_index_to_open,
    get_category,
    load_action_obj_with_valid_scale,
    randomize_block_pose,
    render_input,
    render_input_simple,
)
from taxpose.training.pm_baselines.dataloader_ff_interp_bc import (
    articulate_specific_joints,
)
from taxpose.training.pm_baselines.flow_model import FlowNet
from taxpose.training.pm_baselines.motion_planning import motion_planning_fcl
from taxpose.training.pm_baselines.test_bc import (
    calculate_chamfer_dist,
    get_checkpoint_path,
    get_demo,
)
from taxpose.training.pm_baselines.train import get_ids

"""
This file loads a trained goal inference model and tests the rollout using motion planning in simulation.
"""


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
    root: str,
    goal_id_list: list,
    full_sem_dset: dict,
    object_dict: dict,
    snap_to_surface=True,
    full_obj=True,
    even_downsample=True,
    WHICH=None,
):
    """
    Create demonstration
    """
    # First, create an environment which will generate our source observations.
    visible = False
    while not visible:
        demo_id: str = np.random.choice(goal_id_list)
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
            os.path.join(root, "raw"),
            camera_pos=[-3, 0, 1.2],
            gui=False,
        )
        if partsem != "none":
            links_tomove = find_link_index_to_open(
                full_sem_dset, partsem, obj_id, object_dict, goal_id
            )
            articulate_specific_joints(env, links_tomove, 0.9)

        # Select the action object.
        action_id = np.random.choice(list(ACTION_OBJS.keys()))
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


def is_obs_valid(block_id, sim: PMRenderEnv):
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
    return sum(mask) > 0 and collision_counter == 0


def randomize_start_pose(block_id, goal_pos, sim: PMRenderEnv):
    # This randomizes the starting pose
    valid_start = False
    while not valid_start:
        obs_curr_xyz = randomize_block_pose(
            (os.getpid() * int(time.time())) % 123456789
        )
        p.resetBasePositionAndOrientation(
            block_id,
            posObj=obs_curr_xyz,
            ornObj=[0, 0, 0, 1],
            physicsClientId=sim.client_id,
        )
        valid_start = (
            is_obs_valid(block_id, sim)
            and np.linalg.norm(goal_pos - obs_curr_xyz) >= 0.3
        )
    return obs_curr_xyz


def create_test_env(
    root: str,
    goal_id_list: List[str],
    full_sem_dset: dict,
    object_dict: dict,
    snap_to_surface=True,
    full_obj=True,
    even_downsample=True,
):
    # This creates the test env for demonstration.
    return get_demo_from_list(
        root,
        goal_id_list,
        full_sem_dset,
        object_dict,
        snap_to_surface,
        full_obj,
        even_downsample,
    )


def infer_goal(goalinf_model, P_world, obj_mask, P_world_goal, goal_mask):
    # This infers the goal using the goalinf_modem obs PCD (P_world), and demo PCD (P_world_goal)

    # IS_FLOW: Boolean value indicating if the model is flow-based or not. If TRUE: add pred flow.
    # Otherwise, output goal directly.
    pred = goalinf_model.predict(
        torch.from_numpy(P_world).float(),
        torch.from_numpy(obj_mask).float(),
        torch.from_numpy(P_world_goal).float(),
        torch.from_numpy(goal_mask).float(),
    )
    pred: np.ndarray = pred.cpu().numpy()
    # Mask out the pred_flow, not necessary if flow model is good
    pred[~obj_mask] = 0
    inferred_goal: np.ndarray = P_world + pred
    return inferred_goal


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cat", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--method", type=str, default="goal_flow")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--indist", type=bool, default=True)
    parser.add_argument("--rollout_dir", type=str, default="./baselines/rollouts")
    parser.add_argument(
        "--pm-root", type=str, default=os.path.expanduser("~/datasets/partnet-mobility")
    )
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoints")
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument(
        "--freefloat-dset",
        type=str,
        default="./data/free_floating_traj_interp_multigoals",
    )
    parser.add_argument("--rollout-len", type=int, default=20)
    parser.add_argument("--snap", action="store_true")
    args = parser.parse_args()
    objcat = args.cat
    method = args.method
    expname = args.model
    start_ind = args.start
    in_dist = args.indist
    rollout_dir = args.rollout_dir
    pm_root = args.pm_root
    ckpt_dir = args.ckpt_dir
    freefloat_dset = args.freefloat_dset
    rollout_len = args.rollout_len
    snap_to_surface = args.snap
    trial_len = 20

    with open(SEM_CLASS_DSET_PATH, "rb") as f:
        full_sem_dset = pickle.load(f)
    with open(
        os.path.join(GOAL_INF_DSET_PATH, f"{objcat}_block_dset_multi.pkl"), "rb"
    ) as f:
        object_dict_meta = pickle.load(f)

    # Get goal inference model
    ckpt_path = get_checkpoint_path(method, ckpt_dir, expname)
    goalinf_model = FlowNet.load_from_checkpoint(ckpt_path)
    goalinf_model.eval()

    # Create result directory
    result_dir = f"{rollout_dir}/{objcat}_{method}_{expname}"
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

    for o in tqdm(objs[start_ind // trial_len :]):
        if objcat == "all":
            object_dict = object_dict_meta[get_category(o.split("_")[0]).lower()]
        else:
            object_dict = object_dict_meta
        # Get demo ids list
        demo_id_list = list(object_dict.keys())
        trial = 0
        while trial < trial_len:
            obj_id = f"{o}_{trial}"

            # Obtain Demo data
            (
                P_world_demo,
                pc_seg_obj_demo,
                P_demo_full,
                pc_demo_full,
                rgb_goal,
                demo_id,
                which_goal,
            ) = get_demo(
                demo_id_list,
                full_sem_dset,
                object_dict,
                snap_to_surface=snap_to_surface,
            )

            # Get GT goal position

            which_goal = demo_id.split("_")[1]
            if demo_id == "7263_1":
                continue

            # Create obs env
            obs_env, obs_block_id, gt_goal_xyz, scale, action_id = create_test_env(
                obj_id,
                full_sem_dset,
                object_dict,
                which_goal,
                in_dist=in_dist,
                snap_to_surface=snap_to_surface,
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
            try:
                inferred_goal = infer_goal(
                    goalinf_model,
                    P_world,
                    pc_seg_obj == 99,
                    P_world_demo,
                    pc_seg_obj_demo == 99,
                )
            except IndexError:
                breakpoint()
            # Get GT goal PCD
            gt_flow = np.tile(gt_goal_xyz - start_xyz, (P_world.shape[0], 1))
            gt_flow[~(pc_seg_obj == 99)] = 0
            gt_goal = P_world + gt_flow

            chamf_dist_start = calculate_chamfer_dist(P_world, gt_goal)
            if chamf_dist_start == 0:
                continue

            chamf_dist_goal = calculate_chamfer_dist(inferred_goal, gt_goal)
            result_dict[obj_id] = chamf_dist_goal / chamf_dist_start
            if chamf_dist_goal / chamf_dist_start > 1:
                result_dict[obj_id] = 1

            scene_pose = np.eye(4)
            scene_xyz, scene_rot = p.getBasePositionAndOrientation(
                obs_env.obj_id, physicsClientId=obs_env.client_id
            )
            scene_pose[:3, -1] = np.array(scene_xyz)
            scene_pose[:3, :3] = R.from_quat(np.array(scene_rot)).as_matrix()
            result_dir = result_dir

            # Motion planning
            path_list_fcl = motion_planning_fcl(
                inferred_goal,
                P_world,
                P_world_full,
                pc_seg_obj == 99,
                pc_seg_obj_full == 99,
                start_full_pose,
            )

            if path_list_fcl is None:
                p.disconnect()
                continue

            result_dirs = [result_dir]
            trial += 1
            # for i, path_list in enumerate([path_list_fcl]):
            for i in range(1):
                path_list = path_list_fcl
                result_dir = result_dirs[i]
                exec_gifs = []

                # Initialize mp result logging
                # 1: success
                # 0: failure
                # key is [SUCC, NORM_DIST]
                mp_result_dict[obj_id] = [1]
                collision_counter = 0
                for pp in path_list:
                    p.resetBasePositionAndOrientation(
                        obs_block_id,
                        posObj=pp[:3],
                        ornObj=[0, 0, 0, 1],
                        physicsClientId=obs_env.client_id,
                    )

                    collision_counter_temp = (
                        len(
                            p.getClosestPoints(
                                bodyA=obs_block_id,
                                bodyB=obs_env.obj_id,
                                distance=0,
                                physicsClientId=obs_env.client_id,
                            )
                        )
                        > 0
                    )

                    collision_counter += collision_counter_temp
                    if collision_counter > 2:
                        mp_result_dict[obj_id][0] = 0

                    (
                        rgb,
                        depth,
                        seg,
                        P_cam,
                        P_world,
                        P_rgb,
                        pc_seg,
                        segmap,
                    ) = obs_env.render(True)
                    exec_gifs.append(rgb)
                imageio.mimsave(
                    f"{result_dir}/vids/test_{obj_id}.gif", exec_gifs, fps=25
                )
                imageio.imsave(f"{result_dir}/vids/test_{obj_id}_goal.png", rgb_goal)
                mp_result_dict[obj_id].append(
                    min(
                        1,
                        np.linalg.norm(pp - gt_goal_xyz)
                        / np.linalg.norm(start_xyz - gt_goal_xyz),
                    )
                )

                # Log the result to text file
                goalinf_res_file = open(
                    os.path.join(result_dir, f"rollout_goalinf_res.txt"), "a"
                )
                print(f"{obj_id}: {result_dict[obj_id]}", file=goalinf_res_file)

                if mp:
                    mp_res_file = open(
                        os.path.join(result_dir, f"rollout_mp_res.txt"), "a"
                    )
                    print(f"{obj_id}: {mp_result_dict[obj_id]}", file=mp_res_file)
                    mp_res_file.close()
                goalinf_res_file.close()
            p.disconnect()
            trial_start = 0

    print("Result: \n")
    print(result_dict)
