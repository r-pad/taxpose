import os
import pickle

import imageio
import numpy as np
import pybullet as p
import torch
from rpad.partnet_mobility_utils.render.pybullet import PMRenderEnv
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from taxpose.datasets.pm_placement import (
    GOAL_INF_DSET_PATH,
    RAVENS_ASSETS,
    SEM_CLASS_DSET_PATH,
    get_category,
    render_input,
    subsample_pcd,
)
from taxpose.training.pm_baselines.dataloader_ff_interp_bc import (
    articulate_specific_joints,
)
from taxpose.training.pm_baselines.flow_model import FlowNet
from taxpose.training.pm_baselines.test_bc import (
    calculate_chamfer_dist,
    get_ids,
    randomize_start_pose,
)

"""
This file loads a trained goal inference model and tests the rollout using motion planning in simulation.
"""


def load_model(method: str, exp_name: str) -> FlowNet:
    d = os.path.join(
        os.getcwd(),
        f"checkpoints/{method}/{exp_name}/",
    )
    ckpt = os.listdir(d)[0]
    net: FlowNet = FlowNet.load_from_checkpoint(
        f"{d}/{ckpt}",
    )
    return net


def create_test_env(
    root: str,
    obj_id: str,
    full_sem_dset: dict,
    object_dict: dict,
    which_goal: str,
    freefloat_dset: str,
):
    # This creates the test env for observation.
    obs_env = PMRenderEnv(
        obj_id.split("_")[0],
        os.path.join(root, "raw"),
        camera_pos=[-3, 0, 1.2],
        gui=False,
    )
    block = f"{RAVENS_ASSETS}/block/block.urdf"
    obs_block_id = p.loadURDF(block, physicsClientId=obs_env.client_id, globalScaling=4)

    partsem = object_dict[obj_id.split("_")[0] + f"_{which_goal}"]["partsem"]
    if partsem != "none":
        for mode in full_sem_dset:
            if partsem in full_sem_dset[mode]:
                if obj_id.split("_")[0] in full_sem_dset[mode][partsem]:
                    move_joints = full_sem_dset[mode][partsem][obj_id.split("_")[0]]

        obj_link_id = object_dict[obj_id.split("_")[0] + f"_{which_goal}"]["ind"]
        obj_id_links_tomove = move_joints[obj_link_id]

        # Open the joints.
        articulate_specific_joints(obs_env, obj_id_links_tomove, 0.9)

    randomize_start_pose(obs_block_id, obs_env)
    return obs_env, obs_block_id


def get_demo(goal_id: str, full_sem_dset: dict, object_dict: dict, pm_root: str):
    # This creates the test env for demonstration.

    goal_env = PMRenderEnv(
        goal_id.split("_")[0],
        os.path.join(pm_root, "raw"),
        camera_pos=[-3, 0, 1.2],
        gui=False,
    )
    partsem = object_dict[goal_id]["partsem"]

    if partsem != "none":
        for mode in full_sem_dset:
            if partsem in full_sem_dset[mode]:
                if goal_id.split("_")[0] in full_sem_dset[mode][partsem]:
                    move_joints = full_sem_dset[mode][partsem][goal_id.split("_")[0]]
        goal_link_id = object_dict[goal_id]["ind"]
        goal_id_links_tomove = move_joints[goal_link_id]
        goal_env.articulate_specific_joints(goal_id_links_tomove, 0.9)

    goal_block = f"{RAVENS_ASSETS}/block/block.urdf"
    goal_block_id = p.loadURDF(
        goal_block, physicsClientId=goal_env.client_id, globalScaling=4
    )
    goal_xyz = [
        object_dict[goal_id]["x"],
        object_dict[goal_id]["y"],
        object_dict[goal_id]["z"],
    ]
    p.resetBasePositionAndOrientation(
        goal_block_id,
        posObj=goal_xyz,
        ornObj=[0, 0, 0, 1],
        physicsClientId=goal_env.client_id,
    )
    P_world_goal_full, pc_seg_obj_goal_full, rgb_goal = render_input(
        goal_block_id, goal_env
    )
    P_world_goal, pc_seg_obj_goal = subsample_pcd(
        P_world_goal_full, pc_seg_obj_goal_full
    )
    goal_env.close()
    return (
        P_world_goal,
        pc_seg_obj_goal,
        P_world_goal_full,
        pc_seg_obj_goal_full,
        rgb_goal,
    )


def predict_next_step(
    goalinf_model, P_world, obj_mask, P_world_goal, goal_mask
) -> np.ndarray:
    # This infers the goal using the goalinf_modem obs PCD (P_world), and demo PCD (P_world_goal)
    pred_act = goalinf_model.predict(
        torch.from_numpy(P_world).float(),
        torch.from_numpy(obj_mask).float(),
        torch.from_numpy(P_world_goal).float(),
        torch.from_numpy(goal_mask).float(),
    )
    pred_flow = pred_act.cpu().numpy()
    pred_flow[~obj_mask] = 0
    inferred_next_step: np.ndarray = pred_flow
    return inferred_next_step


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cat", type=str)
    parser.add_argument("--method", type=str, default="gc_traj_flow")
    parser.add_argument("--model", type=str)
    parser.add_argument("--postfix", type=str)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--indist", type=bool, default=True)
    parser.add_argument("--postfix", type=str)
    parser.add_argument("--rollout_dir", type=str, default="./baselines/rollouts")

    args = parser.parse_args()
    objcat = args.cat
    method = args.method
    expname = args.model
    start_ind = args.start
    in_dist = args.indist
    postfix = args.postfix
    rollout_dir = args.rollout_dir

    # Get which joint to open
    full_sem_dset = pickle.load(open(SEM_CLASS_DSET_PATH, "rb"))
    object_dict_meta = pickle.load(
        open(f"{GOAL_INF_DSET_PATH}/{objcat}_block_dset_multi.pkl", "rb")
    )

    # Get goal inference model
    bc_model = load_model(method, expname)

    # Create result directory
    result_dir = f"{rollout_dir}/{objcat}_{method}_{expname}_{postfix}"
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

    for o in tqdm(objs[start_ind // 20 :]):
        if objcat == "all":
            object_dict = object_dict_meta[get_category(o.split("_")[0]).lower()]
        else:
            object_dict = object_dict_meta
        # Get demo ids list
        demo_id_list = list(object_dict.keys())
        trial = 0
        while trial < 20:
            obj_id = f"{o}_{trial}"

            # Get GT goal position

            demo_id = np.random.choice(demo_id_list)
            which_goal = demo_id.split("_")[1]
            if demo_id == "7263_1":
                continue

            # Create obs env
            obs_env, obs_block_id, gt_goal_xyz, _, _ = create_test_env(
                obj_id,
                full_sem_dset,
                object_dict,
                which_goal,
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

            # Obtain Demo data
            (
                P_world_demo,
                pc_seg_obj_demo,
                P_demo_full,
                pc_demo_full,
                rgb_goal,
            ) = get_demo(demo_id, full_sem_dset, object_dict)
            P_world_full, pc_seg_obj_full, _ = render_input(obs_block_id, obs_env)
            P_world_og, pc_seg_obj_og = subsample_pcd(P_world_full, pc_seg_obj_full)

            # BC POLICY ROLLOUT LOOP
            current_xyz = np.array([start_xyz[0], start_xyz[1], start_xyz[2]])
            exec_gifs = []
            mp_result_dict[obj_id] = [1]
            for t in range(60):
                # Obtain observation data
                p.resetBasePositionAndOrientation(
                    obs_block_id,
                    posObj=current_xyz,
                    ornObj=[0, 0, 0, 1],
                    physicsClientId=obs_env.client_id,
                )
                collision_counter = len(
                    p.getClosestPoints(
                        bodyA=obs_block_id,
                        bodyB=obs_env.obj_id,
                        distance=0,
                        physicsClientId=obs_env.client_id,
                    )
                )

                if collision_counter > 2:
                    mp_result_dict[obj_id][0] = 0
                P_world_full, pc_seg_obj_full, rgb_curr = render_input(
                    obs_block_id, obs_env
                )
                exec_gifs.append(rgb_curr)
                P_world, pc_seg_obj = subsample_pcd(P_world_full, pc_seg_obj_full)
                output_len = min(len(P_world), len(P_world_demo))
                P_world = P_world[:output_len]
                P_world_demo = P_world_demo[:output_len]
                pc_seg_obj = pc_seg_obj[:output_len]
                pc_seg_obj_demo = pc_seg_obj_demo[:output_len]
                try:
                    pred_flow = predict_next_step(
                        bc_model,
                        P_world,
                        pc_seg_obj == 99,
                        P_world_demo,
                        pc_seg_obj_demo == 99,
                    )
                    pred_flow = 0.1 * pred_flow
                except IndexError:
                    breakpoint()
                if (pc_seg_obj == 99).any():
                    current_xyz = current_xyz + pred_flow[pc_seg_obj == 99].mean(axis=0)
                if np.linalg.norm(current_xyz - gt_goal_xyz) <= 5e-2:
                    break

            # Get GT goal PCD
            gt_flow = np.tile(gt_goal_xyz - start_xyz, (P_world_og.shape[0], 1))
            gt_flow[~(pc_seg_obj_og == 99)] = 0
            gt_goal = P_world_og + gt_flow

            chamf_dist_start = calculate_chamfer_dist(P_world_og, gt_goal)

            syn_flow = np.tile(current_xyz - start_xyz, (P_world_og.shape[0], 1))
            syn_flow[~(pc_seg_obj_og == 99)] = 0
            inferred_goal = P_world_og + syn_flow

            chamf_dist_goal = calculate_chamfer_dist(inferred_goal, gt_goal)
            if chamf_dist_start == 0:
                p.disconnect()
                continue
            result_dict[obj_id] = chamf_dist_goal / chamf_dist_start
            if chamf_dist_goal / chamf_dist_start > 1:
                result_dict[obj_id] = 1

            trial += 1

            # Initialize mp result logging
            # 1: success
            # 0: failure
            # key is [SUCC, NORM_DIST]
            imageio.mimsave(f"{result_dir}/vids/test_{obj_id}.gif", exec_gifs, fps=25)
            imageio.imsave(f"{result_dir}/vids/test_{obj_id}_goal.png", rgb_goal)
            p.disconnect()
            mp_result_dict[obj_id].append(
                min(
                    1,
                    np.linalg.norm(current_xyz - gt_goal_xyz)
                    / np.linalg.norm(start_xyz - gt_goal_xyz),
                )
            )

            # Log the result to text file
            goalinf_res_file = open(
                os.path.join(result_dir, f"rollout_goalinf_res.txt"), "a"
            )
            print(f"{obj_id}: {result_dict[obj_id]}", file=goalinf_res_file)
            mp_res_file = open(os.path.join(result_dir, f"rollout_mp_res.txt"), "a")
            print(f"{obj_id}: {mp_result_dict[obj_id]}", file=mp_res_file)
            goalinf_res_file.close()
            mp_res_file.close()

    print("Result: \n")
    print(result_dict)