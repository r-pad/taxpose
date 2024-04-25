import json
import os
import pickle

import imageio
import numpy as np
import pybullet as p
import torch
from chamferdist import ChamferDistance
from rpad.partnet_mobility_utils.render.pybullet import PMRenderEnv
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from taxpose.datasets.pm_placement import (
    ACTION_OBJS,
    GOAL_INF_DSET_PATH,
    SEEN_CATS,
    SEM_CLASS_DSET_PATH,
    UMPNET_SPLIT_FULL_FILE,
    UNSEEN_CATS,
    get_category,
    get_dataset_ids_all,
    is_action_pose_valid,
    render_input,
    subsample_pcd,
)
from taxpose.training.pm_baselines.bc_dataset import articulate_specific_joints
from taxpose.training.pm_baselines.bc_model import BCNet as DaggerNet
from taxpose.training.pm_baselines.flow_model import FlowNet as TrajFlowNet

"""
This file loads a trained BC model and tests the rollout in simulation.
"""


def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    # if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


def get_ids(cat):
    if cat != "All":
        split_file = json.load(open(UMPNET_SPLIT_FULL_FILE))
        res = []
        for mode in split_file:
            if cat in split_file[mode]:
                res += split_file[mode][cat]["train"]
                res += split_file[mode][cat]["test"]
        return res
    else:
        _, val_res, unseen_res = get_dataset_ids_all(SEEN_CATS, UNSEEN_CATS)
        return val_res + unseen_res


def randomize_block_pose(seed, in_dist=True):
    rng = np.random.default_rng(seed)
    if in_dist:
        randomized_pose = np.array(
            [
                rng.uniform(low=-1, high=-0.1),
                rng.uniform(low=-1.4, high=-1.0),
                rng.uniform(low=0.1, high=0.25),
            ]
        )
    else:
        randomized_pose = np.array(
            [
                rng.uniform(low=1.5, high=2),
                rng.uniform(low=-3, high=3),
                rng.uniform(low=1.5, high=2),
            ]
        )
    return randomized_pose


def randomize_start_pose(
    block_id, sim: PMRenderEnv, in_dist=True, goal_pos=None, seed=None
):
    # This randomizes the starting pose
    rng = np.random.default_rng(seed)
    valid_start = False
    while not valid_start:
        obs_curr_xyz = randomize_block_pose(rng, in_dist)
        angle = rng.uniform(-60, 60)
        start_ort = R.from_euler("z", angle, degrees=True).as_quat()
        p.resetBasePositionAndOrientation(
            block_id,
            posObj=obs_curr_xyz,
            ornObj=start_ort,
            physicsClientId=sim.client_id,
        )
        valid_start = is_action_pose_valid(block_id, sim, n_valid_points=0) and (
            in_dist or np.linalg.norm(goal_pos - obs_curr_xyz) >= 0.3  # type: ignore
        )
    return obs_curr_xyz


def create_test_env(
    pm_root: str,
    obj_id: str,
    full_sem_dset: dict,
    object_dict: dict,
    which_goal: str,
    in_dist=True,
    seed=None,
):
    # This creates the test env for observation.
    obs_env = PMRenderEnv(
        obj_id.split("_")[0],
        os.path.join(pm_root, "raw"),
        camera_pos=[-3, 0, 1.2],
        gui=False,
    )
    block = ACTION_OBJS["block"].urdf
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

    randomize_start_pose(obs_block_id, obs_env, in_dist, seed)
    return obs_env, obs_block_id


def get_demo(pm_root: str, goal_id: str, full_sem_dset: dict, object_dict: dict):
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
        articulate_specific_joints(goal_env, goal_id_links_tomove, 0.9)

    goal_block = ACTION_OBJS["block"].urdf
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
    return (
        P_world_goal,
        pc_seg_obj_goal,
        P_world_goal_full,
        pc_seg_obj_goal_full,
        rgb_goal,
    )


def calculate_chamfer_dist(inferred_goal, gt_goal):
    # This calculates the chamfer distance between inferred and GT goal.
    chamferDist = ChamferDistance()
    source_pcd = torch.from_numpy(inferred_goal[np.newaxis, :]).cuda()
    target_pcd = torch.from_numpy(gt_goal[np.newaxis, :]).cuda()
    assert len(source_pcd.shape) == 3
    assert len(target_pcd.shape) == 3
    dist = chamferDist(source_pcd, target_pcd).cpu().item()
    return dist


def quaternion_sum(q1, q2):
    R1 = R.from_quat(q1).as_matrix()
    R2 = R.from_quat(q2).as_matrix()
    return R1 @ R2


def quaternion_diff(q1, q2):
    R1 = R.from_quat(q1).as_matrix()
    R2 = R.from_quat(q2).as_matrix()
    return np.linalg.inv(R1) @ R2


def load_bc_model(method: str, exp_name: str) -> DaggerNet:
    d = os.path.join(
        os.getcwd(),
        f"checkpoints/{method}/{exp_name}/",
    )
    ckpt = os.listdir(d)[0]
    net = DaggerNet.load_from_checkpoint(
        f"{d}/{ckpt}",
    )
    return net.cuda()  # type: ignore


def predict_next_step_bc(
    goalinf_model, P_world, obj_mask, P_world_goal, goal_mask, curr_xyz, curr_quat
):
    # This infers the goal using the goalinf_modem obs PCD (P_world), and demo PCD (P_world_goal)
    pred_act = goalinf_model.predict(
        torch.from_numpy(P_world).float().cuda(),
        torch.from_numpy(obj_mask).float().cuda(),
        torch.from_numpy(P_world_goal).float().cuda(),
        torch.from_numpy(goal_mask).float().cuda(),
    )
    pred_act = pred_act.cpu().numpy()[0]
    pred_act_tr = 0.05 * pred_act[:3] / np.linalg.norm(pred_act[:3])
    inferred_next_step_tr: np.ndarray = curr_xyz + pred_act_tr
    pred_act_quat = pred_act[3:]
    # inferred_next_step_quat = quaternion_sum(curr_quat, pred_act_quat)
    pred_act_rot = R.from_quat(pred_act_quat).as_matrix()
    inferred_next_step_quat = R.from_matrix(
        R.from_quat(
            curr_quat.reshape(
                4,
            )
        ).as_matrix()
        @ pred_act_rot
    ).as_quat()
    return inferred_next_step_tr, inferred_next_step_quat


def bc_policy(
    model,
    P_world,
    pc_seg_obj,
    P_world_demo,
    pc_seg_obj_demo,
    current_xyz,
    curr_quat,
):
    pred_action_tr, pred_action_rot = predict_next_step_bc(
        model,
        P_world,
        pc_seg_obj == 99,
        P_world_demo,
        0 * (pc_seg_obj_demo == 99),
        current_xyz,
        curr_quat,
    )
    pred_action_tr = pred_action_tr.reshape(
        -1,
    )
    pred_action_rot = pred_action_rot.reshape(
        -1,
    )

    if (pc_seg_obj == 99).any():
        current_xyz = pred_action_tr
        curr_quat = pred_action_rot

    return current_xyz, curr_quat


def load_model_traj_flow(method: str, exp_name: str) -> TrajFlowNet:
    d = os.path.join(
        os.getcwd(),
        f"checkpoints/{method}/{exp_name}/",
    )
    ckpt = os.listdir(d)[0]
    net: TrajFlowNet = TrajFlowNet.load_from_checkpoint(
        f"{d}/{ckpt}",
    )
    return net.cuda()


def predict_next_step_traj_flow(
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


def traj_flow_policy(
    model, P_world, pc_seg_obj, P_world_demo, pc_seg_obj_demo, current_xyz, curr_quat
):
    pred_flow = predict_next_step_traj_flow(
        model,
        P_world,
        pc_seg_obj == 99,
        P_world_demo,
        pc_seg_obj_demo == 99,
    )
    pred_flow = 0.1 * pred_flow

    if (pc_seg_obj == 99).any():
        curr_obj = P_world[pc_seg_obj == 99]
        next_obj = curr_obj + pred_flow[pc_seg_obj == 99]
        pred_R, pred_t = rigid_transform_3D(curr_obj.T, next_obj.T)
        current_xyz = current_xyz + pred_t.reshape((3,))
        curr_quat = quaternion_sum(curr_quat, R.from_matrix(pred_R).as_quat())
        curr_quat = R.from_matrix(curr_quat).as_quat()

    return current_xyz, curr_quat


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cat", type=str, required=True)
    parser.add_argument("--method", type=str, default="bc")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--indist", type=bool, default=True)
    parser.add_argument("--postfix", type=str)
    parser.add_argument(
        "--pm-root", type=str, default=os.path.expanduser("~/datasets/partnet-mobility")
    )
    args = parser.parse_args()
    objcat = args.cat
    method = args.method
    expname = args.model
    start_ind = args.start
    in_dist = args.indist
    postfix = args.postfix
    pm_root = args.pm_root

    # goal flow gets run in a different file.
    assert method in {"bc", "dagger", "traj_flow"}

    # Get which joint to open
    with open(SEM_CLASS_DSET_PATH, "rb") as f:
        full_sem_dset = pickle.load(f)
    with open(
        os.path.join(GOAL_INF_DSET_PATH, f"{objcat}_block_dset_multi.pkl"), "rb"
    ) as f:
        object_dict_meta = pickle.load(f)

    # Get goal inference model
    if method == "bc" or method == "dagger":
        model = load_bc_model(method, expname)
    elif method == "traj_flow":
        model = load_model_traj_flow(method, expname)
    elif method == "goal_flow":
        raise ValueError("Goal flow not implemented yet")
    else:
        raise ValueError("Invalid method")

    # Create result directory
    result_dir = f"./results/pm_baselines/{method}/{expname}/{postfix}"
    if not os.path.exists(result_dir):
        print("Creating result directory for results")
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(f"{result_dir}/vids", exist_ok=True)

    # If we are doing "test" then we sample 10 times per object
    objs = get_ids(objcat.capitalize())
    if "7292" in objs:
        objs.remove("7292")

    result_dict = {}
    mp_result_dict = {}

    trial_start = start_ind % 20
    which_goal = postfix
    if method == "bc" or method == "dagger":
        num_trials = 10
        rollout_len = 20
    elif method == "traj_flow":
        num_trials = 8
        rollout_len = 60
    elif method == "goal_flow":
        raise ValueError("Goal flow not implemented yet")
    else:
        raise ValueError("Invalid method")

    rng = np.random.default_rng(123456)

    for o in tqdm(objs[start_ind // 20 :]):
        if objcat == "all":
            object_dict = object_dict_meta[get_category(o.split("_")[0]).lower()]
        else:
            object_dict = object_dict_meta
        # Get demo ids list
        demo_id_list = list(object_dict.keys())
        trial = trial_start
        while trial < num_trials:
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

            # Get GT goal position

            demo_id = rng.choice(demo_id_list)
            demo_id = f"{demo_id.split('_')[0]}_{which_goal}"
            if demo_id not in demo_id_list:
                trial += 1
                continue
            if demo_id == "7263_1":
                continue
            gt_goal_xyz = [
                object_dict[f"{o}_{which_goal}"]["x"],
                object_dict[f"{o}_{which_goal}"]["y"],
                object_dict[f"{o}_{which_goal}"]["z"],
            ]

            # Create obs env
            obs_env, obs_block_id = create_test_env(
                pm_root, obj_id, full_sem_dset, object_dict, which_goal, in_dist, rng
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
            ) = get_demo(pm_root, demo_id, full_sem_dset, object_dict)
            P_world_full, pc_seg_obj_full, _ = render_input(obs_block_id, obs_env)
            P_world_og, pc_seg_obj_og = subsample_pcd(P_world_full, pc_seg_obj_full)

            # BC POLICY ROLLOUT LOOP
            current_xyz = np.array([start_xyz[0], start_xyz[1], start_xyz[2]])
            curr_quat = np.array(
                [start_quat[0], start_quat[1], start_quat[2], start_quat[3]]
            )
            exec_gifs = []
            mp_result_dict[obj_id] = [1]
            for t in range(rollout_len):
                # Obtain observation data
                p.resetBasePositionAndOrientation(
                    obs_block_id,
                    posObj=current_xyz,
                    ornObj=curr_quat,
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

                if method == "bc" or method == "dagger":
                    current_xyz, curr_quat = bc_policy(
                        model,
                        P_world,
                        pc_seg_obj,
                        P_world_demo,
                        pc_seg_obj_demo,
                        current_xyz,
                        curr_quat,
                    )
                elif method == "traj_flow":
                    current_xyz, curr_quat = traj_flow_policy(
                        model,
                        P_world,
                        pc_seg_obj,
                        P_world_demo,
                        pc_seg_obj_demo,
                        current_xyz,
                        curr_quat,
                    )
                elif method == "goal_flow":
                    raise ValueError("Goal flow not implemented yet")
                else:
                    raise ValueError("Invalid method")

                if np.linalg.norm(current_xyz - gt_goal_xyz) <= 5e-2:
                    break

            # Get GT goal PCD
            gt_flow = np.tile(gt_goal_xyz - start_xyz, (P_world_og.shape[0], 1))
            gt_flow[~(pc_seg_obj_og == 99)] = 0
            gt_goal = P_world_og + gt_flow

            start_trans_dist = np.linalg.norm(start_xyz - gt_goal_xyz)
            end_trans_dist = np.linalg.norm(current_xyz - gt_goal_xyz)
            start_rot_dist = np.linalg.norm(
                R.from_quat([0, 0, 0, 1]).as_euler("xyz")
                - R.from_quat(start_quat).as_euler("xyz")
            )
            # end_rot_dist = np.linalg.norm(
            #     R.from_quat([0, 0, 0, 1]).as_euler("xyz")
            #     - R.from_quat(curr_quat).as_euler("xyz")
            # )

            A = R.from_quat([0, 0, 0, 1]).as_matrix()
            B = R.from_quat(curr_quat).as_matrix()

            end_rot_dist = np.arccos((np.trace(A.T @ B) - 1) / 2) * 360 / 2 / np.pi

            result_dict[obj_id] = [
                end_trans_dist,
                end_trans_dist / start_trans_dist,
                end_rot_dist,
            ]

            trial += 1

            # Initialize mp result logging
            # 1: success
            # 0: failure
            # key is [SUCC, NORM_DIST]
            imageio.mimsave(f"{result_dir}/vids/test_{obj_id}.gif", exec_gifs, fps=25)
            imageio.imsave(f"{result_dir}/vids/test_{obj_id}_goal.png", rgb_goal)
            p.disconnect()
            nd = np.linalg.norm(current_xyz - gt_goal_xyz) / np.linalg.norm(
                start_xyz - gt_goal_xyz
            )
            mp_result_dict[obj_id].append(min(1, nd))  # type: ignore

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
