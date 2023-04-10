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
    SEEN_CATS,
    UNSEEN_CATS,
    get_category,
    get_dataset_ids_all,
    render_input,
    subsample_pcd,
)
from taxpose.training.pm_baselines.flow_model import FlowNet as TrajFlowNet
from taxpose.training.pm_baselines.test_bc import create_test_env, quaternion_sum

"""
This file loads a trained goal inference model and tests the rollout using motion planning in simulation.
"""


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


def load_model(method: str, exp_name: str) -> TrajFlowNet:
    d = os.path.join(
        os.getcwd(),
        f"checkpoints/{method}/{exp_name}/",
    )
    ckpt = os.listdir(d)[0]
    net: TrajFlowNet = TrajFlowNet.load_from_checkpoint(
        f"{d}/{ckpt}",
    ).cuda()
    return net


def get_demo(goal_id: str, full_sem_dset: dict, object_dict: dict):
    # This creates the test env for demonstration.

    goal_env = PMRenderEnv(
        goal_id.split("_")[0],
        os.path.expanduser("~/partnet-mobility/raw"),
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

    goal_block = os.path.expanduser(
        "/home/harry/discriminative_embeddings/third_party/ravens/ravens/environments/assets/block/block.urdf"
    )
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cat", type=str)
    parser.add_argument("--method", type=str, default="dgcnn_traj_flow")
    parser.add_argument("--model", type=str)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--indist", type=bool, default=True)
    parser.add_argument("--postfix", type=str)
    args = parser.parse_args()
    objcat = args.cat
    method = args.method
    expname = args.model
    start_ind = args.start
    in_dist = args.indist
    postfix = args.postfix

    # Get which joint to open
    full_sem_dset = pickle.load(
        open(
            os.path.expanduser(
                "~/discriminative_embeddings/goal_inf_dset/sem_class_transfer_dset_more.pkl"
            ),
            "rb",
        )
    )
    object_dict_meta = pickle.load(
        open(
            os.path.expanduser(
                f"~/discriminative_embeddings/goal_inf_dset/{objcat}_block_dset_multi.pkl"
            ),
            "rb",
        )
    )

    # Get goal inference model
    bc_model = load_model(method, expname)

    # Create result directory
    result_dir = f"part_embedding/goal_inference/baselines_rotation/rollouts/{objcat}_{method}_{postfix}"
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
    which_goal = postfix

    for o in tqdm(objs[start_ind // 20 :]):
        if objcat == "all":
            object_dict = object_dict_meta[get_category(o.split("_")[0]).lower()]
        else:
            object_dict = object_dict_meta
        # Get demo ids list
        demo_id_list = list(object_dict.keys())
        trial = 0
        while trial < 8:
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

            demo_id = np.random.choice(demo_id_list)
            demo_id = f"{demo_id.split('_')[0]}_{which_goal}"
            if demo_id == "7263_1":
                continue
            if demo_id not in demo_id_list:
                trial += 1
                continue
            gt_goal_xyz = [
                object_dict[f"{o}_{which_goal}"]["x"],
                object_dict[f"{o}_{which_goal}"]["y"],
                object_dict[f"{o}_{which_goal}"]["z"],
            ]

            # Create obs env
            obs_env, obs_block_id = create_test_env(
                obj_id, full_sem_dset, object_dict, which_goal, in_dist
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
            current_quat = np.array(
                [start_quat[0], start_quat[1], start_quat[2], start_quat[3]]
            )
            exec_gifs = []
            mp_result_dict[obj_id] = [1]
            for t in range(60):
                # Obtain observation data
                p.resetBasePositionAndOrientation(
                    obs_block_id,
                    posObj=current_xyz,
                    ornObj=current_quat,
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
                    curr_obj = P_world[pc_seg_obj == 99]
                    next_obj = curr_obj + pred_flow[pc_seg_obj == 99]
                    pred_R, pred_t = rigid_transform_3D(curr_obj.T, next_obj.T)
                    current_xyz = current_xyz + pred_t.reshape(
                        3,
                    )
                    current_quat = quaternion_sum(
                        current_quat, R.from_matrix(pred_R).as_quat()
                    )
                    current_quat = R.from_matrix(current_quat).as_quat()

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
            B = R.from_quat(current_quat).as_matrix()

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
