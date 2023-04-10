import json
import os
import pickle
from copy import deepcopy

import imageio
import numpy as np
import pybullet as p
import torch
from chamferdist import ChamferDistance

# from part_embedding.goal_inference.motion_planning import motion_planning_fcl
# from part_embedding.goal_inference.plots import create_plot, dcp_correspondence_plot
# from part_embedding.goal_inference.train_collision_net import CollisionNet, GFENetParams
from rpad.partnet_mobility_utils.render.pybullet import PMRenderEnv
from scipy.spatial.transform import Rotation as R
from torch import nn
from tqdm import tqdm

from taxpose.datasets.pm_placement import (
    SEEN_CATS,
    TAXPOSE_ROOT,
    UNSEEN_CATS,
    get_category,
    get_dataset_ids_all,
    render_input,
)
from taxpose.training.pm_baselines.flow_model import FlowNet as GoalInfFlowNet
from taxpose.training.pm_baselines.flow_model import (
    FlowNetParams as GoalInfFlowNetParams,
)
from taxpose.training.pm_baselines.test_bc import quaternion_sum
from taxpose.training.pm_baselines.test_bc.test_gc_traj_flow import rigid_transform_3D

# from part_embedding.goal_inference.dcp_residual import DCPResidual
# from part_embedding.goal_inference.dcp_utils import (
#     create_test_env,
#     get_demo_from_list,
#     get_obs_data_from_env,
# )


"""
This file loads a trained goal inference model and tests the rollout using motion planning in simulation.
"""


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


def load_model(method: str, exp_name: str, in_dim: int):
    d = os.path.join(
        os.getcwd(),
        f"checkpoints/{method}/{exp_name}/",
    )
    ckpt = os.listdir(d)[0]
    param = GoalInfFlowNetParams()
    net: nn.Module
    if not "dcp" in method:
        param.in_dim = in_dim
        net = GoalInfFlowNet.load_from_checkpoint(
            f"{d}/{ckpt}",
            p=param,
        )
    else:
        param.in_dim = 1
        net = DCPResidual.load_from_checkpoint(f"{d}/{ckpt}", p=param)
    return net


def load_collision_model(exp_name: str) -> nn.Module:
    d = os.path.join(
        os.getcwd(),
        f"checkpoints/goal_inference_collision_net/{exp_name}/",
    )
    ckpt = os.listdir(d)[0]
    param = GFENetParams()
    net: CollisionNet = CollisionNet.load_from_checkpoint(
        f"{d}/{ckpt}",
        params=param,
    )
    return net


def get_demo(
    goal_id_list: list,
    full_sem_dset: dict,
    object_dict: dict,
    snap_to_surface=True,
    full_obj=True,
    even_downsample=True,
    WHICH=None,
):
    # This creates the test env for demonstration.
    return get_demo_from_list(
        goal_id_list,
        full_sem_dset,
        object_dict,
        snap_to_surface,
        full_obj,
        even_downsample,
        WHICH=WHICH,
    )


def infer_goal(goalinf_model, P_world, obj_mask, P_world_goal, goal_mask, isflow=True):
    # This infers the goal using the goalinf_modem obs PCD (P_world), and demo PCD (P_world_goal)

    # IS_FLOW: Boolean value indicating if the model is flow-based or not. If TRUE: add pred flow.
    # Otherwise, output goal directly.
    pred_ = goalinf_model.predict(
        torch.from_numpy(P_world).float(),
        torch.from_numpy(obj_mask).float(),
        torch.from_numpy(P_world_goal).float(),
        torch.from_numpy(goal_mask).float(),
    )
    # import trimesh

    # scene = trimesh.Scene(
    #     [
    #         trimesh.points.PointCloud(P_world[obj_mask], colors=(255, 0, 0)),
    #         trimesh.points.PointCloud(P_world[~obj_mask]),
    #     ]
    # )
    # scene.show()
    # scene = trimesh.Scene(
    #     [
    #         trimesh.points.PointCloud(P_world_goal[goal_mask], colors=(255, 0, 0)),
    #         trimesh.points.PointCloud(P_world_goal[~goal_mask]),
    #     ]
    # )
    # scene.show()
    if isflow:
        pred_: np.ndarray = pred_.cpu().numpy()
        # Mask out the pred_flow, not necessary if flow model is good
        pred_[~obj_mask] = 0
        inferred_goal: np.ndarray = P_world + pred_
        return inferred_goal
    else:
        pred_after_ref = pred_[0].cpu().numpy()
        pred_before_ref = pred_[1].cpu().numpy()
        inferred_goal_after_refinement, action_before_refinement = (
            pred_after_ref.astype(np.float64),
            pred_before_ref.astype(np.float64),
        )
        inferred_goal_before_refinement = deepcopy(inferred_goal_after_refinement)
        inferred_goal_before_refinement[:200] = action_before_refinement
        if len(pred_) > 2:
            src_corr = pred_[2]
            corr_weights = pred_[3]
            return (
                inferred_goal_after_refinement,
                inferred_goal_before_refinement,
                src_corr,
                corr_weights,
            )
        return inferred_goal_after_refinement, inferred_goal_before_refinement


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
    parser.add_argument("--method", type=str, default="goal_inference_naive")
    parser.add_argument("--model", type=str)
    parser.add_argument("--coll_model", type=str)
    parser.add_argument("--mask", action="store_true")
    parser.add_argument("--mp", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--indist", type=bool, default=True)
    parser.add_argument("--postfix", type=str)
    parser.add_argument("--snap", action="store_true")
    args = parser.parse_args()
    objcat = args.cat
    method = args.method
    postfix = args.postfix
    snap_to_surface = args.snap
    isflow = True
    if "dcp" in method:
        isflow = False
    expname = args.model
    # coll_model = args.coll_model
    mask = args.mask
    start_ind = args.start
    mp = args.mp
    in_dist = args.indist

    if mask:
        in_dim = 1
    else:
        in_dim = 0

    # Decide what collision model to use:
    # if coll_model == "learned":
    #     motion_planning = motion_planning_learned
    # elif coll_model == "fcl":
    #     motion_planning = motion_planning_fcl
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
    goalinf_model = load_model(method, expname, in_dim).cuda()

    # Get collision model
    coll_net = None

    # Create result directory
    result_dir_fcl = (
        TAXPOSE_ROOT
        / f"part_embedding/goal_inference/baselines_rotation/rollouts/{method}_{objcat}_fcl_coll_{expname}_{postfix}"
    )
    result_dir = result_dir_fcl
    if not os.path.exists(result_dir_fcl):
        print("Creating result directory for rollouts")
        os.makedirs(result_dir_fcl, exist_ok=True)
        os.makedirs(f"{result_dir_fcl}/vids", exist_ok=True)

    # If we are doing "test" then we sample 10 times per object
    objs = get_ids(objcat.capitalize())
    if "7292" in objs:
        objs.remove("7292")

    result_dict = {}
    if not isflow:
        result_dict_intermediate = {}
    mp_result_dict = {}

    trial_len = 10
    trial_start = start_ind % trial_len

    for o in tqdm(objs[start_ind // trial_len :]):
        # if get_category(o.split("_")[0]) != "Drawer":
        #     print("skipping category")
        #     continue
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
                demo_id_list,
                full_sem_dset,
                object_dict,
                snap_to_surface=snap_to_surface,
                WHICH=postfix,
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
                    isflow=isflow,
                )
            except IndexError:
                breakpoint()
            current_xyz = np.array([start_xyz[0], start_xyz[1], start_xyz[2]])
            current_quat = np.array(
                [start_quat[0], start_quat[1], start_quat[2], start_quat[3]]
            )
            if not isflow:
                pred_R, pred_t = rigid_transform_3D(
                    P_world[pc_seg_obj == 99].T, (inferred_goal[0][pc_seg_obj == 99]).T
                )
            else:
                pred_R, pred_t = rigid_transform_3D(
                    P_world[pc_seg_obj == 99].T, (inferred_goal[pc_seg_obj == 99]).T
                )
            current_xyz = current_xyz + pred_t.reshape(
                3,
            )
            current_quat = quaternion_sum(current_quat, R.from_matrix(pred_R).as_quat())
            current_quat = R.from_matrix(current_quat).as_quat()

            if isflow:
                start_trans_dist = np.linalg.norm(start_xyz - gt_goal_xyz)
                end_trans_dist = np.linalg.norm(current_xyz - gt_goal_xyz)
                A = R.from_quat([0, 0, 0, 1]).as_matrix()
                B = R.from_quat(current_quat).as_matrix()

                end_rot_dist = np.arccos((np.trace(A.T @ B) - 1) / 2) * 360 / 2 / np.pi

                result_dict[obj_id] = [
                    end_trans_dist,
                    end_trans_dist / start_trans_dist,
                    end_rot_dist,
                ]
            else:
                start_trans_dist = np.linalg.norm(start_xyz - gt_goal_xyz)
                end_trans_dist = np.linalg.norm(current_xyz - gt_goal_xyz)
                A = R.from_quat([0, 0, 0, 1]).as_matrix()
                B = R.from_quat(current_quat).as_matrix()

                end_rot_dist = np.arccos((np.trace(A.T @ B) - 1) / 2) * 360 / 2 / np.pi

                result_dict[obj_id] = [
                    end_trans_dist,
                    end_trans_dist / start_trans_dist,
                    end_rot_dist,
                ]

                # DCP ONLY
                current_xyz_int = np.array([start_xyz[0], start_xyz[1], start_xyz[2]])
                current_quat_int = np.array(
                    [start_quat[0], start_quat[1], start_quat[2], start_quat[3]]
                )
                pred_R_int, pred_t_int = rigid_transform_3D(
                    P_world[pc_seg_obj == 99].T, (inferred_goal[1][pc_seg_obj == 99]).T
                )
                current_xyz_int = current_xyz_int + pred_t_int.reshape(
                    3,
                )
                current_quat_int = quaternion_sum(
                    current_quat_int, R.from_matrix(pred_R_int).as_quat()
                )
                current_quat_int = R.from_matrix(current_quat_int).as_quat()
                end_trans_dist_int = np.linalg.norm(current_xyz_int - gt_goal_xyz)
                A = R.from_quat([0, 0, 0, 1]).as_matrix()
                B = R.from_quat(current_quat_int).as_matrix()

                end_rot_dist_int = (
                    np.arccos((np.trace(A.T @ B) - 1) / 2) * 360 / 2 / np.pi
                )

                result_dict[obj_id] = [
                    end_trans_dist_int,
                    end_trans_dist_int / start_trans_dist,
                    end_rot_dist_int,
                ]

            scene_pose = np.eye(4)
            scene_xyz, scene_rot = p.getBasePositionAndOrientation(
                obs_env.obj_id, physicsClientId=obs_env.client_id
            )
            scene_pose[:3, -1] = np.array(scene_xyz)
            scene_pose[:3, :3] = R.from_quat(np.array(scene_rot)).as_matrix()

            # Motion planning
            if mp:
                if isflow:
                    path_list_fcl = motion_planning_fcl(
                        inferred_goal,
                        P_world,
                        P_world_full,
                        pc_seg_obj == 99,
                        pc_seg_obj_full == 99,
                        start_full_pose,
                        scene_pose,
                        coll_net,
                    )
                else:
                    reordered_goal = deepcopy(P_world)
                    reordered_goal[pc_seg_obj == 99] = inferred_goal[0][:200]
                    path_list_fcl = motion_planning_fcl(
                        reordered_goal,
                        P_world,
                        P_world_full,
                        pc_seg_obj == 99,
                        pc_seg_obj_full == 99,
                        start_full_pose,
                        scene_pose,
                        coll_net,
                    )
                if path_list_fcl is None:
                    p.disconnect()
                    continue

            result_dirs = [result_dir_fcl]
            trial += 1
            for i in range(1):
                if mp:
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
                    imageio.imsave(
                        f"{result_dir}/vids/test_{obj_id}_goal.png", rgb_goal
                    )
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
                if not isflow:
                    goalinf_res_file_intermediate = open(
                        os.path.join(result_dir, f"rollout_goalinf_res_before_ref.txt"),
                        "a",
                    )
                    print(
                        f"{obj_id}: {result_dict_intermediate[obj_id]}",
                        file=goalinf_res_file_intermediate,
                    )
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
