"""
This is the dataloader file for free-floating objects goal inference task.
i.e. Infer goal based on a demo.
"""
import os
import pickle
from typing import Callable, Dict, Optional

import numpy as np
import pybullet as p
import torch
import torch_geometric.data as tgd
from rpad.core.distributed import NPSeed
from rpad.partnet_mobility_utils.data import PMObject as PMRawData
from rpad.partnet_mobility_utils.render.pybullet import PMRenderEnv
from rpad.pyg.dataset import CachedByKeyDataset
from scipy.spatial.transform import Rotation as R
from torch_geometric.data import Data

from taxpose.datasets.pm_placement import (
    ACTION_OBJS,
    ALL_BLOCK_DSET_PATH,
    SEM_CLASS_DSET_PATH,
    get_category,
    is_action_pose_valid,
    randomize_block_pose,
    render_input,
    subsample_pcd,
)
from taxpose.training.pm_baselines.bc_dataset import articulate_specific_joints


def get_random_action_obj(seed: NPSeed = None):
    rng = np.random.default_rng(seed)
    action_obj = ACTION_OBJS[rng.choice(sorted(list(ACTION_OBJS.keys())))]
    return action_obj.urdf, action_obj.random_scale(rng)


class GCGoalFlowDataset(tgd.Dataset):
    def __init__(
        self,
        root: str,
        obj_ids=None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        even_sampling: bool = False,
        randomize_camera: bool = False,
        n_points: Optional[int] = 1200,
    ):
        self.env_names = obj_ids

        # Extract the name.
        self.obj_ids = obj_ids
        with open(SEM_CLASS_DSET_PATH, "rb") as f:
            self.full_sem_dset = pickle.load(f)
        self.even_sampling = even_sampling
        self.randomize_camera = randomize_camera
        self.n_points = n_points
        self.raw_data: Dict[str, PMRawData] = {}
        self.goal_raw_data: Dict[str, PMRawData] = {}
        with open(ALL_BLOCK_DSET_PATH, "rb") as f:
            self.object_dict = pickle.load(f)
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def processed_dir(self) -> str:
        return os.path.join(
            self.root,
            GCGoalFlowDataset.processed_base(self.randomize_camera, self.even_sampling),
        )

    @staticmethod
    def processed_base(randomize_camera, even_sampling):
        chunk = ""
        if randomize_camera:
            chunk += "_random"
        if even_sampling:
            chunk += "_even"
        return f"taxpose_goal_flow" + chunk

    def len(self) -> int:
        return len(self.env_names)

    def get(self, idx: int) -> Data:
        return self.get_data(self.obj_ids[idx])

    def get_data(self, obj_id: str, seed: NPSeed = None) -> Data:
        rng = np.random.default_rng(seed=seed)
        object_dict = self.object_dict[get_category(obj_id.split("_")[0]).lower()]

        obs_env = PMRenderEnv(
            obj_id.split("_")[0], self.raw_dir, camera_pos=[-3, 0, 1.2], gui=False
        )

        action_obj = ACTION_OBJS["block"]
        block, rand_scale = action_obj.urdf, action_obj.random_scale(rng)
        obs_block_id = p.loadURDF(
            block, physicsClientId=obs_env.client_id, globalScaling=rand_scale
        )
        self.raw_data[obj_id] = PMRawData(
            os.path.join(self.raw_dir, obj_id.split("_")[0])
        )

        # Randomly select demo object and which goal
        goal_id_list = list(object_dict.keys())
        goal_id = rng.choice(goal_id_list)
        which_goal = goal_id.split("_")[1]

        partsem = object_dict[goal_id]["partsem"]
        if partsem != "none":
            for mode in self.full_sem_dset:
                if partsem in self.full_sem_dset[mode]:
                    if obj_id.split("_")[0] in self.full_sem_dset[mode][partsem]:
                        move_joints = self.full_sem_dset[mode][partsem][
                            obj_id.split("_")[0]
                        ]

            obj_link_id = object_dict[obj_id.split("_")[0] + f"_{which_goal}"]["ind"]
            obj_id_links_tomove = move_joints[obj_link_id]
            for mode in self.full_sem_dset:
                if partsem in self.full_sem_dset[mode]:
                    if goal_id.split("_")[0] in self.full_sem_dset[mode][partsem]:
                        move_joints = self.full_sem_dset[mode][partsem][
                            goal_id.split("_")[0]
                        ]
            goal_link_id = object_dict[goal_id]["ind"]
            goal_id_links_tomove = move_joints[goal_link_id]

        goal_env = PMRenderEnv(
            goal_id.split("_")[0], self.raw_dir, camera_pos=[-3, 0, 1.2], gui=False
        )
        self.goal_raw_data[goal_id] = PMRawData(
            os.path.join(self.raw_dir, goal_id.split("_")[0])
        )

        # Open the joints.
        if partsem != "none":
            articulate_specific_joints(obs_env, obj_id_links_tomove, 0.9)
            articulate_specific_joints(goal_env, goal_id_links_tomove, 0.9)

        goal_block, goal_rand_scale = get_random_action_obj()
        goal_block_id = p.loadURDF(
            goal_block,
            physicsClientId=goal_env.client_id,
            globalScaling=goal_rand_scale,
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

        if self.randomize_camera:
            obs_env.set_camera("random", rng)

        valid_start = False
        while not valid_start:
            obs_curr_xyz = randomize_block_pose(rng)
            angle = rng.uniform(-60, 60)
            start_ort = R.from_euler("z", angle, degrees=True).as_quat()
            p.resetBasePositionAndOrientation(
                obs_block_id,
                posObj=obs_curr_xyz,
                ornObj=start_ort,
                physicsClientId=obs_env.client_id,
            )
            valid_start = is_action_pose_valid(obs_block_id, obs_env)

        # Obs data and subsample
        P_world, pc_seg_obj, _ = render_input(obs_block_id, obs_env)
        P_world, pc_seg_obj = subsample_pcd(P_world, pc_seg_obj, rng)

        # Goal data
        P_world_goal, pc_seg_obj_goal, _ = render_input(goal_block_id, goal_env)
        P_world_goal, pc_seg_obj_goal = subsample_pcd(
            P_world_goal, pc_seg_obj_goal, rng
        )

        """
        Flow formulation here: point from curr to GT goal pose.
        """
        pmobj_idx = obj_id.split("_")[0]
        obs_goal = np.array(
            [
                object_dict[f"{pmobj_idx}_{which_goal}"]["x"],
                object_dict[f"{pmobj_idx}_{which_goal}"]["y"],
                object_dict[f"{pmobj_idx}_{which_goal}"]["z"],
            ]
        )
        uniflow = obs_goal - obs_curr_xyz
        flowed_action = P_world[pc_seg_obj == 99] @ R.from_quat(start_ort).as_matrix()
        flowed_action = (
            flowed_action
            + (P_world[pc_seg_obj == 99].mean(axis=0) - flowed_action.mean(axis=0))
            + uniflow
        )
        flow = np.zeros_like(P_world)
        flow[pc_seg_obj == 99] = flowed_action - P_world[pc_seg_obj == 99]

        mask = (~(flow == 0.0).all(axis=-1)).astype(int)

        mask_goal = (pc_seg_obj_goal == 99).astype(int)

        output_len = min(len(P_world), len(P_world_goal))
        # Downsample.
        obs_data = Data(
            id=obj_id,
            pos=torch.from_numpy(P_world[:output_len]).float(),
            flow=torch.from_numpy(flow[:output_len]).float(),
            mask=torch.from_numpy(mask[:output_len]).float(),
            x=torch.from_numpy(mask[:output_len].reshape((-1, 1))).float(),
        )
        goal_data = Data(
            id=goal_id,
            pos=torch.from_numpy(P_world_goal[:output_len]).float(),
            mask=torch.from_numpy(mask_goal[:output_len]).float(),
            x=torch.from_numpy(mask_goal[:output_len].reshape((-1, 1))).float(),
        )
        obs_env.close()
        goal_env.close()

        return goal_data, obs_data


def create_gf_dataset(
    root,
    obj_ids,
    even_sampling=False,
    randomize_camera=False,
    n_points=1200,
    n_repeat=1,
    n_workers=30,
    n_proc_per_worker=2,
    seed=0,
) -> CachedByKeyDataset[GCGoalFlowDataset]:
    """Creates the GCBC dataset."""
    return CachedByKeyDataset(
        dset_cls=GCGoalFlowDataset,
        dset_kwargs={
            "root": root,
            "obj_ids": obj_ids,
            "even_sampling": even_sampling,
            "randomize_camera": randomize_camera,
            "n_points": n_points,
        },
        data_keys=obj_ids,
        root=root,
        processed_dirname=GCGoalFlowDataset.processed_base(
            randomize_camera, even_sampling
        ),
        n_repeat=n_repeat,
        n_workers=n_workers,
        n_proc_per_worker=n_proc_per_worker,
        seed=seed,
    )
