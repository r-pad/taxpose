"""
This is the dataloader file for free-floating object placement task.
i.e. Placing a box into the oven
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
from torch_geometric.data import Data

from taxpose.datasets.pm_placement import (
    ALL_BLOCK_DSET_PATH,
    RAVENS_ASSETS,
    SEM_CLASS_DSET_PATH,
    get_category,
    render_input,
    subsample_pcd,
)


def articulate_specific_joints(sim, joint_list, amount):
    for i in range(p.getNumJoints(sim.obj_id, sim.client_id)):
        jinfo = p.getJointInfo(sim.obj_id, i, sim.client_id)
        if jinfo[12].decode("UTF-8") in joint_list:
            lower, upper = jinfo[8], jinfo[9]
            angle = amount * (upper - lower) + lower
            p.resetJointState(sim.obj_id, i, angle, 0, sim.client_id)


def randomize_dagger(src_pose, dst_pose, seed: NPSeed = None):
    rng = np.random.default_rng(seed)
    normalized_vec = (dst_pose - src_pose) / np.linalg.norm(dst_pose - src_pose)
    r = rng.uniform(low=0, high=np.linalg.norm(dst_pose - src_pose))
    v = rng.uniform(low=0, high=1, size=3)
    v = v / np.linalg.norm(v)
    if np.dot(normalized_vec, v) < 0:
        v = -v
    v = v * r
    return src_pose + v


class GCDaggerDataset(tgd.Dataset):
    def __init__(
        self,
        root: str,
        freefloat_dset_path: str,
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
        self.freefloat_dset_path = freefloat_dset_path
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
            GCDaggerDataset.processed_base(self.randomize_camera, self.even_sampling),
        )

    @staticmethod
    def processed_base(randomize_camera, even_sampling):
        chunk = ""
        if randomize_camera:
            chunk += "_random"
        if even_sampling:
            chunk += "_even"
        return f"goal_cond_dagger_traj" + chunk

    def len(self) -> int:
        return len(self.env_names)

    def get(self, idx: int) -> Data:
        return self.get_data(self.obj_ids[idx])

    def get_data(self, obj_id: str, seed=NPSeed) -> Data:
        # Get the trajectory
        freefloat_dset = self.freefloat_dset_path
        traj_name = f"{'_'.join(obj_id.split('_')[:-1])}.npy"
        traj = np.load(os.path.join(freefloat_dset, traj_name))

        rng = np.random.default_rng(seed)

        curr_traj_idx = int(obj_id.split("_")[-1]) % (len(traj) - 1)

        obs_env = PMRenderEnv(
            obj_id.split("_")[0], self.raw_dir, camera_pos=[-3, 0, 1.2], gui=False
        )

        block = f"{RAVENS_ASSETS}/block/block.urdf"

        obs_block_id = p.loadURDF(
            block, physicsClientId=obs_env.client_id, globalScaling=4
        )
        bern = rng.uniform()
        curr_xyz_p1 = traj[curr_traj_idx + 1]
        curr_xyz = traj[curr_traj_idx]
        if bern < 0.1:
            curr_xyz = traj[curr_traj_idx]
        else:
            curr_xyz = randomize_dagger(curr_xyz, curr_xyz_p1, rng)
        p.resetBasePositionAndOrientation(
            obs_block_id,
            posObj=curr_xyz,
            ornObj=[0, 0, 0, 1],
            physicsClientId=obs_env.client_id,
        )
        self.raw_data[obj_id] = PMRawData(
            os.path.join(self.raw_dir, obj_id.split("_")[0])
        )

        object_dict = self.object_dict[get_category(obj_id.split("_")[0]).lower()]

        goal_id_list = list(object_dict.keys())
        goal_id = rng.choice(goal_id_list)
        which_goal = obj_id.split("_")[1]
        goal_id = f"{goal_id.split('_')[0]}_{which_goal}"

        partsem = object_dict[goal_id]["partsem"]
        if partsem != "none":
            for mode in self.full_sem_dset:
                if partsem in self.full_sem_dset[mode]:
                    if obj_id.split("_")[0] in self.full_sem_dset[mode][partsem]:
                        move_joints = self.full_sem_dset[mode][partsem][
                            obj_id.split("_")[0]
                        ]

            obj_link_id = object_dict[obj_id.split("_")[0] + f"_{which_goal}"]["ind"]
            try:
                obj_id_links_tomove = move_joints[obj_link_id]
            except:
                obj_id_links_tomove = "link_0"
            for mode in self.full_sem_dset:
                if partsem in self.full_sem_dset[mode]:
                    if goal_id.split("_")[0] in self.full_sem_dset[mode][partsem]:
                        move_joints = self.full_sem_dset[mode][partsem][
                            goal_id.split("_")[0]
                        ]
            goal_link_id = object_dict[goal_id]["ind"]
            try:
                goal_id_links_tomove = move_joints[goal_link_id]
            except:
                goal_id_links_tomove = "link_0"

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

        goal_block = f"{RAVENS_ASSETS}/block/block.urdf"
        goal_block_id = p.loadURDF(
            goal_block, physicsClientId=goal_env.client_id, globalScaling=4
        )
        goal_xyz = [
            object_dict[goal_id]["x"],
            object_dict[goal_id]["y"],
            object_dict[goal_id]["z"],
        ]
        # Goal relabeling
        p.resetBasePositionAndOrientation(
            goal_block_id,
            posObj=goal_xyz,
            ornObj=[0, 0, 0, 1],
            physicsClientId=goal_env.client_id,
        )

        if self.randomize_camera:
            obs_env.set_camera("random", rng)

        # Obs data
        P_world, pc_seg_obj, rgb = render_input(obs_block_id, obs_env)
        P_world, pc_seg_obj = subsample_pcd(P_world, pc_seg_obj, rng)

        # Goal data
        P_world_goal, pc_seg_obj_goal, _ = render_input(goal_block_id, goal_env)
        P_world_goal, pc_seg_obj_goal = subsample_pcd(
            P_world_goal, pc_seg_obj_goal, rng
        )

        action = curr_xyz_p1 - curr_xyz
        flow = np.tile(action, (P_world.shape[0], 1))
        flow[pc_seg_obj != 99] = [0, 0, 0]

        mask = (~(flow == 0.0).all(axis=-1)).astype(int)

        mask_goal = pc_seg_obj_goal == 99

        output_len = min(len(P_world), len(P_world_goal))
        # Downsample.
        obs_data = Data(
            id=obj_id,
            pos=torch.from_numpy(P_world[:output_len]).float(),
            flow=torch.from_numpy(flow[:output_len]).float(),
            mask=torch.from_numpy(mask[:output_len]).float(),
            action=torch.from_numpy(action).float(),
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


def create_gcdagger_dataset(
    root,
    freefloat_dset_path,
    obj_ids,
    even_sampling=False,
    randomize_camera=False,
    n_points=1200,
    n_repeat=1,
    n_workers=30,
    n_proc_per_worker=2,
    seed=0,
) -> CachedByKeyDataset[GCDaggerDataset]:
    """Creates the GCBC dataset."""
    return CachedByKeyDataset(
        dset_cls=GCDaggerDataset,
        dset_kwargs={
            "root": root,
            "freefloat_dset_path": freefloat_dset_path,
            "obj_ids": obj_ids,
            "even_sampling": even_sampling,
            "randomize_camera": randomize_camera,
            "n_points": n_points,
        },
        data_keys=obj_ids,
        root=root,
        processed_dirname=GCDaggerDataset.processed_base(
            randomize_camera, even_sampling
        ),
        n_repeat=n_repeat,
        n_workers=n_workers,
        n_proc_per_worker=n_proc_per_worker,
        seed=seed,
    )