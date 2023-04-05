"""
This is the dataloader file for free-floating object placement task.
i.e. Placing a box into the oven
"""
import multiprocessing
import os
import pickle
import time
from typing import Callable, Dict, List, Optional, Protocol

import numpy as np
import pybullet as p
import torch
import torch.utils.data as td
import torch_geometric.data as tgd
import tqdm
from rpad.partnet_mobility_utils.data import PMObject as PMRawData
from rpad.partnet_mobility_utils.render.pybullet import PMRenderEnv
from rpad.pyg.dataset import SinglePathDataset as SingleObjDataset
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


def randomize_pcd_pose(src_pose, max_r):
    r = np.random.uniform(low=0, high=max_r)
    phi = np.random.uniform(0, np.pi)
    theta = np.random.uniform(0, 2 * np.pi)
    delta_x = r * np.cos(theta) * np.sin(phi)
    delta_y = r * np.sin(theta) * np.sin(phi)
    delta_z = r * np.cos(phi)
    return src_pose + np.array([delta_x, delta_y, delta_z])


class PCData(Protocol):
    id: str  # Object ID.

    pos: torch.Tensor

    x: Optional[torch.Tensor] = None


_dset = None
_args = None


def _init():
    global _dset
    # _dset = SemGoalCondFlowDataset(*_args)
    _dset = GCBCDataset(*_args)


def _sample(d):
    global _dset
    dset = _dset
    (i, env_name) = d
    os.sched_setaffinity(os.getpid(), [i % 128])
    base = f"{env_name}_{dset.nrepeat}.pt"
    outfile = os.path.join(dset.processed_dir, base)
    if os.path.exists(outfile):
        print(f"data exists for {env_name}")
        return False
    else:
        print(f"sampling {dset.nrepeat} times for {env_name}")

    data_list = []
    try:
        for j in range(dset.nrepeat):
            data_list.append(dset.get_sample(i * dset.nrepeat + j))

        data, slices = tgd.InMemoryDataset.collate(data_list)
        torch.save((data, slices), outfile)
    except Exception as e:
        breakpoint()
        print(f"failed for {env_name}")
        return True
    return False


class GCBCDataset(tgd.Dataset):
    def __init__(
        self,
        root: str,
        freefloat_dset_path: str,
        obj_ids=None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        nrepeat: int = 50,
        process: bool = True,
        even_sampling: bool = False,
        randomize_camera: bool = False,
        n_points: Optional[int] = 1200,
    ):
        self.env_names = obj_ids

        # Extract the name.
        self.obj_ids = obj_ids
        self.full_sem_dset = pickle.load(open(SEM_CLASS_DSET_PATH, "rb"))

        self.freefloat_dset_path = freefloat_dset_path
        self.nrepeat = nrepeat
        self.even_sampling = even_sampling
        self.randomize_camera = randomize_camera
        self.n_points = n_points
        self.envs: Dict[str, PMRenderEnv] = {}
        self.goal_envs: Dict[str, PMRenderEnv] = {}
        self.raw_data: Dict[str, PMRawData] = {}
        self.goal_raw_data: Dict[str, PMRawData] = {}
        self.use_processed = process
        self.object_dict = pickle.load(open(ALL_BLOCK_DSET_PATH, "rb"))

        super().__init__(root, transform, pre_transform, pre_filter)

        if self.use_processed:
            self.inmem: td.ConcatDataset = td.ConcatDataset(
                [SingleObjDataset(data_path) for data_path in self.processed_paths]
            )

    @property
    def processed_dir(self) -> str:
        chunk = ""
        if self.randomize_camera:
            chunk += "_random"
        if self.even_sampling:
            chunk += "_even"
        return os.path.join(self.root, f"goal_cond_bc_traj" + chunk)

    @property
    def processed_file_names(self) -> List[str]:
        return [f"{env}_{self.nrepeat}.pt" for env in self.env_names]

    def len(self) -> int:
        return len(self.env_names)

    def process(self):
        if not self.use_processed:
            return

        else:
            global _args
            _args = (
                self.root,
                self.freefloat_dset_path,
                self.obj_ids,
                self.transform,
                self.pre_transform,
                self.pre_filter,
                self.nrepeat,
                False,
                self.even_sampling,
                self.randomize_camera,
                self.n_points,
            )
            # _init()
            # for i, envname in enumerate(self.env_names):
            #     _sample((i, envname))

            pool = multiprocessing.Pool(
                processes=os.cpu_count() - 20, initializer=_init
            )

            failed_attempts = list(
                tqdm.tqdm(
                    pool.imap(_sample, enumerate(self.env_names)),
                    total=len(self.env_names),
                )
            )

            if sum(failed_attempts) > 0:
                raise ValueError("JUST PRINTED THE FAILED ATTEMPTS")

    def get(self, idx: int) -> Data:
        if self.use_processed:
            return self.inmem[idx]
        else:
            return self.get_sample(idx)

    def get_sample(self, idx: int) -> Data:
        idx = idx // self.nrepeat
        obj_id = self.obj_ids[idx % len(self.obj_ids)]

        # Get the trajectory
        traj_name = f"{'_'.join(obj_id.split('_')[:-1])}.npy"
        traj = np.load(os.path.join(self.freefloat_dset_path, traj_name))
        curr_traj_idx = int(obj_id.split("_")[-1])

        if obj_id not in self.envs:
            obs_env = PMRenderEnv(
                obj_id.split("_")[0], self.raw_dir, camera_pos=[-3, 0, 1.2], gui=False
            )
            block = f"{RAVENS_ASSETS}/block/block.urdf"
            obs_block_id = p.loadURDF(
                block, physicsClientId=obs_env.client_id, globalScaling=4
            )
            curr_xyz = traj[curr_traj_idx]
            curr_xyz_p1 = traj[curr_traj_idx + 1]
            p.resetBasePositionAndOrientation(
                obs_block_id,
                posObj=curr_xyz,
                ornObj=[0, 0, 0, 1],
                physicsClientId=obs_env.client_id,
            )
            self.envs[obj_id] = obs_env
            self.raw_data[obj_id] = PMRawData(
                os.path.join(self.raw_dir, obj_id.split("_")[0])
            )

        object_dict = self.object_dict[get_category(obj_id.split("_")[0]).lower()]
        curr_xyz = traj[curr_traj_idx]
        curr_xyz_p1 = traj[curr_traj_idx + 1]

        np.random.seed((os.getpid() * int(time.time())) % 123456789)
        goal_id_list = list(object_dict.keys())
        goal_id = np.random.choice(goal_id_list)
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

        if goal_id not in self.goal_envs:
            goal_env = PMRenderEnv(
                goal_id.split("_")[0], self.raw_dir, camera_pos=[-3, 0, 1.2], gui=False
            )
            self.goal_envs[goal_id] = goal_env
            self.goal_raw_data[goal_id] = PMRawData(
                os.path.join(self.raw_dir, goal_id.split("_")[0])
            )

        obs_env = self.envs[obj_id]
        goal_env = self.goal_envs[goal_id]

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
            obs_env.randomize_camera()

        # Obs data
        P_world, pc_seg_obj, _ = render_input(obs_block_id, obs_env)
        P_world, pc_seg_obj = subsample_pcd(P_world, pc_seg_obj)

        # Goal data
        P_world_goal, pc_seg_obj_goal, _ = render_input(goal_block_id, goal_env)
        P_world_goal, pc_seg_obj_goal = subsample_pcd(P_world_goal, pc_seg_obj_goal)

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

        return goal_data, obs_data
