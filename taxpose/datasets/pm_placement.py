import itertools
import pickle
from typing import Dict, List, Literal, Optional, Protocol, Sequence, Tuple, Union, cast

import dgl.geometry
import numpy as np
import pybullet as p
import torch
import torch_cluster
import torch_geometric.data as tgd
from rpad.partnet_mobility_utils.render.pybullet import PMRenderEnv, get_obj_z_offset
from rpad.pyg.dataset import CachedByKeyDataset, NPSeed
from scipy.spatial.transform import Rotation as R

import taxpose.datasets.pm_splits as splits
from taxpose.datasets.pm_utils import (
    ACTION_OBJS,
    GOAL_DATA_PATH,
    SNAPPED_GOAL_FILE,
    TAXPOSE_ROOT,
    ActionObj,
)

SceneID = Union[Tuple[str, str, str], Tuple[str, str, str, str]]

SEM_CLASS_DSET_PATH = GOAL_DATA_PATH / "sem_class_transfer_dset_more.pkl"
ACTION_CLOUD_DIR = (
    TAXPOSE_ROOT / "taxpose" / "datasets" / "pm_data" / "action_point_clouds"
)


# TODO: move to splits.
def __id_to_cat():
    id_to_cat = {}
    for cat, cat_dict in splits.split_data["train"].items():
        for _, obj_ids in cat_dict.items():
            for obj_id in obj_ids:
                id_to_cat[obj_id] = cat
    return id_to_cat


CATEGORIES = __id_to_cat()


class GIData(Protocol):
    mode: Literal["obs", "goal"]

    # Action Info
    action_id: str
    goal_id: str
    action_pos: torch.FloatTensor
    t_action_anchor: Optional[torch.FloatTensor]
    R_action_anchor: Optional[torch.FloatTensor]
    flow: Optional[torch.FloatTensor]

    # Anchor Info
    obj_id: str
    anchor_pos: torch.FloatTensor
    loc: Optional[int]


def filter_bad_scenes(scenes: Sequence[SceneID], mode) -> List[SceneID]:
    scenes_set = set(scenes)
    if mode == "goal":
        bad_scenes = splits.BAD_GOAL_SCENES
    else:
        bad_scenes = splits.BAD_OBS_SCENES

    for bad_scene in bad_scenes:
        if bad_scene in scenes_set:
            scenes_set.remove(bad_scene)
    scenes = list(scenes_set)
    return scenes


def default_scenes(split, mode="obs", only_cat="all", goal="all") -> List[SceneID]:
    scenes = []

    # Get all the objs for that split, binned by category.
    cat_dict = {
        cat.lower(): split_dict[split]
        for cat, split_dict in splits.split_data["train"].items()
    }

    if goal == "all":
        goal_ids = ["0", "1", "2", "3"]
    else:
        goal_ids = [goal]

    # Get all the
    for cat, obj_ids in cat_dict.items():
        if only_cat != "all" and cat != only_cat:
            continue
        for obj_id in obj_ids:
            for goal_id in goal_ids:
                key = f"{obj_id}_{goal_id}"
                if key in splits.all_objs[cat]:
                    for action_id in list(ACTION_OBJS.keys()):
                        scenes.append((obj_id, action_id, goal_id))

    return filter_bad_scenes(scenes, mode)


def randomize_block_pose(seed: NPSeed = None):
    rng = np.random.default_rng(seed)
    randomized_pose = np.array(
        [
            rng.uniform(low=-1, high=0.5),
            rng.uniform(low=-1.4, high=1.4),
            rng.uniform(low=0.1, high=0.2),
        ]
    )
    return randomized_pose


def base_from_bottom(body_id, env: PMRenderEnv, goal_bottom_pos):
    # Compute the z offset between the base frame and the bottom of the object.
    curr_base_pos, curr_ori = p.getBasePositionAndOrientation(
        body_id, physicsClientId=env.client_id
    )
    obj_lowest_z = get_obj_z_offset(body_id, env.client_id, starting_min=np.inf)
    z_diff = curr_base_pos[2] - obj_lowest_z

    goal_base_pos = goal_bottom_pos[0], goal_bottom_pos[1], goal_bottom_pos[2] + z_diff

    return goal_base_pos


def find_link_index_to_open(full_sem_dset, partsem, obj_id, object_dict, goal_id):
    move_joints = None
    for mode in full_sem_dset:
        if partsem in full_sem_dset[mode]:
            if obj_id in full_sem_dset[mode][partsem]:
                move_joints = full_sem_dset[mode][partsem][obj_id]

    assert move_joints is not None
    link_id = object_dict[f"{obj_id}_{goal_id}"]["ind"]
    links_tomove = move_joints[link_id]

    return links_tomove


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


def find_valid_action_initial_pose(
    action_body_id, env, seed: NPSeed = None
) -> np.ndarray:
    valid_pos = False
    i = 0
    MAX_ATTEMPTS = 1000
    while not valid_pos:
        action_pos = randomize_block_pose(seed)

        p.resetBasePositionAndOrientation(
            action_body_id,
            posObj=action_pos,
            ornObj=[0, 0, 0, 1],
            physicsClientId=env.client_id,
        )
        # valid_pos = is_action_pose_valid(action_body_id, env)
        valid_pos = not has_collisions(action_body_id, env)
        i += 1

        if i > MAX_ATTEMPTS:
            raise ValueError("unable to sample, invalid start")

    return action_pos  # type: ignore


def load_action_obj_with_valid_scale(
    action_obj: ActionObj, action_pos, env, seed: NPSeed = None
):
    valid_goal = False
    scale = None
    action_body_id = None
    i = 0
    MAX_ATTEMPTS = 20
    while not valid_goal:
        # Sample a new scale.
        scale = action_obj.random_scale(seed)

        action_body_id = p.loadURDF(
            action_obj.urdf,
            globalScaling=scale,
            physicsClientId=env.client_id,
        )
        p.resetBasePositionAndOrientation(
            action_body_id,
            posObj=action_pos,
            ornObj=[0, 0, 0, 1],
            physicsClientId=env.client_id,
        )
        # Check to see if the object is valid. If not, remove and try again.
        valid_goal = is_action_pose_valid(action_body_id, env, n_valid_points=1)

        if not valid_goal:
            # Get rid of the body.
            p.removeBody(action_body_id, physicsClientId=env.client_id)

        i += 1

        if i > MAX_ATTEMPTS:
            raise ValueError("unable to sample, invalid goal")

    return action_body_id, scale


def render_input_new(action_id, sim: PMRenderEnv):
    rgb, _, _, _, P_world, pc_seg, _ = sim.render(link_seg=False)

    is_obj = pc_seg != -1
    P_world = P_world[is_obj]
    pc_seg = pc_seg[is_obj]

    action_mask = pc_seg == action_id
    return P_world, pc_seg, rgb, action_mask


def downsample_pcd_fps(pcd, n, use_dgl=True, seed=None):
    rng = np.random.default_rng(seed)

    if len(pcd) <= n:
        if len(pcd) > 0:
            return torch.arange(0, len(pcd))
        else:
            raise ValueError(f"WHAT IS GOING ON, len(pcd) = {len(pcd)}, n={n}")
    if use_dgl:
        if len(pcd.shape) == 2:
            pcd = pcd.unsqueeze(0)
        start_idx = rng.choice(len(pcd))
        ixs = dgl.geometry.farthest_point_sampler(pcd, n, start_idx=start_idx)
        ixs = ixs.squeeze()
    else:
        ratio = n / len(pcd)
        # Random shuffle.
        pcd = pcd[rng.permutation(len(pcd))]
        ixs = torch_cluster.fps(pcd, None, ratio, random_start=False)
    return ixs


class PlaceDataset(tgd.Dataset):
    def __init__(
        self,
        root: str,
        scene_ids: Optional[List[SceneID]] = None,
        randomize_camera: bool = False,
        mode: Literal["obs", "goal"] = "obs",
        snap_to_surface: bool = False,
        full_obj: bool = False,
        even_downsample: bool = False,
        rotate_anchor: bool = False,
    ):
        if scene_ids is None:
            scene_ids = default_scenes("train", mode=mode)

        # Whether we're sampling a goal or not.
        self.mode = mode

        # Keep around a list of (obj_id, action_id, goal_id) pairs.
        self.scene_ids = scene_ids

        # IDK what this really is....
        self.full_sem_dset = pickle.load(open(SEM_CLASS_DSET_PATH, "rb"))

        self.randomize_camera = randomize_camera
        self.snap_to_surface = snap_to_surface
        self.full_obj = full_obj
        self.even_downsample = even_downsample
        self.rotate_anchor = rotate_anchor

        super().__init__(root)

        with open(SNAPPED_GOAL_FILE, "rb") as f:
            self.snapped_goal_dict = pickle.load(f)

        self.class_map: Dict[str, Dict[str, List[SceneID]]] = {}
        for scene_id in self.scene_ids:
            obj_id, action_id, goal_id = scene_id[0], scene_id[1], scene_id[2]
            cat = CATEGORIES[obj_id]
            if cat not in self.class_map:
                self.class_map[cat] = {}
            if goal_id not in self.class_map[cat]:
                self.class_map[cat][goal_id] = []
            self.class_map[cat][goal_id].append(scene_id)

    @staticmethod
    def processed_dir_name(
        mode, randomize_camera, snap_to_surface, full_obj, even_downsample
    ) -> str:
        chunk = ""
        if randomize_camera:
            chunk += "_random"
        if snap_to_surface:
            chunk += "_snap"
        if full_obj:
            chunk += "_full"
        if even_downsample:
            chunk += "_even"
        chunk += f"_{mode}"
        return f"taxpose{chunk}"

    def len(self) -> int:
        return len(self.scene_ids)

    def get(self, idx: int) -> GIData:
        scene_id = self.scene_ids[idx]
        data = self.get_data(*scene_id)
        return data

    def get_data(
        self,
        obj_id: str,
        action_id: str,
        goal_id: str,
        loc: Optional[str] = None,
        scale: Optional[float] = None,
        seed: NPSeed = None,
    ) -> GIData:
        """Get a single observation sample.
        Args:
            obj_id: The anchor object ID from Partnet-Mobility.
            action_id: The actor object ID (from Ravens).
            goal_id: The
            loc: the semantic location of the object.
            scale: Optional, actor scale which is random if not specified.
            action_pos: Optional, actor position which is random if not specified.
        Returns:
            ObsActionData and AnchorData, both in the world frame, with a relative transform.
        """

        rng = np.random.default_rng(seed)

        # First, create an environment which will generate our source observations.
        env = PMRenderEnv(obj_id, self.raw_dir, camera_pos=[-3, 0, 1.2])
        object_dict = splits.all_objs[CATEGORIES[obj_id].lower()]

        # Next, check to see if the object needs to be opened in any way.
        partsem = object_dict[f"{obj_id}_{goal_id}"]["partsem"]
        if partsem != "none":
            links_tomove = find_link_index_to_open(
                self.full_sem_dset, partsem, obj_id, object_dict, goal_id
            )

            # Get the joint name.
            joint_name = p.getJointInfo(
                env.obj_id,
                env.link_name_to_index[links_tomove],
                physicsClientId=env.client_id,
            )[1].decode("UTF-8")

            amount = 0.9
            jas: Dict[str, float] = {}
            ranges = env.get_joint_ranges()
            lower, upper = ranges[joint_name]
            angle = amount * (upper - lower) + lower
            jas[joint_name] = angle

            env.set_joint_angles(jas, seed=rng)

        # Select the action object.
        action_obj = ACTION_OBJS[action_id]

        # Load the object at the original floating goal, with a size that is valid there.
        info = object_dict[f"{obj_id}_{goal_id}"]
        floating_goal = np.array([info["x"], info["y"], info["z"]])
        action_body_id, scale = load_action_obj_with_valid_scale(
            action_obj, floating_goal, env, rng
        )

        # Find the actual desired goal position. In the case where we snap to the
        # goal surface, we need to calculate the position in which to reset (base_pos).
        if self.snap_to_surface:
            action_goal_pos_pre = self.snapped_goal_dict[CATEGORIES[obj_id].lower()][
                f"{obj_id}_{goal_id}"
            ]
            action_goal_pos = base_from_bottom(action_body_id, env, action_goal_pos_pre)
        else:
            action_goal_pos = floating_goal

        MAX_ATTEMPTS = 20

        if self.mode == "obs":
            # If we're in the "obs" mode, we need to sample a pose for the action object.
            # We do this by sampling a pose, rendering the scene, and checking if the
            # action object is visible. If it is, we're done. If not, we try again.

            i = 0
            rgbs = []
            while True:
                # Sample a pose.
                action_pos = find_valid_action_initial_pose(
                    action_body_id, env, seed=rng
                )

                # Set the object in the environment.
                p.resetBasePositionAndOrientation(
                    action_body_id,
                    posObj=action_pos,
                    ornObj=[0, 0, 0, 1],
                    physicsClientId=env.client_id,
                )

                # Maybe randomize the camera.
                if self.randomize_camera:
                    env.set_camera("random", seed=rng)

                # Render the scene.
                P_world, pc_seg, rgb, action_mask = render_input_new(
                    action_body_id, env
                )
                rgbs.append(rgb)

                # If we can see the object, we're done.
                if sum(action_mask) > 0:
                    break

                i += 1
                if i > MAX_ATTEMPTS:
                    p.removeBody(action_body_id, physicsClientId=env.client_id)

                    raise ValueError("couldn't find a valid obs pose :(")
        elif self.mode == "goal":
            # If we're in the "goal" mode, the goal is fixed, but the camera angle
            # might be lousy. We need to sample the camera angle until we can see
            # the object.
            action_pos = action_goal_pos

            # Set the object in the environment.
            p.resetBasePositionAndOrientation(
                action_body_id,
                posObj=action_pos,
                ornObj=[0, 0, 0, 1],
                physicsClientId=env.client_id,
            )

            i = 0
            rgbs = []
            while True:
                # Maybe randomize the camera.
                if self.randomize_camera:
                    env.set_camera("random", seed=rng)

                # Render the scene.
                P_world, pc_seg, rgb, action_mask = render_input_new(
                    action_body_id, env
                )
                rgbs.append(rgb)

                # If we can see the object, we're done.
                if sum(action_mask) > 0:
                    break

                i += 1
                if i > MAX_ATTEMPTS or not self.randomize_camera:
                    p.removeBody(action_body_id, physicsClientId=env.client_id)

                    # Make an animation of the rgbs using matplotlib.
                    import matplotlib.animation as animation
                    import matplotlib.pyplot as plt

                    fig = plt.figure()
                    ims = []
                    for rgb in rgbs:
                        im = plt.imshow(rgb, animated=True)
                        ims.append([im])
                    ani = animation.ArtistAnimation(
                        fig, ims, interval=200, blit=True, repeat_delay=1000
                    )
                    plt.show()
                    raise ValueError(
                        "couldn't find a camera pose where the goal was visible"
                    )
        else:
            raise ValueError(f"Invalid mode {self.mode}")

        # Separate out the action and anchor points.
        P_action_world = P_world[action_mask]
        P_anchor_world = P_world[~action_mask]

        # Decide if we want to swap out the action points for the full point cloud.
        if self.full_obj:
            P_action_world = np.load(ACTION_CLOUD_DIR / f"{action_id}.npy")
            P_action_world *= scale
            P_action_world += action_pos

        P_action_world = torch.from_numpy(P_action_world)
        P_anchor_world = torch.from_numpy(P_anchor_world)

        # Now, downsample
        if self.even_downsample:
            action_ixs = downsample_pcd_fps(P_action_world, n=200, seed=rng)
            anchor_ixs = downsample_pcd_fps(P_anchor_world, n=1800, seed=rng)
        else:
            action_ixs = torch.from_numpy(rng.permutation(len(P_action_world))[:200])
            anchor_ixs = torch.from_numpy(rng.permutation(len(P_anchor_world))[:1800])

        # Rebuild the world
        P_action_world = P_action_world[action_ixs]
        P_anchor_world = P_anchor_world[anchor_ixs]
        P_world = np.concatenate([P_action_world, P_anchor_world], axis=0)

        # Regenerate a mask.
        mask_act = torch.ones(len(P_action_world)).int()
        mask_anc = torch.zeros(len(P_anchor_world)).int()
        mask = torch.cat([mask_act, mask_anc])

        # Depending on what mode we're in, create the ground truth displacement data or not.
        if self.mode == "obs":
            # Compute the transform from action object to goal.
            t_action_anchor = action_goal_pos - action_pos
            t_action_anchor = torch.from_numpy(t_action_anchor).float().unsqueeze(0)

            # Compute the ground-truth flow.
            flow = np.tile(t_action_anchor, (P_world.shape[0], 1))
            flow[~mask] = 0
            flow = torch.from_numpy(flow[mask == 1]).float()  # type: ignore
            if len(flow) != len(P_action_world):
                raise ValueError("flow is not the same as the point cloud")
        else:
            t_action_anchor = None
            flow = None

        # Unload the object from the scene.
        p.removeBody(action_body_id, physicsClientId=env.client_id)

        # Assemble the data.
        data = tgd.Data(
            mode=self.mode,
            action_id=action_id,
            goal_id=goal_id,
            action_pos=P_action_world,
            t_action_anchor=t_action_anchor,
            R_action_anchor=None,
            flow=flow,
            obj_id=obj_id,
            anchor_pos=P_anchor_world,
            loc={"in": 0, "top": 1, "left": 2, "right": 3, "under": 4, None: None}[loc],
        )

        # TODO: Decide what this is really doing.........
        if self.mode == "obs":
            data.t_action_anchor = data.t_action_anchor.float()
            data.action_pos = data.action_pos.float()
            data.anchor_pos = data.anchor_pos.float()

            if self.rotate_anchor:
                theta = (rng.random() - 0.5) * np.pi / 4
                R_rand = R.from_rotvec(np.asarray([0, 0, 1]) * theta).as_matrix()
                T_goal_goalnew = torch.eye(4)
                T_goal_goalnew[:3, :3] = torch.from_numpy(R_rand)

                P_goal = data.action_pos + data.t_action_anchor
                P_start = data.action_pos
                P_anc = data.anchor_pos

                # Take points in the start and move to the goal.
                T_start_goal = torch.eye(4)
                T_start_goal[:3, :3] = torch.eye(3)
                T_start_goal[:3, 3] = data.t_action_anchor

                # P_goalnew = T_goal_goalnew * P_goal
                # P_goal = T_start_goal * P_start
                P_goalnew = P_goal @ T_goal_goalnew[:3, :3].T + T_goal_goalnew[:3, 3:].T
                P_ancnew = P_anc @ T_goal_goalnew[:3, :3].T + T_goal_goalnew[:3, 3:].T

                # This labeling is gross.
                T_start_goalnew = T_goal_goalnew @ T_start_goal

                anchor_pos = P_ancnew
                t_action_anchor = T_start_goalnew[:3, 3:].T
                R_action_anchor = T_start_goalnew[:3, :3]
                flow = None  # not implemented
            else:
                anchor_pos = data.anchor_pos
                t_action_anchor = data.t_action_anchor
                R_action_anchor = torch.eye(3)
                flow = data.flow

            data.anchor_pos = anchor_pos
            data.t_action_anchor = t_action_anchor
            data.R_action_anchor = R_action_anchor
            data.flow = flow

        env.close()
        # TODO: rewrite this so that there's only a pos and a mask (anchor, action).
        # OR: return a tuple of data.
        return cast(GIData, data)


class GoalInferenceDataset(tgd.Dataset):
    def __init__(
        self,
        obs_dset: Union[PlaceDataset, CachedByKeyDataset[PlaceDataset]],
        goal_dset: Union[PlaceDataset, CachedByKeyDataset[PlaceDataset]],
        rotate_anchor=False,
        seed: NPSeed = None,
    ):
        self.obs_dset = obs_dset
        self.goal_dset = goal_dset

        self.obs_cached = isinstance(obs_dset, CachedByKeyDataset)
        self.goal_cached = isinstance(goal_dset, CachedByKeyDataset)

        # Get all the different combinations of objects.
        obs_classmap = (
            obs_dset.class_map if not self.obs_cached else obs_dset.dataset.class_map
        )
        goal_classmap = (
            goal_dset.class_map if not self.goal_cached else goal_dset.dataset.class_map
        )

        # Get all pairs.
        self.pairs: List[Tuple[SceneID, SceneID]] = []
        for cat, goal_dict in obs_classmap.items():
            if cat in goal_classmap:
                for goal_id, obs_obj_ids in goal_dict.items():
                    if goal_id in goal_classmap[cat]:
                        goal_obj_ids = goal_classmap[cat][goal_id]

                        self.pairs.extend(itertools.product(obs_obj_ids, goal_obj_ids))
        super().__init__()

        self.rotate_anchor = rotate_anchor
        self.rng = np.random.default_rng(seed)

    def len(self):
        return len(self.pairs)

    def get(self, ix: int) -> Tuple[tgd.Data, tgd.Data, tgd.Data, tgd.Data]:
        # TODO: implement random class balancing...
        obs_scene_id, goal_scene_id = self.pairs[ix]
        assert obs_scene_id[2] == goal_scene_id[2]
        assert CATEGORIES[obs_scene_id[0]] == CATEGORIES[goal_scene_id[0]]

        # Get obs and goal data from each dataset.
        if self.obs_cached:
            dset: CachedByKeyDataset = self.obs_dset
            obs_id_dset = dset.inmem_dsets[obs_scene_id]
            obs_data = obs_id_dset[self.rng.integers(0, len(obs_id_dset))]
        else:
            dset: PlaceDataset = self.obs_dset
            obs_data = dset.get_data(*obs_scene_id, seed=self.rng)

        if self.goal_cached:
            dset: CachedByKeyDataset = self.goal_dset
            goal_id_dset = dset.inmem_dsets[goal_scene_id]
            goal_data = goal_id_dset[self.rng.integers(0, len(goal_id_dset))]
        else:
            dset: PlaceDataset = self.goal_dset
            goal_data = dset.get_data(*goal_scene_id, seed=self.rng)

        # TODO: IMPLEMENT THE FLOW ABOVE...

        # mimic the interface of the original dataset.
        obs_act_pos = obs_data.action_pos
        obs_anc_pos = obs_data.anchor_pos
        goal_act_pos = goal_data.action_pos
        goal_anc_pos = goal_data.anchor_pos

        # Potentially resample... this probably won't ever get used...

        N_ACTION_POINTS = 200
        N_ANCHOR_POINTS = 1800

        if len(obs_act_pos) != N_ACTION_POINTS:
            obs_act_ixs = resample_to_n(len(obs_act_pos), n=N_ACTION_POINTS)
            obs_act_pos = obs_act_pos[obs_act_ixs].float()

        if len(obs_anc_pos) != N_ANCHOR_POINTS:
            obs_anc_ixs = resample_to_n(len(obs_anc_pos), n=N_ANCHOR_POINTS)
            obs_anc_pos = obs_anc_pos[obs_anc_ixs].float()

        if len(goal_act_pos) != N_ACTION_POINTS:
            goal_act_ixs = resample_to_n(len(goal_act_pos), n=N_ACTION_POINTS)
            goal_act_pos = goal_act_pos[goal_act_ixs].float()

        if len(goal_anc_pos) != N_ANCHOR_POINTS:
            goal_anc_ixs = resample_to_n(len(goal_anc_pos), n=N_ANCHOR_POINTS)
            goal_anc_pos = goal_anc_pos[goal_anc_ixs].float()

        t_action_anchor = obs_data.t_action_anchor
        R_action_anchor = obs_data.R_action_anchor

        obs_data_action = tgd.Data(
            id=obs_data.action_id,
            goal_id=obs_data.goal_id,
            pos=obs_act_pos,
            # flow=obs_act_flow,
            t_action_anchor=t_action_anchor,
            R_action_anchor=R_action_anchor,
            loc=goal_data.loc if hasattr(goal_data, "loc") else None,
        )
        obs_data_anchor = tgd.Data(
            id=obs_data.obj_id,
            pos=obs_anc_pos,
            # flow=torch.zeros(len(obs_data.anchor_pos), 3).float(),
        )
        goal_data_action = tgd.Data(
            id=goal_data.action_id,
            pos=goal_act_pos,
        )
        goal_data_anchor = tgd.Data(
            id=goal_data.obj_id,
            pos=goal_anc_pos,
        )

        return obs_data_action, obs_data_anchor, goal_data_action, goal_data_anchor


def resample_to_n(k, n=200):
    ixs = torch.arange(k)
    orig_ixs = torch.arange(k)
    while len(ixs) < n:
        num_needed = n - len(ixs)
        if num_needed > len(ixs):
            ixs = torch.cat([ixs, ixs], dim=-2)
        else:
            resampled = orig_ixs[torch.randperm(len(orig_ixs))[:num_needed]]
            ixs = torch.cat([ixs, resampled], dim=-2)

    return ixs


def scenes_by_location(split, mode, goal_desc):
    scenes = []

    if goal_desc == "top":
        goal_map = splits.TOPS
    elif goal_desc == "left":
        goal_map = splits.LEFTS
    elif goal_desc == "right":
        goal_map = splits.RIGHTS
    elif goal_desc == "in":
        goal_map = splits.INSIDES
    elif goal_desc == "under":
        goal_map = splits.UNDERS
    else:
        raise ValueError("bad goal_desc")

    cat_dict = {
        cat.lower(): split_dict[split]
        for cat, split_dict in splits.split_data["train"].items()
    }

    for cat, obj_ids in cat_dict.items():
        for obj_id in obj_ids:
            if cat in goal_map:
                goal_id = goal_map[cat]
                key = f"{obj_id}_{goal_id}"
                if key in splits.all_objs[cat]:
                    for action_id in list(ACTION_OBJS.keys()):
                        scenes.append((obj_id, action_id, goal_id))

    return filter_bad_scenes(scenes, mode)
