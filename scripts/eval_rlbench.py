import logging
import os
import pickle
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Protocol

import hydra
import joblib
import numpy as np
import open3d as o3d
import pandas as pd
import torch
import tqdm
import wandb
from omegaconf import OmegaConf
from pyrep.backend import sim
from pyrep.const import RenderMode
from pytorch3d.ops import sample_farthest_points
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.utils import name_to_task_class
from rpad.rlbench_utils.placement_dataset import (
    BACKGROUND_NAMES,
    GRIPPER_OBJ_NAMES,
    ROBOT_NONGRIPPER_NAMES,
    TASK_DICT,
    ActionMode,
    AnchorMode,
    filter_out_names,
    get_rgb_point_cloud_by_object_names,
    obs_to_rgb_point_cloud,
)
from scipy.spatial.transform import Rotation as R

from taxpose.datasets.rlbench import (
    colorize_symmetry_labels,
    remove_outliers,
    rotational_symmetry_labels,
)
from taxpose.nets.transformer_flow import ResidualFlow_DiffEmbTransformer
from taxpose.train_pm_placement import theta_err
from taxpose.training.flow_equivariance_training_module_nocentering import (
    EquivarianceTrainingModule,
)
from taxpose.utils.load_model import get_weights_path

TASK_TO_IGNORE_COLLISIONS = {
    "pick_and_lift": {
        "pregrasp": False,
        "grasp": False,
        "lift": True,
        "final": True,
    },
    "pick_up_cup": {
        "pregrasp": False,
        "grasp": False,
        "lift": True,
    },
    "put_knife_on_chopping_board": {
        "pregrasp": False,
        "grasp": False,
        "lift": True,
        "place": True,
    },
    "put_money_in_safe": {
        "pregrasp": False,
        "grasp": False,
        "lift": True,
        "preplace": False,
        "place": False,
    },
    "push_button": {
        "prepush": False,
        "postpush": True,
    },
    "reach_target": {
        "reach": False,
    },
    "slide_block_to_target": {
        "preslide": False,
        "postslide": True,
    },
    "stack_wine": {
        "pregrasp": False,
        "grasp": False,
        "lift": True,
        "preplace": False,
        "place": False,
    },
    "take_money_out_safe": {
        "pregrasp": False,
        "grasp": True,
        "lift": True,
        "place": False,
    },
    "take_umbrella_out_of_umbrella_stand": {
        "pregrasp": False,
        "grasp": False,
        "lift": True,
    },
}


class TaskVideoRecorder:
    def __init__(self, scene, cam_name: str):
        self.frames = []
        self.scene = scene
        self.cam_name = cam_name

    def step(self):
        obs = self.scene.get_observation()

        # the cam_name is an attribute of the observation, not a dict key.
        if not hasattr(obs, self.cam_name):
            raise ValueError(f"Camera {self.cam_name} not found in observation.")

        self.frames.append(getattr(obs, self.cam_name))

    def write_video(self, video_path: str):
        if len(self.frames) == 0:
            return

        import cv2

        # Write the video.
        height, width, _ = self.frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))

        for frame in self.frames:
            out.write(frame)

        out.release()


# # HEHEH extremely hacky way to add a method to a class.
# def RecordedTask(task: Type[Task], video_path: str) -> Type[Task]:

#     def step(self):
#         super(task, self).step()
#         breakpoint()

#     _RecordedTask = type(
#         task.__name__,
#         (task,),
#         {
#             "video_path": video_path,
#             "frames": [],
#             "step": step,
#         },
#     )

#     return _RecordedTask


class RelativePosePredictor(Protocol):
    @abstractmethod
    def predict(self, obs, phase: str) -> np.ndarray:
        pass


class PickPlacePredictor(Protocol):
    @abstractmethod
    def grasp(self, obs) -> np.ndarray:
        pass

    @abstractmethod
    def place(self, init_obs, post_grasp_obs) -> np.ndarray:
        pass


def create_coordinate_frame(T, size=0.1):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame.transform(T)
    return frame


def bottle_symmetry_features(points):
    # TODO - hack: right now assume the bottle is vertical.
    assert len(points.shape) == 2

    action_symmetry_axis = np.array([0, 0, 1], dtype=np.float32)
    action_second_axis = np.array([1, 0, 0], dtype=np.float32)

    action_symmetry_features = rotational_symmetry_labels(
        points, action_symmetry_axis, action_second_axis
    )[..., None]

    return action_symmetry_features


def rack_symmetry_features(points):
    return np.ones((points.shape[0], 1), dtype=np.float32)


def viz_symmetry_features(
    action_pc,
    anchor_pc,
    action_symmetry_features,
    anchor_symmetry_features,
):
    action_symmetry_rgb = colorize_symmetry_labels(action_symmetry_features[..., 0])
    anchor_symmetry_rgb = colorize_symmetry_labels(anchor_symmetry_features[..., 0])

    # Convert to 0-1 range.
    action_symmetry_rgb = action_symmetry_rgb / 255.0
    anchor_symmetry_rgb = anchor_symmetry_rgb / 255.0

    action_symmetry_pc = o3d.geometry.PointCloud()
    action_symmetry_pc.points = o3d.utility.Vector3dVector(action_pc.cpu().numpy())
    action_symmetry_pc.colors = o3d.utility.Vector3dVector(action_symmetry_rgb)

    anchor_symmetry_pc = o3d.geometry.PointCloud()
    anchor_symmetry_pc.points = o3d.utility.Vector3dVector(anchor_pc.cpu().numpy())
    anchor_symmetry_pc.colors = o3d.utility.Vector3dVector(anchor_symmetry_rgb)

    o3d.visualization.draw_geometries(
        [
            action_symmetry_pc,
            anchor_symmetry_pc,
        ],
        window_name="Symmetry Features",
        width=1024,
        height=768,
    )


class TAXPoseRelativePosePredictor(RelativePosePredictor):
    def __init__(
        self,
        policy_spec,
        task_cfg,
        wandb_cfg,
        checkpoints_cfg,
        run=None,
        debug_viz=False,
    ):
        self.models = {}
        for phase in TASK_DICT[task_cfg.name]["phase_order"]:
            if hasattr(checkpoints_cfg, "single_model_override"):
                model_path = checkpoints_cfg.single_model_override
            else:
                model_path = checkpoints_cfg[phase].ckpt_file
            self.models[phase] = self.load_model(
                model_path,
                policy_spec.model,
                wandb_cfg,
                task_cfg.phases[phase],
                run=run,
            )

        self.model_cfg = policy_spec.model
        self.task_name = task_cfg.name
        self.debug_viz = debug_viz
        self.action_mode = task_cfg.action_mode
        self.anchor_mode = task_cfg.anchor_mode

    @staticmethod
    def render(obs, inputs, preds, T_action_world, T_actionfinal_world):
        # Draw a single point cloud.
        action_pc_o3d = o3d.geometry.PointCloud()
        action_pc_o3d.points = o3d.utility.Vector3dVector(
            inputs["action_pc"].cpu().numpy()
        )
        action_pc_o3d.colors = o3d.utility.Vector3dVector(
            np.array([1, 0, 0]) * np.ones((inputs["action_pc"].shape[0], 3))
        )

        anchor_pc_o3d = o3d.geometry.PointCloud()
        anchor_pc_o3d.points = o3d.utility.Vector3dVector(
            inputs["anchor_pc"].cpu().numpy()
        )
        anchor_pc_o3d.colors = o3d.utility.Vector3dVector(
            np.array([0, 1, 0]) * np.ones((inputs["anchor_pc"].shape[0], 3))
        )

        pred_pc_o3d = o3d.geometry.PointCloud()
        pred_pc_o3d.points = o3d.utility.Vector3dVector(
            preds["pred_points_action"][0].cpu().numpy()
        )
        pred_pc_o3d.colors = o3d.utility.Vector3dVector(
            np.array([0, 0, 1]) * np.ones((preds["pred_points_action"].shape[0], 3))
        )

        # Draw the coordinate frames.
        world_frame = create_coordinate_frame(np.eye(4))
        action_frame = create_coordinate_frame(T_action_world)
        actionfinal_frame = create_coordinate_frame(T_actionfinal_world)

        # Draw the geometries
        geometries = [
            action_pc_o3d,
            anchor_pc_o3d,
            pred_pc_o3d,
            world_frame,
            action_frame,
            actionfinal_frame,
        ]
        o3d.visualization.draw_geometries(geometries)

    @staticmethod
    def load_model(model_path, model_cfg, wandb_cfg, task_cfg, run=None):
        ckpt_file = get_weights_path(model_path, wandb_cfg, run=run)
        network = ResidualFlow_DiffEmbTransformer(
            pred_weight=model_cfg.pred_weight,
            emb_nn=model_cfg.emb_nn,
            emb_dims=model_cfg.emb_dims,
            return_flow_component=model_cfg.return_flow_component,
            center_feature=model_cfg.center_feature,
            # inital_sampling_ratio=model_cfg.inital_sampling_ratio,
            residual_on=model_cfg.residual_on,
            multilaterate=model_cfg.multilaterate,
            sample=model_cfg.mlat_sample,
            mlat_nkps=model_cfg.mlat_nkps,
            break_symmetry=model_cfg.break_symmetry,
            conditional=model_cfg.conditional if "conditional" in model_cfg else False,
        )
        model = EquivarianceTrainingModule(
            network,
            weight_normalize=task_cfg.weight_normalize,
            softmax_temperature=task_cfg.softmax_temperature,
            sigmoid_on=True,
            flow_supervision="both",
        )
        weights = torch.load(ckpt_file)["state_dict"]
        model.load_state_dict(weights)

        model.eval()
        model = model.cuda()
        return model

    def predict(self, obs, phase: str, handlemap) -> np.ndarray:
        inputs = obs_to_input(
            obs,
            self.task_name,
            phase,
            self.action_mode,
            self.anchor_mode,
            handlemap,
        )
        model = self.models[phase]
        device = model.device

        action_pc = inputs["action_pc"].unsqueeze(0).to(device)
        anchor_pc = inputs["anchor_pc"].unsqueeze(0).to(device)

        K = self.model_cfg.num_points
        action_pc, _ = sample_farthest_points(action_pc, K=K, random_start_point=True)
        anchor_pc, _ = sample_farthest_points(anchor_pc, K=K, random_start_point=True)

        if self.model_cfg.break_symmetry:
            raise NotImplementedError()
            action_symmetry_features = bottle_symmetry_features(
                action_pc.cpu().numpy()[0]
            )
            anchor_symmetry_features = rack_symmetry_features(
                anchor_pc.cpu().numpy()[0]
            )

            if self.debug_viz:
                viz_symmetry_features(
                    action_pc[0],
                    anchor_pc[0],
                    action_symmetry_features,
                    anchor_symmetry_features,
                )

            # Torchify and GPU
            action_symmetry_features = torch.from_numpy(action_symmetry_features).to(
                device
            )
            anchor_symmetry_features = torch.from_numpy(anchor_symmetry_features).to(
                device
            )
            action_symmetry_features = action_symmetry_features[None]
            anchor_symmetry_features = anchor_symmetry_features[None]
        else:
            action_symmetry_features = None
            anchor_symmetry_features = None

        if "conditional" in self.model_cfg and self.model_cfg.conditional:
            phase_ix = TASK_DICT[self.task_name]["phase_order"].index(phase)
            phase_onehot = np.zeros(len(TASK_DICT[self.task_name]["phase_order"]))
            phase_onehot[phase_ix] = 1
            phase_onehot = torch.from_numpy(phase_onehot).to(device)[None]
        else:
            phase_onehot = None

        preds = model(
            action_pc,
            anchor_pc,
            action_symmetry_features,
            anchor_symmetry_features,
            phase_onehot,
        )

        # Get the current pose of the gripper.
        T_gripper_world = np.eye(4)
        T_gripper_world[:3, 3] = obs.gripper_pose[:3]
        T_gripper_world[:3, :3] = R.from_quat(obs.gripper_pose[3:]).as_matrix()
        T_gripperfinal_gripper = preds["pred_T_action"].get_matrix()[0].T.cpu().numpy()
        T_gripperfinal_world = T_gripperfinal_gripper @ T_gripper_world

        if self.debug_viz:
            self.render(obs, inputs, preds, T_gripper_world, T_gripperfinal_world)

        return T_gripperfinal_world


class RandomPickPlacePredictor(RelativePosePredictor):
    def __init__(self, cfg, xrange, yrange, zrange, grip_rot=None):
        self.cfg = cfg
        self.xrange = xrange
        self.yrange = yrange
        self.zrange = zrange
        self.grip_rot = grip_rot

    @staticmethod
    def _get_random_pose(xrange=None, yrange=None, zrange=None, grip_rot=None):
        if xrange is None:
            xrange = (-0.3, 0.3)
        if yrange is None:
            yrange = (-0.3, 0.3)
        if zrange is None:
            zrange = (0.3, 0.5)

        t_rand = np.array(
            [
                np.random.uniform(xrange[0], xrange[1]),
                np.random.uniform(yrange[0], yrange[1]),
                np.random.uniform(zrange[0], zrange[1]),
            ]
        )

        if grip_rot is None:
            R_rand = R.random().as_matrix()
        else:
            R_rand = R.from_quat(grip_rot).as_matrix()

        T_rand = np.eye(4)
        T_rand[:3, :3] = R_rand
        T_rand[:3, 3] = t_rand
        return T_rand

    def predict(self, obs, phase) -> np.ndarray:
        return self._get_random_pose(
            self.xrange, self.yrange, self.zrange, self.grip_rot
        )


def get_handle_mapping():
    handles = {
        sim.simGetObjectName(handle): handle
        for handle in sim.simGetObjects(sim.sim_handle_all)
    }
    return handles


# TODO: put this in the original rlbench library.
def obs_to_input(
    obs,
    task_name,
    phase,
    action_mode: ActionMode,
    anchor_mode: AnchorMode,
    handlemap,
):
    rgb, point_cloud, mask = obs_to_rgb_point_cloud(obs)

    action_names = TASK_DICT[task_name]["phase"][phase]["action_obj_names"]
    if action_mode == ActionMode.GRIPPER_AND_OBJECT:
        action_names += GRIPPER_OBJ_NAMES
    elif action_mode == ActionMode.OBJECT:
        pass
    else:
        raise ValueError(f"Invalid action mode: {action_mode}")

    action_rgb, action_point_cloud = get_rgb_point_cloud_by_object_names(
        rgb, point_cloud, mask, action_names
    )

    if anchor_mode == AnchorMode.RAW:
        raise NotImplementedError()
    elif anchor_mode == AnchorMode.BACKGROUND_REMOVED:
        raise NotImplementedError()
    elif anchor_mode == AnchorMode.BACKGROUND_ROBOT_REMOVED:
        anchor_rgb, anchor_point_cloud = filter_out_names(
            rgb,
            point_cloud,
            mask,
            handlemap,
            BACKGROUND_NAMES + ROBOT_NONGRIPPER_NAMES,
        )
    elif anchor_mode == AnchorMode.SINGLE_OBJECT:
        # Get the rgb and point cloud for the anchor objects.
        anchor_rgb, anchor_point_cloud = get_rgb_point_cloud_by_object_names(
            rgb,
            point_cloud,
            mask,
            TASK_DICT[task_name]["phase"][phase]["anchor_obj_names"],
        )
    else:
        raise ValueError(f"Invalid anchor mode: {anchor_mode}")

    # Remove outliers in the same way...
    action_point_cloud = remove_outliers(action_point_cloud[None])[0]

    if (
        anchor_mode != AnchorMode.RAW
        and anchor_mode != AnchorMode.BACKGROUND_ROBOT_REMOVED
    ):
        anchor_point_cloud = remove_outliers(anchor_point_cloud[None])[0]

    # Visualize the point clouds.
    # act_pc = o3d.geometry.PointCloud()
    # act_pc.points = o3d.utility.Vector3dVector(action_point_cloud)
    # act_pc.colors = o3d.utility.Vector3dVector(
    #     np.array([1, 0, 0]) * np.ones((action_point_cloud.shape[0], 3))
    # )

    # anc_pc = o3d.geometry.PointCloud()
    # anc_pc.points = o3d.utility.Vector3dVector(anchor_point_cloud)
    # anc_pc.colors = o3d.utility.Vector3dVector(
    #     np.array([0, 1, 0]) * np.ones((anchor_point_cloud.shape[0], 3))
    # )

    # o3d.visualization.draw_geometries([act_pc, anc_pc])

    return {
        "action_rgb": torch.from_numpy(action_rgb),
        "action_pc": torch.from_numpy(action_point_cloud).float(),
        "anchor_rgb": torch.from_numpy(anchor_rgb),
        "anchor_pc": torch.from_numpy(anchor_point_cloud).float(),
    }


# We may need to add more failure reasons.
class FailureReason(str, Enum):
    NOT_ATTEMPTED = "not_attempted"
    NO_FAILURE = "no_failure"
    PREDICTION_OUTSIDE_WORKSPACE = "prediction_outside_workspace"
    MOTION_PLANNING_FAILURE = "motion_planning_failure"
    PREDICTION_FAILURE = "prediction_failure"
    CAUSED_TERMINATION = "caused_termination"
    UNKNOWN_FAILURE = "unknown_failure"
    FINAL_PHASE_NO_SUCCESS = "final_phase_no_success"


@dataclass
class PhaseResult:
    phase: str
    failure_reason: FailureReason


@dataclass
class TrialResult:
    success: bool

    # Failure is either a global failure, or a map from phase to failure reason.
    failure_reason: List[PhaseResult]


def get_obs_config():
    """We have to do this to match the distribution of the training data!"""

    img_size = (256, 256)
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    obs_config.right_shoulder_camera.image_size = img_size
    obs_config.left_shoulder_camera.image_size = img_size
    obs_config.overhead_camera.image_size = img_size
    obs_config.wrist_camera.image_size = img_size
    obs_config.front_camera.image_size = img_size

    # Store depth as 0 - 1
    obs_config.right_shoulder_camera.depth_in_meters = False
    obs_config.left_shoulder_camera.depth_in_meters = False
    obs_config.overhead_camera.depth_in_meters = False
    obs_config.wrist_camera.depth_in_meters = False
    obs_config.front_camera.depth_in_meters = False

    # We DON'T want to save masks as rgb.
    obs_config.left_shoulder_camera.masks_as_one_channel = True
    obs_config.right_shoulder_camera.masks_as_one_channel = True
    obs_config.overhead_camera.masks_as_one_channel = True
    obs_config.wrist_camera.masks_as_one_channel = True
    obs_config.front_camera.masks_as_one_channel = True

    obs_config.right_shoulder_camera.render_mode = RenderMode.OPENGL3
    obs_config.left_shoulder_camera.render_mode = RenderMode.OPENGL3
    obs_config.overhead_camera.render_mode = RenderMode.OPENGL3
    obs_config.wrist_camera.render_mode = RenderMode.OPENGL3
    obs_config.front_camera.render_mode = RenderMode.OPENGL3

    return obs_config


def move(action: np.ndarray, task, env, max_tries=10, ignore_collisions=False):
    try_count = 0

    p_desired = action[:3]
    q_desired = action[3:7]

    env._action_mode.arm_action_mode._collision_checking = not ignore_collisions

    while try_count < max_tries:
        obs, reward, terminate = task.step(action)

        # Check to make sure that the achieved pose is close to the desired pose.
        p_achieved = obs.gripper_pose[:3]
        q_achieved = obs.gripper_pose[3:7]

        p_diff = np.linalg.norm(p_desired - p_achieved)
        q_diff = theta_err(
            torch.from_numpy(R.from_quat(q_desired).as_matrix()),
            torch.from_numpy(R.from_quat(q_achieved).as_matrix()),
        )

        criteria = (p_diff < 5e-3, q_diff < 1)

        if all(criteria) or reward == 1:
            break

        logging.warning(
            f"Too far away (pos: {p_diff:.3f}, rot: {q_diff:.3f}... Retrying..."
        )
        try_count += 1

    return obs, reward, terminate


@torch.no_grad()
def run_trial(
    policy_spec,
    task_spec,
    env_spec,
    checkpoints,
    wandb_cfg,
    headless=True,
    run=None,
    trial_num=0,
) -> TrialResult:
    # Seed the random number generator.

    # TODO: beisner need to seeeeeed.

    # Create the policy.
    # policy = TAXPosePickPlacePredictor(
    #     policy_spec,
    #     task_spec,
    #     task_spec.wandb,
    #     task_spec.checkpoints,
    #     run=run,
    #     debug_viz=not headless,
    # )

    policy: RelativePosePredictor = TAXPoseRelativePosePredictor(
        policy_spec,
        task_spec,
        wandb_cfg,
        checkpoints,
        run=run,
        debug_viz=not headless,
    )
    # policy: RelativePosePredictor = RandomPickPlacePredictor(
    #     policy_spec, (-0.3, 0.3), (-0.3, 0.3), (0.3, 0.5)
    # )

    # Create the environment.
    collision_checking = (
        policy_spec.collision_checking if "collision_checking" in policy_spec else True
    )
    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(
                collision_checking=collision_checking
            ),
            gripper_action_mode=Discrete(),
        ),
        obs_config=get_obs_config(),
        headless=headless,
    )

    phase_order = TASK_DICT[task_spec.name]["phase_order"]

    # Keep track of the results for each phase.
    phase_results = {phase: FailureReason.NOT_ATTEMPTED for phase in phase_order}

    # Helper function to order the phases sequentially in the results.
    def pr():
        return [PhaseResult(phase, phase_results[phase]) for phase in phase_order]

    try:
        task_cls = name_to_task_class(task_spec.name)
        # task_cls = RecordedTask(task_cls, video_path="video.mp4")
        task = env.get_task(task_cls)
        task.sample_variation()

        recorder = TaskVideoRecorder(task._scene, "front_rgb")

        task._scene.register_step_callback(recorder.step)

        handlemap = get_handle_mapping()

        # Reset the environment. For now, ignore descriptions.
        _, obs = task.reset()
    except Exception as e:
        env.shutdown()
        print(f"Failed to reset the environment: {e}")
        return TrialResult(False, pr())

    try:

        # Loop through the phases, and predict.
        for phase in phase_order:
            # Try to make a predictions.
            try:
                T_gripper_world = policy.predict(obs, phase, handlemap)
            except Exception as e:
                print(e)
                phase_results[phase] = FailureReason.PREDICTION_FAILURE
                return TrialResult(False, pr())

            # Compute the transform.
            p_gripper_world = T_gripper_world[:3, 3]
            q_gripper_world = R.from_matrix(T_gripper_world[:3, :3]).as_quat()
            if TASK_DICT[task_spec.name]["phase"][phase]["gripper_open"]:
                gripper_state = np.array([1.0])
            else:
                gripper_state = np.array([0.0])
            action = np.concatenate([p_gripper_world, q_gripper_world, gripper_state])

            # if phase == "place":
            #     # Add the z offset.
            #     action[2] += policy_spec.z_offset

            # Attempt the action.
            try:
                obs, reward, terminate = move(
                    action,
                    task,
                    env,
                    max_tries=10,
                    ignore_collisions=TASK_TO_IGNORE_COLLISIONS[task_spec.name][phase],
                )  # Eventually add collision checking.

            except Exception as e:
                if "workspace" in str(e):
                    phase_results[phase] = FailureReason.PREDICTION_OUTSIDE_WORKSPACE
                else:
                    phase_results[phase] = FailureReason.MOTION_PLANNING_FAILURE
                return TrialResult(False, pr())

            phase_results[phase] = FailureReason.NO_FAILURE

            if reward == 1:
                return TrialResult(True, pr())

            if terminate:
                phase_results[phase] = FailureReason.CAUSED_TERMINATION
                return TrialResult(False, pr())

            # Check if it's the final phase, and if we haven't succeeded...
            if phase == phase_order[-1]:
                phase_results[phase] = FailureReason.FINAL_PHASE_NO_SUCCESS
                return TrialResult(False, pr())

        print("unknown failure")
        return TrialResult(False, pr())
    except Exception as e:
        print(e)
        return TrialResult(False, pr())

    finally:
        # Close the environment.
        env.shutdown()

        if not os.path.exists("videos"):
            os.makedirs("videos")

        fn = f"videos/{trial_num}.mp4"
        recorder.write_video(fn)


def run_trials(
    policy_spec,
    task_spec,
    env_spec,
    checkpoints,
    wandb_cfg,
    num_trials,
    headless=True,
    run=None,
    num_workers=10,
):
    # TODO: Parallelize this. Should all be picklable.

    if num_workers <= 1:
        results = []
        for i in range(num_trials):
            result = run_trial(
                policy_spec,
                task_spec,
                env_spec,
                checkpoints,
                wandb_cfg,
                headless=headless,
                run=run,
                trial_num=i,
            )
            logging.info(f"Trial {i}: {result}")
            results.append(result)
    else:
        # Try with joblib. Let's see if this works.
        job_results = joblib.Parallel(n_jobs=num_workers, return_as="generator")(
            joblib.delayed(run_trial)(
                policy_spec,
                task_spec,
                env_spec,
                checkpoints,
                wandb_cfg,
                headless=headless,
                run=None,  # run is unpickleable...
                trial_num=i,
            )
            for i in range(num_trials)
        )
        results = [r for r in tqdm.tqdm(job_results, total=num_trials)]

    return results


@hydra.main(version_base="1.2", config_path="../configs", config_name="eval_rlbench")
def main(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    # Force the hydra config cfg to be resolved
    cfg = OmegaConf.create(OmegaConf.to_yaml(cfg, resolve=True))

    # Initialize wandb.
    run = wandb.init(
        job_type=cfg.job_type,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.wandb.group,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )

    results = run_trials(
        cfg.policy_spec,
        cfg.task,
        None,
        cfg.checkpoints,
        cfg.wandb,
        cfg.num_trials,
        headless=cfg.headless,
        run=run,
        num_workers=cfg.resources.num_workers,
    )

    # Compute some metrics.
    num_successes = sum([1 for r in results if r.success])
    num_failures = sum([1 for r in results if not r.success])
    print(f"Successes: {num_successes}")
    print(f"Failures: {num_failures}")

    # Count failure reasons
    # Aggregate per-phase failure reasons.
    failure_reason_counts = {
        phase: {reason.name: 0 for reason in FailureReason}
        for phase in TASK_DICT[cfg.task.name]["phase_order"]
    }
    for result in results:
        for pr in result.failure_reason:
            pr: PhaseResult
            failure_reason_counts[pr.phase][pr.failure_reason.name] += 1

    # Flatten the structure so that the keys are phase_reason.
    failure_reason_counts = {
        f"{phase}_{reason}": count
        for phase, reasons in failure_reason_counts.items()
        for reason, count in reasons.items()
    }

    results_dir = Path(cfg.output_dir)

    # Create a pandas dataframe for the results statistics.
    # The df should have a column for successes, failures, success rate, and a column with
    # counts for each failure reason.
    df = pd.DataFrame(
        {
            "successes": [num_successes],
            "failures": [num_failures],
            "success_rate": [num_successes / cfg.num_trials],
            **failure_reason_counts,
        }
    )

    for phase in TASK_DICT[cfg.task.name]["phase_order"]:
        # Get all columns that start with the phase name.
        print(f"Phase: {phase}")
        phase_cols = [col for col in df.columns if col.startswith(phase)]
        phase_df = df[phase_cols]
        phase_df = phase_df.rename(columns=lambda x: x[len(phase) + 1 :])
        print(phase_df)

    # Save the results to wandb as a table.
    run.log(
        {
            "results_table": wandb.Table(dataframe=df),
        }
    )

    # Pickle the results and stats.
    # breakpoint()
    # Somehow, pickle isn't working with nested enum stuff.

    pkl_results = [
        {
            "success": r.success,
            "failure_reason": [
                {"phase": pr.phase, "failure_reason": pr.failure_reason.name}
                for pr in r.failure_reason
            ],
        }
        for r in results
    ]

    with open(results_dir / "results.pkl", "wb") as f:
        pickle.dump(pkl_results, f)

    with open(results_dir / "stats.pkl", "wb") as f:
        pickle.dump(
            {
                "num_successes": num_successes,
                "num_failures": num_failures,
                "success_rate": num_successes / cfg.num_trials,
                "failure_reason_counts": failure_reason_counts,
            },
            f,
        )


if __name__ == "__main__":
    main()
