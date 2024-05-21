import logging
import os
import pickle
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Protocol, Tuple

import hydra
import joblib
import numpy as np
import open3d as o3d
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import rpad.visualize_3d.plots as rvp
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
    TASK_DICT,
    ActionMode,
    AnchorMode,
    get_action_points,
    get_anchor_points,
    obs_to_rgb_point_cloud,
)
from scipy.spatial.transform import Rotation as R

import taxpose.utils.website as tuw
from taxpose.datasets.rlbench import (
    colorize_symmetry_labels,
    remove_outliers,
    rotational_symmetry_labels,
)
from taxpose.nets.transformer_flow import create_network
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
        # "place": True,
        "place": False,
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
        "pregasp": False,
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
        # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(f"{video_path}.mp4", fourcc, 20.0, (width, height))

        for frame in self.frames:
            out.write(frame)

        out.release()

        # ffmpeg -i "$file" -c:v libx264 -c:a aac "${file%.mp4}.mov"

        # Redirect the output to /dev/null to suppress the output.
        os.system(
            f"/usr/bin/ffmpeg -i {video_path}.mp4 -c:v libx264 -c:a aac {video_path}.mov > /dev/null 2>&1"
        )


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
    def predict(self, obs, phase: str) -> Tuple[np.ndarray, Dict[str, Any]]:
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
            if hasattr(checkpoints_cfg, "ckpt_file"):
                model_path = checkpoints_cfg.ckpt_file
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
        self.policy_spec = policy_spec

        self.phase_attempt_counts = {
            phase: 0 for phase in TASK_DICT[task_cfg.name]["phase_order"]
        }

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

        network = create_network(model_cfg)
        model = EquivarianceTrainingModule(
            network,
            weight_normalize=task_cfg.weight_normalize,
            softmax_temperature=task_cfg.softmax_temperature,
            sigmoid_on=True,
            flow_supervision="both",
        )
        if model_path is not None:
            ckpt_file = get_weights_path(model_path, wandb_cfg, run=run)
            weights = torch.load(ckpt_file)["state_dict"]
            model.load_state_dict(weights)

        model.eval()
        model = model.cuda()
        return model

    def predict(self, obs, phase: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        inputs = obs_to_input(
            obs,
            self.task_name,
            phase,
            self.action_mode,
            self.anchor_mode,
        )

        model = self.models[phase]
        device = model.device

        action_pc = inputs["action_pc"].unsqueeze(0).to(device)
        anchor_pc = inputs["anchor_pc"].unsqueeze(0).to(device)
        action_rgb = inputs["action_rgb"].unsqueeze(0).to(device)
        anchor_rgb = inputs["anchor_rgb"].unsqueeze(0).to(device)

        K = self.policy_spec.num_points
        action_pc, action_idx = sample_farthest_points(
            action_pc, K=K, random_start_point=True
        )
        anchor_pc, anchor_idx = sample_farthest_points(
            anchor_pc, K=K, random_start_point=True
        )
        action_rgb = action_rgb[0][action_idx[0]].unsqueeze(0) / 255.0
        anchor_rgb = anchor_rgb[0][anchor_idx[0]].unsqueeze(0) / 255.0

        if self.policy_spec.include_rgb_features:
            action_features = action_rgb
            anchor_features = anchor_rgb
        else:
            action_features = None
            anchor_features = None

        if self.policy_spec.break_symmetry:
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
            action_features,
            anchor_features,
            phase_onehot,
        )

        # Get the current pose of the gripper.
        T_gripper_world = np.eye(4)
        T_gripper_world[:3, 3] = obs.gripper_pose[:3]
        T_gripper_world[:3, :3] = R.from_quat(obs.gripper_pose[3:]).as_matrix()
        T_gripperfinal_gripper = preds["pred_T_action"].get_matrix()[0].T.cpu().numpy()
        T_gripperfinal_world = T_gripperfinal_gripper @ T_gripper_world

        pred_points = preds["pred_points_action"][0].cpu().numpy()

        # If we've already attempted this phase, then we should add some random jitter.
        if self.phase_attempt_counts[phase] > 0 and self.policy_spec.add_random_jitter:
            # Add some random motion in the positive z
            # T_gripperfinal_world[2, 3] += np.random.uniform(0, 0.03)

            T_random_jitter = np.eye(4)

            # Add some random uniform noise to the position.
            T_random_jitter[:3, 3] = np.random.normal(0, 0.01, 3)

            # Add some random noise to the rotation.
            # R_random_jitter = R.from_euler(
            #     "xyz", np.random.normal(0, 0.07, 3)
            # ).as_matrix()
            # T_random_jitter[:3, :3] = R_random_jitter

            T_gripperfinal_world = T_random_jitter @ T_gripperfinal_world

            # Also apply the same jitter to the predicted points.
            pred_points = (T_random_jitter[:3, :3] @ pred_points.T).T + T_random_jitter[
                :3, 3
            ]

        if self.debug_viz:
            self.render(obs, inputs, preds, T_gripper_world, T_gripperfinal_world)

        fig = rvp.segmentation_fig(
            np.concatenate(
                [
                    inputs["action_pc"].cpu().numpy(),
                    inputs["anchor_pc"].cpu().numpy(),
                    pred_points,
                ],
                axis=0,
            ),
            np.concatenate(
                [
                    np.zeros((inputs["action_pc"].shape[0],)),
                    np.ones((inputs["anchor_pc"].shape[0],)),
                    2 * np.ones((preds["pred_points_action"][0].shape[0],)),
                ]
            ).astype(np.int32),
            labelmap={
                0: "action",
                1: "anchor",
                2: "pred",
            },
        )

        self.phase_attempt_counts[phase] += 1

        return T_gripperfinal_world, {"plot": fig}


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

    def predict(self, obs, phase) -> Tuple[np.ndarray, Dict[str, Any]]:
        return (
            self._get_random_pose(self.xrange, self.yrange, self.zrange, self.grip_rot),
            {},
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
):
    rgb, point_cloud, mask = obs_to_rgb_point_cloud(obs)

    ##############################
    # Action points.
    ##############################

    action_rgb, action_point_cloud = get_action_points(
        action_mode,
        rgb,
        point_cloud,
        mask,
        task_name,
        phase,
        use_from_simulator=True,
    )

    # Remove outliers in the same way...
    action_point_cloud = remove_outliers(action_point_cloud[None])[0]

    ##############################
    # Anchor points.
    ##############################

    anchor_rgb, anchor_point_cloud = get_anchor_points(
        anchor_mode,
        rgb,
        point_cloud,
        mask,
        task_name,
        phase,
        use_from_simulator=True,
    )

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

    policy: RelativePosePredictor = TAXPoseRelativePosePredictor(
        policy_spec,
        task_spec,
        wandb_cfg,
        checkpoints,
        run=run,
        debug_viz=not headless,
    )

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
        task.set_variation(0)

        recorder = TaskVideoRecorder(task._scene, "front_rgb")

        task._scene.register_step_callback(recorder.step)

        # Reset the environment. For now, ignore descriptions.
        _, obs = task.reset()
    except Exception as e:
        env.shutdown()
        print(f"Failed to reset the environment: {e}")
        success = False
        return TrialResult(success, pr())

    try:

        # Loop through the phases, and predict.
        phase_plots: List[go.Figure] = []
        for phase in phase_order:

            N_MOTION_PLANNING_SAMPLING_TRIES = 20
            motion_succeeded = False  # whether any of the predicted motions succeeded.

            # Try to make a prediction a number of times. Hopefully randomized.
            for i in range(N_MOTION_PLANNING_SAMPLING_TRIES):

                # Try to make a predictions.
                try:
                    T_gripper_world, extras = policy.predict(obs, phase)
                    if "plot" in extras:
                        phase_plots.append((f"{phase}-{i}", extras["plot"]))
                except Exception as ex:

                    phase_results[phase] = FailureReason.PREDICTION_FAILURE
                    success = False
                    return TrialResult(success, pr())

                # Compute the transform.
                p_gripper_world = T_gripper_world[:3, 3]
                q_gripper_world = R.from_matrix(T_gripper_world[:3, :3]).as_quat()
                if TASK_DICT[task_spec.name]["phase"][phase]["gripper_open"]:
                    gripper_state = np.array([1.0])
                else:
                    gripper_state = np.array([0.0])
                action = np.concatenate(
                    [p_gripper_world, q_gripper_world, gripper_state]
                )

                # Attempt the action.
                try:
                    obs, reward, terminate = move(
                        action,
                        task,
                        env,
                        max_tries=10,
                        ignore_collisions=TASK_TO_IGNORE_COLLISIONS[task_spec.name][
                            phase
                        ],
                    )  # Eventually add collision checking.
                    motion_succeeded = True

                except Exception as ex:
                    if "workspace" in str(ex):
                        phase_results[
                            phase
                        ] = FailureReason.PREDICTION_OUTSIDE_WORKSPACE
                    elif "A path could not be found." in str(ex):
                        phase_results[phase] = FailureReason.MOTION_PLANNING_FAILURE
                    else:
                        logging.error(f"Unknown error: {ex}")
                        phase_results[phase] = FailureReason.UNKNOWN_FAILURE

                if motion_succeeded:
                    break
                else:
                    logging.warning(
                        f"Failed to execute action {i} for reason {phase_results[phase]}. Retrying..."
                    )

            # If we didn't succeed in any of the motions, then we failed.
            if not motion_succeeded:
                success = False
                return TrialResult(success, pr())
            elif i > 0:
                logging.warning(f"Success on attempt {i}")

            phase_results[phase] = FailureReason.NO_FAILURE

            if reward == 1:
                success = True
                return TrialResult(success, pr())

            if terminate:
                phase_results[phase] = FailureReason.CAUSED_TERMINATION
                success = False
                return TrialResult(success, pr())

            # Check if it's the final phase, and if we haven't succeeded...
            if phase == phase_order[-1]:
                phase_results[phase] = FailureReason.FINAL_PHASE_NO_SUCCESS
                success = False
                return TrialResult(success, pr())

        print("unknown failure")
        return TrialResult(False, pr())
    except Exception as e:
        print(e)
        success = False
        return TrialResult(success, pr())

    finally:
        # Close the environment.
        env.shutdown()

        # Make a directory for the entire episode.
        if not os.path.exists("episodes"):
            os.makedirs("episodes")

        # Make a directory for the trial.
        if not os.path.exists(f"episodes/{trial_num}"):
            os.makedirs(f"episodes/{trial_num}")
        else:
            raise ValueError(f"Directory episodes/{trial_num} already exists.")

        # website = PlotlyWebsiteBuilder(f"episode_{trial_num}")
        website_plots = []
        website_video = None

        # Save phase plots.
        for phase, fig in phase_plots:
            # Add a title to the figure:
            fig.update_layout(title_text=f"Phase: {phase}")
            fig.write_html(f"episodes/{trial_num}/{phase}.html")
            # website.add_plot(task_spec.name, f"{task_spec.name}_{trial_num}", fig)
            website_plots.append(
                pio.to_html(fig, full_html=False, include_plotlyjs=False)
            )

        # Save the episode video.
        fn = f"episodes/{trial_num}/video"
        recorder.write_video(fn)
        # website.add_video(task_spec.name, f"{task_spec.name}_{trial_num}", f"video.mov")
        website_video = f"video.mov"

        # Save the episode results as a text file.
        with open(f"episodes/{trial_num}/results.txt", "w") as f:

            # Print overall task success.
            f.write(f"OVERALL TASK SUCCESS: {success}\n")

            for phase_result in pr():
                f.write(f"{phase_result.phase}: {phase_result.failure_reason}\n")

        #
        html = tuw.render_episode_page(
            title=f"Episode {trial_num}, {task_spec.name}",
            phase_plots=website_plots,
            video=website_video,
        )
        with open(f"episodes/{trial_num}/index.html", "w") as f:
            f.write(html)

        # website.write_site(f"episodes/{trial_num}")


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

    # Make sure the model weights can be downloaded.
    get_weights_path(cfg.checkpoints.ckpt_file, cfg.wandb, run=run)

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

    html = tuw.render_experiment_page(
        title=f"Experiment {run.id}",
        episode_nums=range(cfg.num_trials),
    )
    with open(results_dir / "index.html", "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
