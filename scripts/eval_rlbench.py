from abc import abstractmethod
from dataclasses import dataclass
from typing import Protocol

import hydra
import numpy as np
import open3d as o3d
import torch
from omegaconf import OmegaConf
from pytorch3d.ops import sample_farthest_points
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import StackWine
from rpad.rlbench_utils.placement_dataset import (
    TASK_DICT,
    get_rgb_point_cloud_by_object_names,
    obs_to_rgb_point_cloud,
)
from scipy.spatial.transform import Rotation as R

from taxpose.nets.transformer_flow import ResidualFlow_DiffEmbTransformer
from taxpose.training.flow_equivariance_training_module_nocentering import (
    EquivarianceTrainingModule,
)
from taxpose.utils.load_model import get_weights_path


class PickPlacePredictor(Protocol):
    @abstractmethod
    def grasp(self, obs) -> np.ndarray:
        pass

    @abstractmethod
    def place(self, init_obs, post_grasp_obs) -> np.ndarray:
        pass


@dataclass
class TAXPosePickPlacePredictorConfig:
    pass


def create_coordinate_frame(T, size=0.1):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame.transform(T)
    return frame


class TAXPosePickPlacePredictor:
    def __init__(self, model_cfg, task_cfg, wandb_cfg, checkpoints_cfg):
        self.grasp_model = self.load_model(
            checkpoints_cfg.grasp, model_cfg, wandb_cfg, task_cfg.grasp_task
        )
        self.place_model = self.load_model(
            checkpoints_cfg.place, model_cfg, wandb_cfg, task_cfg.place_task
        )

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
    def load_model(model_path, model_cfg, wandb_cfg, task_cfg):
        ckpt_file = get_weights_path(model_path, wandb_cfg)
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

    def grasp(self, obs) -> np.ndarray:
        inputs = obs_to_input(obs, "stack_wine", "grasp")

        action_pc = inputs["action_pc"].unsqueeze(0).to(self.grasp_model.device)
        anchor_pc = inputs["anchor_pc"].unsqueeze(0).to(self.grasp_model.device)

        action_pc, _ = sample_farthest_points(action_pc, K=256, random_start_point=True)
        anchor_pc, _ = sample_farthest_points(anchor_pc, K=256, random_start_point=True)

        preds = self.grasp_model(action_pc, anchor_pc, None, None)

        # Get the current pose of the gripper.
        T_gripper_world = np.eye(4)
        T_gripper_world[:3, 3] = obs.gripper_pose[:3]
        T_gripper_world[:3, :3] = R.from_quat(obs.gripper_pose[3:]).as_matrix()
        T_gripperfinal_gripper = preds["pred_T_action"].get_matrix()[0].T.cpu().numpy()
        T_gripperfinal_world = T_gripperfinal_gripper @ T_gripper_world

        self.render(obs, inputs, preds, T_gripper_world, T_gripperfinal_world)

        return T_gripperfinal_world

    def place(self, init_obs, post_grasp_obs) -> np.ndarray:
        inputs = obs_to_input(init_obs, "stack_wine", "place")

        action_pc = inputs["action_pc"].unsqueeze(0).to(self.place_model.device)
        anchor_pc = inputs["anchor_pc"].unsqueeze(0).to(self.place_model.device)

        action_pc, _ = sample_farthest_points(action_pc, K=256, random_start_point=True)
        anchor_pc, _ = sample_farthest_points(anchor_pc, K=256, random_start_point=True)

        preds = self.place_model(action_pc, anchor_pc, None, None)

        # Get the current pose of the gripper.
        T_gripper_world = np.eye(4)
        T_gripper_world[:3, 3] = post_grasp_obs.gripper_pose[:3]
        T_gripper_world[:3, :3] = R.from_quat(
            post_grasp_obs.gripper_pose[3:]
        ).as_matrix()

        T_gripperfinal_gripper = preds["pred_T_action"].get_matrix()[0].T.cpu().numpy()
        T_gripperfinal_world = T_gripperfinal_gripper @ T_gripper_world

        self.render(init_obs, inputs, preds, T_gripper_world, T_gripperfinal_world)

        return T_gripperfinal_world


class RandomPickPlacePredictor(PickPlacePredictor):
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
        # R_rand = R.random().as_matrix()
        # R_rand = np.eye(3)
        # # Rotate 180 degrees about the x-axis.
        # R_rand[1, 1] = -1
        # R_rand[2, 2] = -1

        T_rand = np.eye(4)
        T_rand[:3, :3] = R_rand
        T_rand[:3, 3] = t_rand
        return T_rand

    def grasp(self, obs) -> np.ndarray:
        return self._get_random_pose(
            self.xrange, self.yrange, self.zrange, self.grip_rot
        )

    def place(self, obs) -> np.ndarray:
        return self._get_random_pose(
            self.xrange, self.yrange, self.zrange, self.grip_rot
        )


# TASK_DICT = {
#     "stack_wine": {
#         "phase": {
#             "grasp": {
#                 "action_obj_handles": {
#                     "Panda_leftfinger_visual",
#                     "Panda_rightfinger_visual",
#                     "Panda_gripper_visual",
#                 },
#                 "anchor_obj_handles": {"wine_bottle_visual"},
#             },
#             "place": {
#                 "action_obj_handles": {"wine_bottle_visual"},
#                 "anchor_obj_handles": {"rack_bottom_visual", "rack_top_visual"},
#             },
#         }
#     }
# }


# TODO: put this in the original rlbench library.
def obs_to_input(obs, task_name, phase):
    rgb, point_cloud, mask = obs_to_rgb_point_cloud(obs)

    action_rgb, action_point_cloud = get_rgb_point_cloud_by_object_names(
        rgb,
        point_cloud,
        mask,
        TASK_DICT[task_name]["phase"][phase]["action_obj_names"],
    )

    # Get the rgb and point cloud for the anchor objects.
    anchor_rgb, anchor_point_cloud = get_rgb_point_cloud_by_object_names(
        rgb,
        point_cloud,
        mask,
        TASK_DICT[task_name]["phase"][phase]["anchor_obj_names"],
    )

    return {
        "action_rgb": torch.from_numpy(action_rgb),
        "action_pc": torch.from_numpy(action_point_cloud).float(),
        "anchor_rgb": torch.from_numpy(anchor_rgb),
        "anchor_pc": torch.from_numpy(anchor_point_cloud).float(),
    }


@torch.no_grad()
def run_trial(policy_spec, task_spec, env_spec):
    policy = TAXPosePickPlacePredictor(
        policy_spec, task_spec, task_spec.wandb, task_spec.checkpoints
    )
    # Create the environment.
    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(collision_checking=True),
            gripper_action_mode=Discrete(),
        ),
        headless=False,
    )

    task = env.get_task(StackWine)
    task.sample_variation()

    # Reset the environment.
    descriptions, obs = task.reset()

    # First, move the arm down a bit so that the cameras can see it!
    # This is a hack to get around the fact that the cameras are not
    # in the same position as the robot's arm.
    down_action = np.array([*obs.gripper_pose, 1.0])
    down_action[2] -= 0.4
    obs, reward, terminate = task.step(down_action)

    # policy = RandomPickPlacePredictor(
    #     None,
    #     xrange=(env._scene._workspace_minx, env._scene._workspace_maxx),
    #     yrange=(env._scene._workspace_miny, env._scene._workspace_maxy),
    #     zrange=(env._scene._workspace_minz, env._scene._workspace_maxz),
    #     grip_rot=obs.gripper_pose[3:],
    # )

    # Make predictions for grasp and place.
    T_grippergrasp_world = policy.grasp(obs)

    breakpoint()

    # Execute the grasp. The transform needs to be converted from a 4x4 matrix to xyz+quat.
    # And the gripper action is a single dimensional action.
    p_grippergrasp_world = T_grippergrasp_world[:3, 3]
    q_grippergrasp_world = R.from_matrix(T_grippergrasp_world[:3, :3]).as_quat()
    gripper_close = np.array([0.0])
    grasp_action = np.concatenate(
        # [p_grippergrasp_world, q_grippergrasp_world, gripper_close]
        [
            p_grippergrasp_world,
            q_grippergrasp_world,
            gripper_close,
        ]
    )

    # TODO remove this because it's in the wrong order.
    # T_bottle_world = policy.place(obs, obs)

    # breakpoint()

    post_grasp_obs, reward, terminate = task.step(grasp_action)

    # Lift the object off the table. Add .1 to the z-coordinate.
    p_gripperlift_world = post_grasp_obs.gripper_pose[:3]
    p_gripperlift_world[2] += 0.1
    q_gripperlift_world = post_grasp_obs.gripper_pose[3:]
    grasp_action = np.concatenate(
        [p_gripperlift_world, q_gripperlift_world, gripper_close]
    )

    obs, reward, terminate = task.step(grasp_action)

    breakpoint()

    # env._action_mode.arm_action_mode = EndEffectorPoseViaPlanning(
    #     collision_checking=False
    # )

    T_bottle_world = policy.place(obs, post_grasp_obs)

    breakpoint()

    # Execute the place.
    p_bottle_world = T_bottle_world[:3, 3]
    q_bottle_world = R.from_matrix(T_bottle_world[:3, :3]).as_quat()
    gripper_open = np.array([1.0])

    # Add 0.03 to the z-coordinate.
    # TODO: beisner, this is a hack. It should be dead on, but the motion planner doesn't like this.
    # The "right" fix would be to do some sort of motion planning improvement to get like, a min-collision
    # trajectory...
    p_bottle_world[2] += 0.08
    place_action = np.concatenate([p_bottle_world, q_bottle_world, gripper_open])

    obs, reward, terminate = task.step(place_action)

    # Close the environment.


@hydra.main(config_path="../configs", config_name="eval_rlbench")
def main(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    run_trial(cfg.model, cfg, None)


if __name__ == "__main__":
    main()
