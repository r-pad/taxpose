"""
This file uses OMPL to generate training data for goal-conditioned FF placement task
"""

import os
import pickle
import sys
import time
from itertools import product

import imageio
import numpy as np
import pybullet as p
from ompl import base as ob
from ompl import geometric as og
from rpad.partnet_mobility_utils.render.pybullet import PMRenderEnv
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from taxpose.datasets.pm_placement import (
    ACTION_OBJS,
    ALL_BLOCK_DSET_PATH,
    SEM_CLASS_DSET_PATH,
)


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


def checker(block_id, sim):
    return (
        len(
            p.getClosestPoints(
                bodyA=block_id,
                bodyB=sim.obj_id,
                distance=0,
                physicsClientId=sim.client_id,
            )
        )
        == 0
    )


def randomize_block_pose(seed=None):
    if seed:
        np.random.seed(seed)
    randomized_pose = np.array(
        [
            np.random.uniform(low=-1, high=-0.8),
            np.random.uniform(low=-1.4, high=-1.2),
            np.random.uniform(low=0.1, high=0.15),
        ]
    )
    return randomized_pose


def is_state_valid(state):
    state_list = [state[i] for i in range(3)]
    value = state_list
    p.resetBasePositionAndOrientation(
        block_id,
        posObj=value[:3],
        ornObj=[0, 0, 0, 1],
        physicsClientId=sim.client_id,
    )
    if not (checker(block_id, sim) and value[2] >= 0.1):
        return False
    return True


if __name__ == "__main__":
    # Load relevant data
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str)
    parser.add_argument("pm_dir", type=str)
    args = parser.parse_args()
    usr_inp_result_dir = args.save_path
    pm_raw_dir = args.pm_dir

    _dset = pickle.load(
        open(
            ALL_BLOCK_DSET_PATH,
            "rb",
        )
    )
    full_sem_dset = pickle.load(
        open(
            SEM_CLASS_DSET_PATH,
            "rb",
        )
    )
    block = ACTION_OBJS["block"].urdf

    # Set up global OMPL params
    joint_limits = [
        (-1.55, 0.3),
        (-1.4, 1.4),
        (0.1, 1.8),
        (-1, 1),
        (-1, 1),
        (-1, 1),
        (-1, 1),
    ]
    n_joints = 7
    bounds = ob.RealVectorBounds(n_joints)
    for i, (low, hi) in enumerate(joint_limits):
        bounds.setHigh(i, hi)
        bounds.setLow(i, low)

    # Create directory for saving the data
    result_dir = os.path.expanduser(f"./taxpose/datasets/pm_data/{usr_inp_result_dir}")
    if not os.path.exists(result_dir):
        print("Creating dataset directory")
        os.makedirs(result_dir, exist_ok=True)

    for obj in tqdm(_dset):
        for id in _dset[obj]:
            # Get designated joints to open
            partsem = _dset[obj][id]["partsem"]
            if partsem != "none":
                for mode in full_sem_dset:
                    if partsem in full_sem_dset[mode]:
                        if id.split("_")[0] in full_sem_dset[mode][partsem]:
                            move_joints = full_sem_dset[mode][partsem][id.split("_")[0]]
            trial = 0
            goal_pos = [
                _dset[obj][id]["x"],
                _dset[obj][id]["y"],
                _dset[obj][id]["z"],
            ]
            while trial < 5:
                if f"{id}_traj_{trial}.npy" in os.listdir(result_dir):
                    print("skippin")
                    trial += 1
                    continue
                sim = PMRenderEnv(id.split("_")[0], os.path.expanduser(f"{pm_raw_dir}"))

                # Open designated door according to dataset
                if partsem != "none":
                    sim.articulate_specific_joints(
                        move_joints[_dset[obj][id]["ind"]], 0.9
                    )
                block_id = p.loadURDF(
                    block, physicsClientId=sim.client_id, globalScaling=4
                )
                valid_start = False
                while not valid_start:
                    start = randomize_block_pose(
                        (os.getpid() * int(time.time())) % 123456789
                    )
                    angle = np.random.uniform(-60, 60)
                    start_ort = R.from_euler("z", angle, degrees=True).as_quat()
                    p.resetBasePositionAndOrientation(
                        block_id,
                        posObj=start,
                        ornObj=[0, 0, 0, 1],
                        physicsClientId=sim.client_id,
                    )
                    valid_start = is_action_pose_valid(block_id, sim)
                start_x, start_y, start_z = start
                p.resetBasePositionAndOrientation(
                    block_id,
                    posObj=[start_x, start_y, start_z],
                    ornObj=start_ort,
                    physicsClientId=sim.client_id,
                )
                state_space = ob.RealVectorStateSpace(n_joints)
                startstate = False

                # Describe the general bounds of the optimization problem.
                state_space.setBounds(bounds)

                moving_bodies = [block_id]
                obstacles = [0, sim.obj_id]
                check_body_pairs = list(product(moving_bodies, obstacles))

                # construct an instance of space information from this state space
                si = ob.SpaceInformation(state_space)
                # set state validity checking for this space
                si.setStateValidityChecker(ob.StateValidityCheckerFn(is_state_valid))

                # Set up starting and goal configurations
                start, goal = ob.State(state_space), ob.State(state_space)
                start[0] = start_x
                start[1] = start_y
                start[2] = start_z
                start[3] = start_ort[0]
                start[4] = start_ort[1]
                start[5] = start_ort[2]
                start[6] = start_ort[3]
                goal[0] = goal_pos[0]
                goal[1] = goal_pos[1]
                goal[2] = goal_pos[2]
                goal[3] = 0
                goal[4] = 0
                goal[5] = 0
                goal[6] = 1

                # create a problem instance
                pdef = ob.ProblemDefinition(si)
                # set the start and goal states
                pdef.setStartAndGoalStates(start, goal)
                # create a planner for the defined space
                planner = og.RRTstar(si)
                # set the problem we are trying to solve for the planner
                planner.setProblemDefinition(pdef)
                # perform setup steps for the planner
                planner.setup()
                # print the settings for this space
                print(si.settings())
                # print the problem settings
                print(pdef)
                # attempt to solve the problem within one second of planning time
                solved = planner.solve(1.0)

                if solved:
                    # get the goal representation from the problem definition (not the same as the goal state)
                    # and inquire about the found path
                    path = pdef.getSolutionPath()
                    print("Found solution:\n%s" % path)
                else:
                    print("No solution found")
                    p.disconnect()
                    continue
                ps = og.PathSimplifier(pdef.getSpaceInformation())
                ps.simplifyMax(path)
                # ps.smoothBSpline(path)
                path.interpolate(10)
                path_states = path.getStates()
                path_list = np.array(
                    [[state[i] for i in range(n_joints)] for state in path_states]
                )
                # Save the trajectory
                np.save(os.path.join(result_dir, f"{id}_traj_{trial}.npy"), path_list)
                trial += 1
                p.disconnect()
