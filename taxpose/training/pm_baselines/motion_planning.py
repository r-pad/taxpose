import numpy as np
import trimesh
from ompl import base as ob
from ompl import geometric as og
from scipy.interpolate import interp1d


def interp_traj(src, dst, interp_len):
    """
    This interpolates the last piece of the trajectory if goal is not reachable.
    """
    x = np.linspace(0, 1, 2)
    interp_x = interp1d(x, [src[0], dst[0]])
    interp_y = interp1d(x, [src[1], dst[1]])
    interp_z = interp1d(x, [src[2], dst[2]])
    interp_path = []
    for delta in np.linspace(0, 1, interp_len):
        pt_t = np.array([interp_x(delta), interp_y(delta), interp_z(delta)])
        interp_path.append(pt_t)

    return interp_path


def setup_planner(start_pose, goal_pose, is_state_valid):
    """
    This sets up the planner using OMPL.
    """
    # joint_limits = [(-1.5, 0.5), (-1.4, 1.4), (0.1, 1.8)]
    joint_limits = [(-2.5, 2.5), (-1.4, 1.4), (0.05, 1.8)]
    n_joints = 3
    bounds = ob.RealVectorBounds(n_joints)
    for i, (low, hi) in enumerate(joint_limits):
        bounds.setHigh(i, hi)
        bounds.setLow(i, low)
    state_space = ob.RealVectorStateSpace(n_joints)
    startstate = False

    # Describe the general bounds of the optimization problem.
    state_space.setBounds(bounds)
    # construct an instance of space information from this state space
    si = ob.SpaceInformation(state_space)
    # set state validity checking for this space
    si.setStateValidityChecker(ob.StateValidityCheckerFn(is_state_valid))
    start, goal = ob.State(state_space), ob.State(state_space)
    start[0] = start_pose[0, -1]
    start[1] = start_pose[1, -1]
    start[2] = start_pose[2, -1]

    # TODO: Should we put the block slightly outside to avoid wrong prediction?
    goal[0] = goal_pose[0, -1]
    goal[1] = goal_pose[1, -1]
    goal[2] = goal_pose[2, -1]
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
    return planner, pdef, n_joints


def motion_planning_fcl(
    inferred_goal: np.ndarray,
    curr_pcd: np.ndarray,
    full_curr_pcd: np.ndarray,
    mask: np.ndarray,
    full_mask: np.ndarray,
    start_obj_pose: np.ndarray,
):
    """
    This does motion planning based on the inferred goal PCD (inferred_goal) and obs PCD (curr_obs)
    Returns a planned path in the format of a numpy array

    INFERRED_GOAL: The predicted goal from the goal prediction model.
    CURR_PCD: Current subsampled PCD observation of the scene.
    FULL_CURR_PCD: Current FULL PCD observation of the scene. For better collision checking/mesh reconstruction.
    MASK: Mask of the object.
    FULL_MASK: UN-subsampled mask.
    START_OBJ_POSE: Current object pose as a 4x4 homogenous matrix.

    Returns: A planned path. If OMPL decides to terminate the path prematurely, we interpolate.
    """
    # R, t = fit_rigid_transformation(curr_pcd[mask], inferred_goal[mask])
    R = np.eye(3)
    t = (inferred_goal[mask] - curr_pcd[mask]).mean(axis=0)
    pose_diff = np.vstack(
        [
            np.hstack([R, t.reshape(3, 1)]),
            np.array([0, 0, 0, 1]).reshape(1, 4),
        ]
    )
    goal_full_pose = pose_diff @ start_obj_pose

    # State checker function
    def checker(
        proposed_pos: np.ndarray,
        curr_state: np.ndarray,
        curr_pcd: np.ndarray,
        mask: np.ndarray,
        goal_xyz: np.ndarray,
    ):
        proposed_flow = np.tile(
            proposed_pos - curr_state[:3, -1], (curr_pcd.shape[0], 1)
        )
        proposed_flow[~mask] = 0
        # proposed_goal = curr_pcd + proposed_flow
        object_proposed_pose = np.eye(4)
        object_proposed_pose[:3, -1] = proposed_pos
        object_proposed_pose[:3, :3] = start_obj_pose[:3, :3]

        # Set up FCL
        pcA = trimesh.PointCloud(curr_pcd[~mask])
        scene_mesh = trimesh.voxel.ops.points_to_marching_cubes(pcA.vertices, pitch=0.1)
        pcB = trimesh.PointCloud(curr_pcd[mask] + proposed_flow[mask])

        xlim = curr_pcd[mask][:, 0].max() - curr_pcd[mask][:, 0].min()
        ylim = curr_pcd[mask][:, 1].max() - curr_pcd[mask][:, 1].min()
        zlim = curr_pcd[mask][:, 2].max() - curr_pcd[mask][:, 2].min()
        obj_mesh = trimesh.creation.box(extents=[xlim, ylim, zlim])

        scene_manager = trimesh.collision.CollisionManager()
        obj_manager = trimesh.collision.CollisionManager()
        scene_manager.add_object("scene", scene_mesh)
        # print(proposed_pos)
        obj_manager.add_object("obj", obj_mesh, transform=object_proposed_pose)
        coll_out = scene_manager.in_collision_other(obj_manager)

        # Debugging visualization
        # print(coll_out)
        # s = trimesh.scene.scene.Scene()
        # s.add_geometry(obj_mesh, node_name="obj")
        # s.add_geometry(scene_mesh, node_name="scene")
        # s.graph.update("obj", "world", matrix=object_proposed_pose)
        # s.show("gl")

        valid_pos = proposed_pos[0] <= goal_xyz[0] and proposed_pos[1] >= goal_xyz[1]
        return not coll_out

    # Valid state function for solver
    def is_state_valid(state):
        state_list = [state[i] for i in range(3)]
        value = state_list
        proposed_pos = value[:3]

        # Pass in full pcd for better performance
        if not (
            checker(
                proposed_pos,
                start_obj_pose,
                full_curr_pcd,
                full_mask,
                goal_full_pose[:3, -1],
            )
        ):
            return False
        return True

    # Set up planner
    planner, pdef, n_joints = setup_planner(
        start_obj_pose, goal_full_pose, is_state_valid
    )
    solved = planner.solve(25.0)

    if solved:
        # get the goal representation from the problem definition (not the same as the goal state)
        # and inquire about the found path
        path = pdef.getSolutionPath()
        print("Found solution:\n%s" % path)
    else:
        print("No solution found")
        return None

    # TODO: Should we simplify the path?
    ps = og.PathSimplifier(pdef.getSpaceInformation())
    ps.simplifyMax(path)
    # ps.smoothBSpline(path)

    path.interpolate(30)
    path_states = path.getStates()
    path_list = np.array([[state[i] for i in range(n_joints)] for state in path_states])
    print(path_list[-1])
    print(goal_full_pose[:3, -1])
    if np.linalg.norm(goal_full_pose[:3, -1] - path_list[-1]) > 5e-2:
        # Interpolating the path
        print("Interpolating the last segment.")
        path_list = np.concatenate(
            [
                path_list,
                interp_traj(path_list[-1], goal_full_pose[:3, -1], interp_len=10),
            ]
        )
        # return None

    return path_list
