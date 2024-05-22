import os
import os.path as osp
import re
import time

import numpy as np
import torch
from pytorch3d.transforms import Rotate, Transform3d, Translate
from scipy.spatial.transform import Rotation as R

from taxpose.utils.se3 import (
    get_2transform_min_rotation_errors,
    get_2transform_min_translation_errors,
    get_degree_angle,
    get_transform_list_min_rotation_errors,
    get_transform_list_min_translation_errors,
    get_translation,
)


def get_2rack_errors(
    pred_T_action,
    T0,
    T1,
    mode="batch_min_rack",
    verbose=False,
    T_aug_list=None,
    action_center=None,
    anchor_center=None,
):
    """
    Debugging function for calculating the error in predicting a mug-on-rack transform
    when there are 2 possible racks to place the mug on, and they are 0.3m apart in the x direction
    """
    assert mode in ["demo_rack", "bad_min_rack", "batch_min_rack", "aug_min_rack"]

    if mode == "demo_rack":
        gt_T_action = T0.inverse().compose(T1)
        if action_center is not None and anchor_center is not None:
            gt_T_action = (
                action_center.inverse().compose(gt_T_action).compose(anchor_center)
            )

        error_R_max, error_R_min, error_R_mean = get_degree_angle(
            gt_T_action.compose(pred_T_action.inverse())
        )
        error_t_max, error_t_min, error_t_mean = get_translation(
            gt_T_action.compose(pred_T_action.inverse())
        )
    elif mode == "bad_min_rack":
        error_R_max0, error_R_min0, error_R_mean0 = get_degree_angle(
            T0.inverse().compose(T1).compose(pred_T_action.inverse())
        )
        error_t_max0, error_t_min0, error_t_mean0 = get_translation(
            T0.inverse().compose(T1).compose(pred_T_action.inverse())
        )
        error_R_max1, error_R_min1, error_R_mean1 = get_degree_angle(
            T0.inverse()
            .translate(0.3, 0, 0)
            .compose(T1)
            .compose(pred_T_action.inverse())
        )
        error_R_max2, error_R_min2, error_R_mean2 = get_degree_angle(
            T0.inverse()
            .translate(-0.3, 0, 0)
            .compose(T1)
            .compose(pred_T_action.inverse())
        )
        error_t_max1, error_t_min1, error_t_mean1 = get_translation(
            T0.inverse()
            .translate(0.3, 0, 0)
            .compose(T1.compose(pred_T_action.inverse()))
        )
        error_t_max2, error_t_min2, error_t_mean2 = get_translation(
            T0.inverse()
            .translate(-0.3, 0, 0)
            .compose(T1.compose(pred_T_action.inverse()))
        )
        error_R_mean = min(error_R_mean0, error_R_mean1, error_R_mean2)
        error_t_mean = min(error_t_mean0, error_t_mean1, error_t_mean2)
        if verbose:
            print(
                f"\t\/ a min over {error_t_mean0:.3f}, {error_t_mean1:.3f}, {error_t_mean2:.3f}"
            )
    elif mode == "batch_min_rack":
        T = T0.inverse().compose(T1).compose(pred_T_action.inverse())
        Ts = torch.stack(
            [
                T0.inverse().compose(T1).compose(pred_T_action.inverse()).get_matrix(),
                T0.inverse()
                .translate(0.3, 0, 0)
                .compose(T1)
                .compose(pred_T_action.inverse())
                .get_matrix(),
                T0.inverse()
                .translate(-0.3, 0, 0)
                .compose(T1)
                .compose(pred_T_action.inverse())
                .get_matrix(),
            ]
        )

        error_R_mean, error_t_mean = 0, 0
        B = T0.get_matrix().shape[0]
        if verbose:
            print("\t\/ an average over ", end="")
        for b in range(B):  # for every batch
            _max, error_R_min, _mean = get_degree_angle(
                Transform3d(matrix=Ts[:, b, :, :])
            )
            error_R_mean += error_R_min

            _max, error_t_min, _mean = get_translation(
                Transform3d(matrix=Ts[:, b, :, :])
            )
            error_t_mean += error_t_min
            if verbose:
                print(f"{error_t_min:.3f}", end=" ")
        if verbose:
            print()
        error_R_mean /= B
        error_t_mean /= B
    elif mode == "aug_min_rack":
        assert (
            T_aug_list is not None
        ), "T_aug_list must be provided for aug_min_rack mode"

        gt_T_action = T0.inverse().compose(T1)
        if action_center is not None and anchor_center is not None:
            gt_T_action = (
                action_center.inverse().compose(gt_T_action).compose(anchor_center)
            )

        aug_T_list = []
        for T_aug in T_aug_list:
            aug_T_action = T0.inverse().compose(T_aug).compose(T1)
            if action_center is not None and anchor_center is not None:
                aug_T_action = (
                    action_center.inverse().compose(aug_T_action).compose(anchor_center)
                )
            aug_T_list.append(aug_T_action)

        T_demo = gt_T_action.compose(pred_T_action.inverse())

        T_distractor_list = []
        for aug_T_action in aug_T_list:
            T_distractor = aug_T_action.compose(pred_T_action.inverse())
            T_distractor_list.append(T_distractor)

        error_t_max, error_t_min, error_t_mean = (
            get_transform_list_min_translation_errors(T_demo, T_distractor_list)
        )
        error_R_max, error_R_min, error_R_mean = get_transform_list_min_rotation_errors(
            T_demo, T_distractor_list
        )
    else:
        raise ValueError("Invalid rack error type!")

    return error_R_mean, error_t_mean


def print_rack_errors(name, error_R_mean, error_t_mean):
    print(f"{name}- R error: {error_R_mean:.3f}, t error: {error_t_mean:.3f}")


def get_all_sample_errors(pred_T_actions, T0, T1, mode="demo_rack", T_aug_list=None):
    error_R_maxs, error_R_mins, error_R_means = [], [], []
    error_t_maxs, error_t_mins, error_t_means = [], [], []
    for pred_T_action in pred_T_actions:
        if mode == "demo_rack":
            error_R_max, error_R_min, error_R_mean = get_degree_angle(
                T0.inverse().compose(T1).compose(pred_T_action.inverse())
            )

            error_t_max, error_t_min, error_t_mean = get_translation(
                T0.inverse().compose(T1).compose(pred_T_action.inverse())
            )
        elif mode == "aug_min_rack":
            assert (
                T_aug_list is not None
            ), "T_aug_list must be provided for aug_min_rack mode"

            gt_T_action = T0.inverse().compose(T1)

            aug_T_list = []
            for T_aug in T_aug_list:
                aug_T_action = T0.inverse().compose(T_aug).compose(T1)
                aug_T_list.append(aug_T_action)

            T_demo = gt_T_action.compose(pred_T_action.inverse())

            T_distractor_list = []
            for aug_T_action in aug_T_list:
                T_distractor = aug_T_action.compose(pred_T_action.inverse())
                T_distractor_list.append(T_distractor)

            error_t_max, error_t_min, error_t_mean = (
                get_transform_list_min_translation_errors(T_demo, T_distractor_list)
            )
            error_R_max, error_R_min, error_R_mean = (
                get_transform_list_min_rotation_errors(T_demo, T_distractor_list)
            )
        else:
            raise ValueError(f"Sample errors not implemented for: {mode}")

        error_R_maxs.append(error_R_max)
        error_R_mins.append(error_R_min)
        error_R_means.append(error_R_mean)
        error_t_maxs.append(error_t_max)
        error_t_mins.append(error_t_min)
        error_t_means.append(error_t_mean)

    return (
        error_R_maxs,
        error_R_mins,
        error_R_means,
        error_t_maxs,
        error_t_mins,
        error_t_means,
    )


def matrix_from_list(pose_list):
    trans = pose_list[:3]
    quat = pose_list[3:]

    T = np.eye(4)
    T[:-1, :-1] = R.from_quat(quat).as_matrix()
    T[:-1, -1] = trans
    return T


def get_rpdiff_min_errors(
    pred_T_action, batch, trans_thresh=0.02, rpdiff_descriptions_path=None
):
    T0 = Transform3d(matrix=batch["T0"])
    T1 = Transform3d(matrix=batch["T1"])

    # We are doing the new WTA implementation for mug on rack
    if "scene_rack_transforms" in batch:
        # Each sample in the batch potentially has a variable number of racks,
        # so batches are padded with empty or NaN elements. Need to extract the valid info
        # Get each batch idx's mug id
        batch_mug_ids_list = []
        for idx in range(len(batch["ids"]["mug"])):
            batch_mug_ids_list.append(batch["ids"]["mug"][idx])

        # Get each batch idx's rack ids
        batch_rack_ids_list = []
        for idx in range(len(batch["ids"]["racks"])):
            for batch_idx, rack in enumerate(batch["ids"]["racks"][idx]):
                if idx == 0:
                    batch_rack_ids_list.append([])
                if rack != "":
                    batch_rack_ids_list[batch_idx].append(rack)

        # Get each batch idx's demo rack idx
        batch_demo_rack_idx = batch["demo_rack_idx"]["demo_rack_idx"]
        # Get each batch idx's rack final poses
        batch_final_poses = batch["final_poses"]
        batch_rack_final_poses_list = []
        for idx, rack_pose_list in enumerate(batch_final_poses["racks"]):
            nans = torch.all(torch.all(rack_pose_list.isnan(), dim=1) == True, dim=1)
            cur_pose_tensor = rack_pose_list[~nans]
            batch_rack_final_poses_list.append(cur_pose_tensor.squeeze(1))

        # Get each batch idx's mug final poses
        batch_mug_final_poses = torch.stack(batch_final_poses["mug"][0])
        batch_mug_final_poses_list = batch_mug_final_poses.permute(1, 0).unsqueeze(
            1
        )  # Match rack pose dims

        # Get each batch idx's rack transforms
        batch_rack_transforms = torch.stack(
            batch["scene_rack_transforms"]["racks"]
        ).permute(1, 0, 2, 3, 4)
        batch_rack_transforms_list = []
        for batch_idx in range(len(batch_rack_transforms)):
            nans = torch.all(
                torch.all(
                    torch.all(batch_rack_transforms[batch_idx].isnan(), dim=1) == True,
                    dim=1,
                )
                == True,
                dim=1,
            )
            cur_rack_transforms = batch_rack_transforms[batch_idx][~nans]
            batch_rack_transforms_list.append(cur_rack_transforms.squeeze(1))

        # Get each batch idx's available poses for the current mug given the current scene racks (each pose w.r.t respective rack origin)
        batch_avail_mug_poses = []
        for batch_idx in range(len(batch_mug_ids_list)):
            batch_mug_id = batch_mug_ids_list[batch_idx]
            batch_rack_ids = batch_rack_ids_list[batch_idx]
            cur_avail_mug_poses = []
            for rack in batch_rack_ids:
                rack_type = re.match(r"^(syn_rack_[a-zA-Z]+)_[0-9-]+$", rack).group(1)
                rack_avail_mug_poses_dir = osp.join(
                    rpdiff_descriptions_path,
                    f"objects/{rack_type}_unnormalized/available_mug_poses/{rack}",
                )
                cur_mug_poses = np.load(
                    osp.join(rack_avail_mug_poses_dir, f"{batch_mug_id}/mug_poses.npz"),
                    allow_pickle=True,
                )
                cur_avail_mug_poses.append(cur_mug_poses.get("mug_poses"))
            batch_avail_mug_poses.append(cur_avail_mug_poses)

        # Get each batch idx's comparable available poses for the current mug given the current scene racks (each pose w.r.t the action points)
        batch_avail_mug_poses_comparable = []
        for batch_idx in range(len(batch_mug_ids_list)):
            cur_batch_avail_mug_poses_comparable = []
            for rack_idx in range(len(batch_avail_mug_poses[batch_idx])):
                #######################################################
                # Get the available pose transforms from rack scene origin to anchor frame

                # Get the available poses w.r.t the rack origin
                cur_rack_avail_poses = batch_avail_mug_poses[batch_idx][rack_idx]
                avail_mug_poses_tf = Transform3d(
                    matrix=torch.from_numpy(cur_rack_avail_poses)
                    .float()
                    .permute(0, 2, 1)
                ).to(batch["points_action"].device)

                # Get the transform that moves each rack from the origin to its scene pose
                scene_rack_transforms = Transform3d(
                    matrix=batch_rack_transforms_list[batch_idx][rack_idx].T
                ).to(batch["points_action"].device)

                # Get the available poses at the scene rack poses
                avail_mug_poses_scene = avail_mug_poses_tf.compose(
                    scene_rack_transforms
                )

                # Center the available poses about the action mean
                scene_to_action_mean = Translate(
                    -batch["points_action_mean"][batch_idx].unsqueeze(0)
                ).to(batch["points_action"].device)
                avail_mug_poses_scene_centered = avail_mug_poses_scene.compose(
                    scene_to_action_mean
                )

                # Move the available poses to the anchor frame
                avail_poses_origin_to_anchor = avail_mug_poses_scene_centered.compose(
                    T1[batch_idx]
                )

                #######################################################
                # Get the transform from transformed action points to the rack scene origin

                # Get transform from action points to demo position
                transformed_to_demo_tf = T0[batch_idx].inverse()

                # Undo the centering of the action points
                translate_to_action_mean = Translate(
                    batch["points_action_mean"][batch_idx].unsqueeze(0)
                ).to(batch["points_action"].device)
                demo_to_action_mean_tf = transformed_to_demo_tf.compose(
                    translate_to_action_mean
                )

                # Undo rack scene transform
                demo_rack_scene_transform = Transform3d(
                    matrix=batch_rack_transforms_list[batch_idx][
                        batch_demo_rack_idx[batch_idx]
                    ].T
                ).to(batch["points_action"].device)
                action_mean_to_scene_tf = demo_to_action_mean_tf.compose(
                    demo_rack_scene_transform.inverse()
                )

                # Move to demo final rack pose
                demo_rack_final_pose = batch_rack_final_poses_list[batch_idx][
                    batch_demo_rack_idx[batch_idx]
                ]
                demo_rack_final_pose_mat = matrix_from_list(
                    demo_rack_final_pose.detach().cpu().tolist()
                )
                demo_rack_final_pose_tf = Transform3d(
                    matrix=torch.Tensor(demo_rack_final_pose_mat).T
                ).to(batch["points_action"].device)
                scene_to_demo_rack_final_pose_tf = action_mean_to_scene_tf.compose(
                    demo_rack_final_pose_tf
                )

                # Move to mug origin
                cur_mug_final_pose = batch_mug_final_poses_list[batch_idx]
                cur_mug_final_pose_mat = matrix_from_list(
                    cur_mug_final_pose[0].detach().cpu().tolist()
                )
                cur_mug_final_pose_tf = Transform3d(
                    matrix=torch.Tensor(cur_mug_final_pose_mat).T
                ).to(batch["points_action"].device)
                demo_rack_final_pose_to_mug_origin_tf = (
                    scene_to_demo_rack_final_pose_tf.compose(
                        cur_mug_final_pose_tf.inverse()
                    )
                )

                # Move action points to available poses
                transformed_action_to_avail_poses_tf = (
                    demo_rack_final_pose_to_mug_origin_tf.compose(
                        avail_poses_origin_to_anchor
                    )
                )

                cur_batch_avail_mug_poses_comparable.append(
                    transformed_action_to_avail_poses_tf
                )
            batch_avail_mug_poses_comparable.append(
                cur_batch_avail_mug_poses_comparable
            )

        batch_min_dists = []
        for batch_idx in range(len(batch_mug_ids_list)):
            pred_pose_mat = (
                pred_T_action[batch_idx]
                .get_matrix()
                .squeeze(0)
                .T.detach()
                .cpu()
                .numpy()
            )
            # pred_pose_mat = T0[batch_idx].inverse().compose(T1[batch_idx]).get_matrix().squeeze(0).T.detach().cpu().numpy()
            pred_pose_trans = pred_pose_mat[:-1, -1]
            pred_pose_rot = pred_pose_mat[:-1, :-1]

            min_trans_dist = float("inf")
            min_rot_dist = float("inf")
            close_trans_list = []
            for rack_poses in batch_avail_mug_poses_comparable[batch_idx]:
                rack_poses_mat = (
                    rack_poses.get_matrix().permute(0, 2, 1).detach().cpu().numpy()
                )
                for pose in rack_poses_mat:
                    pose_trans = pose[:-1, -1]
                    pose_rot = pose[:-1, :-1]

                    trans_ = np.linalg.norm(pred_pose_trans - pose_trans, axis=-1)

                    # q_pred = R.from_matrix(pred_pose_rot).as_quat()
                    # q_pose = R.from_matrix(pose_rot).as_quat()

                    # quat_scalar_prod = np.sum(q_pred * q_pose)
                    # rot_ = 1 - quat_scalar_prod**2

                    rot_pred = Rotate(torch.Tensor(pred_pose_rot.T)).to(
                        batch["points_action"].device
                    )
                    rot_pose = Rotate(torch.Tensor(pose_rot.T)).to(
                        batch["points_action"].device
                    )
                    _, _, rot_ = get_degree_angle(rot_pose.compose(rot_pred.inverse()))

                    # Find the available pose thats closest to the pred_T_action (in terms of translation)
                    if trans_ < min_trans_dist:
                        min_trans_dist = trans_
                        min_rot_dist = rot_

                    # If pred_T_action is close to many available poses, need to find the one with best translation AND rotation error
                    if trans_ < min_trans_dist + trans_thresh:
                        close_trans_list.append((trans_, rot_))

            min_trans_rot_dist = float("inf")
            for dist_pair in close_trans_list:
                trans_, rot_ = dist_pair

                if trans_ + rot_ < min_trans_rot_dist:
                    min_trans_rot_dist = trans_ + rot_
                    min_trans_dist = trans_
                    min_rot_dist = rot_

            batch_min_dists.append([min_trans_dist, min_rot_dist])

    # We are doing the initial WTA implementation for book or can
    elif "multi_obj_mesh_file" in batch:
        multi_obj_mesh_file = batch["multi_obj_mesh_file"]
        parent_fnames = multi_obj_mesh_file["parent"][0]
        child_fnames = multi_obj_mesh_file["child"][0]

        # Get the final poses of the parent and child
        multi_obj_final_obj_poses = batch["multi_obj_final_obj_pose"]

        parent_final_poses = multi_obj_final_obj_poses["parent"]
        parent_final_poses = torch.stack(parent_final_poses).permute(1, 0, 2)

        child_final_poses = multi_obj_final_obj_poses["child"]
        child_final_poses = torch.stack(child_final_poses).permute(1, 0, 2)

        # Get the mean of the original points
        points_action_means = batch["points_action_mean"]

        batch_min_dists = []
        for batch_idx in range(len(parent_fnames)):
            # Get parent final poses (parent in world frame) as matrices
            parent_final_pose = parent_final_poses[batch_idx]
            parent_final_pose_mat = matrix_from_list(
                parent_final_pose.squeeze(-1).detach().cpu().numpy()
            )

            # Get child final poses (child in world frame) as matrices
            child_final_pose = child_final_poses[batch_idx]
            child_final_pose_mat = matrix_from_list(
                child_final_pose.squeeze(-1).detach().cpu().numpy()
            )

            # Extract rotation and translation, by default mat. rotation component is the inverse of the actual rotation
            parent_final_pose_rot = parent_final_pose_mat[:3, :3]
            parent_final_pose_rot_tf = Rotate(torch.Tensor(parent_final_pose_rot)).to(
                batch["points_action"].device
            )
            parent_final_pose_translation = parent_final_pose_mat[:3, 3]
            parent_final_pose_translation_tf = Translate(
                torch.Tensor(-parent_final_pose_translation).unsqueeze(0)
            ).to(batch["points_action"].device)

            child_final_pose_rot = child_final_pose_mat[:3, :3]
            child_final_pose_rot_tf = Rotate(torch.Tensor(child_final_pose_rot)).to(
                batch["points_action"].device
            )
            child_final_pose_translation = child_final_pose_mat[:3, 3]
            child_final_pose_translation_tf = Translate(
                torch.Tensor(-child_final_pose_translation).unsqueeze(0)
            ).to(batch["points_action"].device)

            # Compose the transform from the parent final frame (parent in world frame) to the parent frame
            parent_final_pose_inv_tf = parent_final_pose_translation_tf.compose(
                parent_final_pose_rot_tf
            )

            # Compose the transform from the child final frame (child in world frame) to the child frame
            child_final_pose_inv_tf = child_final_pose_translation_tf.compose(
                child_final_pose_rot_tf
            )

            # By default the action/anchor points are centered about the action mean
            translate_to_action_mean = Translate(
                -points_action_means[batch_idx].unsqueeze(0)
            ).to(batch["points_action"].device)

            # Transform pose from parent frame to parent final frame (parent in world frame), then to anchor frame (centered about action mean)
            parent_pose_to_anchor_frame = parent_final_pose_inv_tf.inverse().compose(
                translate_to_action_mean
            )

            # Transform pose from child frame to child final frame (child in world frame), then to anchor frame (centered about action mean)
            child_pose_to_anchor_frame = child_final_pose_inv_tf.inverse().compose(
                translate_to_action_mean
            )

            # Transform pose from anchor frame to anchor trans frame
            parent_pose_to_trans_anchor_frame = parent_pose_to_anchor_frame.compose(
                T1[batch_idx]
            )

            # Get the predicted child pose in the trans anchor frame
            child_pred_pose = child_pose_to_anchor_frame.compose(T0[batch_idx]).compose(
                pred_T_action[batch_idx]
            )

            if (
                "book" in child_fnames[batch_idx]
                and "bookshelf" in parent_fnames[batch_idx]
            ):
                parent_fname = parent_fnames[batch_idx]

                bookshelf_name = (
                    parent_fname.split("/")[-1].replace(".obj", "").replace("_dec", "")
                )
                parent_full_fname = osp.join(
                    rpdiff_descriptions_path,
                    parent_fname.split(bookshelf_name)[0].split("descriptions/")[-1],
                )
                saved_available_poses_fname = osp.join(
                    parent_full_fname,
                    "open_slot_poses",
                    bookshelf_name + "_open_slot_poses.txt",
                )

                loaded_poses = np.loadtxt(saved_available_poses_fname)
                loaded_poses = [matrix_from_list(pose) for pose in loaded_poses]

                # get avail poses in the trans anchor frame
                avail_poses_trans_anchor_frame_base = [
                    np.matmul(
                        parent_pose_to_trans_anchor_frame.get_matrix()
                        .squeeze(0)
                        .T.detach()
                        .cpu()
                        .numpy(),
                        pose,
                    )
                    for pose in loaded_poses
                ]

                avail_poses_trans_anchor_frame = []
                for p_idx, pose in enumerate(avail_poses_trans_anchor_frame_base):
                    # get all four orientations that work
                    r1 = R.from_euler("xyz", [0, 0, 0]).as_matrix()
                    r2 = R.from_euler("xyz", [np.pi, 0, 0]).as_matrix()
                    # r3 = R.from_euler('xyz', [0, 0, np.pi]).as_matrix()
                    # r4 = R.from_euler('xyz', [np.pi, 0, np.pi]).as_matrix()
                    r5 = R.from_euler("xyz", [0, np.pi, 0]).as_matrix()
                    r6 = R.from_euler("xyz", [np.pi, np.pi, 0]).as_matrix()
                    # r7 = R.from_euler('xyz', [0, np.pi, np.pi]).as_matrix()
                    # r8 = R.from_euler('xyz', [np.pi, np.pi, np.pi]).as_matrix()

                    tf1 = np.eye(4)
                    tf1[:-1, :-1] = r1
                    tf2 = np.eye(4)
                    tf2[:-1, :-1] = r2
                    # tf3 = np.eye(4); tf3[:-1, :-1] = r3
                    # tf4 = np.eye(4); tf4[:-1, :-1] = r4
                    tf5 = np.eye(4)
                    tf5[:-1, :-1] = r5
                    tf6 = np.eye(4)
                    tf6[:-1, :-1] = r6
                    # tf7 = np.eye(4); tf7[:-1, :-1] = r7
                    # tf8 = np.eye(4); tf8[:-1, :-1] = r8

                    p1 = np.matmul(pose, tf1)
                    p2 = np.matmul(pose, tf2)
                    # p3 = np.matmul(pose, tf3)
                    # p4 = np.matmul(pose, tf4)
                    p5 = np.matmul(pose, tf5)
                    p6 = np.matmul(pose, tf6)
                    # p7 = np.matmul(pose, tf7)
                    # p8 = np.matmul(pose, tf8)

                    # all_poses_to_save = [p1, p2, p3, p4, p5, p6, p7, p8]
                    all_poses_to_save = [p1, p2, p5, p6]

                    # Don't save poses that are too close to existing ones
                    for p_to_save in all_poses_to_save:

                        a_rotmat = p_to_save[:-1, :-1]
                        close_to_existing = False
                        for p2_idx, pose2 in enumerate(avail_poses_trans_anchor_frame):
                            trans_ = np.linalg.norm(
                                p_to_save[:-1, -1] - pose2[:-1, -1], axis=-1
                            )

                            b_rotmat = pose2[:-1, :-1]
                            qa = R.from_matrix(a_rotmat).as_quat()
                            qb = R.from_matrix(b_rotmat).as_quat()

                            quat_scalar_prod = np.sum(qa * qb)
                            rot_ = 1 - quat_scalar_prod**2

                            if trans_ < 0.02 and rot_ < np.deg2rad(5):
                                close_to_existing = True
                                break

                        if not close_to_existing:
                            avail_poses_trans_anchor_frame.append(p_to_save)
                    # avail_poses_trans_anchor_frame.extend(all_poses_to_save)

            elif (
                "can" in child_fnames[batch_idx]
                and "cabinet" in parent_fnames[batch_idx]
            ):
                parent_fname = parent_fnames[batch_idx]

                cabinet_name = (
                    parent_fname.split("/")[-1].replace(".obj", "").replace("_dec", "")
                )
                parent_full_fname = osp.join(
                    rpdiff_descriptions_path,
                    parent_fname.split(cabinet_name)[0].split("descriptions/")[-1],
                )
                saved_available_poses_fname = osp.join(
                    parent_full_fname,
                    "open_slot_poses",
                    cabinet_name + "_open_slot_poses.npz",
                )

                loaded_poses = np.load(saved_available_poses_fname, allow_pickle=True)
                avail_pose_info_all = loaded_poses["avail_top_poses"]

                points_action = batch["points_action"][batch_idx, :, :3]
                # Get the extents of the action points
                action_min = points_action.min(dim=0).values
                action_max = points_action.max(dim=0).values
                action_extents = action_max - action_min

                action_h = action_extents[-1]

                top_poses = [pose_info["pose"] for pose_info in avail_pose_info_all]
                base_poses = []
                for pose in top_poses:
                    base_pose = pose.copy()
                    base_pose[2, -1] += action_h / 2
                    base_poses.append(base_pose)

                # get avail poses in the trans anchor frame
                avail_poses_trans_anchor_frame_base = [
                    np.matmul(
                        parent_pose_to_trans_anchor_frame.get_matrix()
                        .squeeze(0)
                        .T.detach()
                        .cpu()
                        .numpy(),
                        pose,
                    )
                    for pose in base_poses
                ]

                avail_poses_trans_anchor_frame = []
                for p_idx, pose in enumerate(avail_poses_trans_anchor_frame_base):
                    # get all orientations that work
                    r1 = R.from_euler("xyz", [0, 0, 0]).as_matrix()
                    r2 = R.from_euler("xyz", [np.pi, 0, 0]).as_matrix()
                    # r3 = R.from_euler('xyz', [0, 0, np.pi]).as_matrix()
                    # r4 = R.from_euler('xyz', [np.pi, 0, np.pi]).as_matrix()
                    # r5 = R.from_euler('xyz', [0, np.pi, 0]).as_matrix()
                    # r6 = R.from_euler('xyz', [np.pi, np.pi, 0]).as_matrix()
                    # r7 = R.from_euler('xyz', [0, np.pi, np.pi]).as_matrix()
                    # r8 = R.from_euler('xyz', [np.pi, np.pi, np.pi]).as_matrix()

                    tf1 = np.eye(4)
                    tf1[:-1, :-1] = r1
                    tf2 = np.eye(4)
                    tf2[:-1, :-1] = r2
                    # tf3 = np.eye(4); tf3[:-1, :-1] = r3
                    # tf4 = np.eye(4); tf4[:-1, :-1] = r4
                    # tf5 = np.eye(4); tf5[:-1, :-1] = r5
                    # tf6 = np.eye(4); tf6[:-1, :-1] = r6
                    # tf7 = np.eye(4); tf7[:-1, :-1] = r7
                    # tf8 = np.eye(4); tf8[:-1, :-1] = r8

                    p1 = np.matmul(pose, tf1)
                    p2 = np.matmul(pose, tf2)
                    # p3 = np.matmul(pose, tf3)
                    # p4 = np.matmul(pose, tf4)
                    # p5 = np.matmul(pose, tf5)
                    # p6 = np.matmul(pose, tf6)
                    # p7 = np.matmul(pose, tf7)
                    # p8 = np.matmul(pose, tf8)

                    # avail_poses_trans_anchor_frame.append(p1)
                    # avail_poses_trans_anchor_frame.append(p2)

                    # all_poses_to_save = [p1, p2, p3, p4, p5, p6, p7, p8]
                    all_poses_to_save = [p1, p2]

                    for p_to_save in all_poses_to_save:

                        a_rotmat = p_to_save[:-1, :-1]
                        close_to_existing = False
                        for p2_idx, pose2 in enumerate(avail_poses_trans_anchor_frame):
                            trans_ = np.linalg.norm(
                                p_to_save[:-1, -1] - pose2[:-1, -1], axis=-1
                            )

                            b_rotmat = pose2[:-1, :-1]
                            qa = R.from_matrix(a_rotmat).as_quat()
                            qb = R.from_matrix(b_rotmat).as_quat()

                            quat_scalar_prod = np.sum(qa * qb)
                            rot_ = 1 - quat_scalar_prod**2

                            if trans_ < 0.02 and rot_ < np.deg2rad(5):
                                close_to_existing = True
                                break

                        if not close_to_existing:
                            avail_poses_trans_anchor_frame.append(p_to_save)

            child_pred_pose_mat = (
                child_pred_pose.get_matrix().squeeze(0).T.detach().cpu().numpy()
            )
            child_pred_pose_trans = child_pred_pose_mat[:-1, -1]
            child_pred_pose_rot = child_pred_pose_mat[:-1, :-1]

            min_trans_dist = float("inf")
            min_rot_dist = float("inf")
            close_trans_list = []
            for pose in avail_poses_trans_anchor_frame:
                pose_trans = pose[:-1, -1]
                pose_rot = pose[:-1, :-1]

                trans_ = np.linalg.norm(child_pred_pose_trans - pose_trans, axis=-1)

                # q_child_pred = R.from_matrix(child_pred_pose_rot).as_quat()
                # q_pose = R.from_matrix(pose_rot).as_quat()

                # quat_scalar_prod = np.sum(q_child_pred * q_pose)
                # rot_ = 1 - quat_scalar_prod**2

                rot_child_pred = Rotate(torch.Tensor(child_pred_pose_rot.T)).to(
                    batch["points_action"].device
                )
                rot_pose = Rotate(torch.Tensor(pose_rot.T)).to(
                    batch["points_action"].device
                )
                _, _, rot_ = get_degree_angle(
                    rot_pose.compose(rot_child_pred.inverse())
                )

                # Find the available pose thats closest to the pred_T_action (in terms of translation)
                if trans_ < min_trans_dist:
                    min_trans_dist = trans_
                    min_rot_dist = rot_

                # If pred_T_action is close to many available poses, need to find the one with best translation AND rotation error
                if trans_ < min_trans_dist + trans_thresh:
                    close_trans_list.append((trans_, rot_))

            min_trans_rot_dist = float("inf")
            for dist_pair in close_trans_list:
                trans_, rot_ = dist_pair

                if trans_ + rot_ < min_trans_rot_dist:
                    min_trans_rot_dist = trans_ + rot_
                    min_trans_dist = trans_
                    min_rot_dist = rot_

            batch_min_dists.append([min_trans_dist, min_rot_dist])

    # Calculate the batch mean of the min translation and rotation errors
    batch_min_dists = np.array(batch_min_dists)
    batch_min_dists = np.mean(batch_min_dists, axis=0)

    return batch_min_dists
