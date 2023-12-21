from taxpose.utils.se3 import random_se3
import numpy as np
import torch

from pytorch3d.transforms import Transform3d, Rotate, Translate, axis_angle_to_matrix
from pytorch3d.ops import sample_farthest_points


def points_to_axis_aligned_rect(points, buffer=0.1):
    # points: torch.tensor with shape (B, N, 3)
    # buffer: % bigger that the bounding box will be relative to the object on each side
    # Returns (x_low, y_low, z_low, x_high, y_high, z_high)
    assert points.ndim == 3
    rect_prism = torch.hstack([points.min(axis=1)[0], points.max(axis=1)[0]])
    buffer_w = (rect_prism[:, 3:6] - rect_prism[:, 0:3]) * buffer
    rect_prism = rect_prism + torch.hstack([-buffer_w, buffer_w])

    return rect_prism


def combine_axis_aligned_rect(rect_prisms):
    # rect_prisms: list of torch.tensor with shape (B, 6)
    #     - per batch: (xmin, ymin, zmin, xmax, ymax, zmax)
    # Returns (x_low, y_low, z_low, x_high, y_high, z_high)
    assert len(rect_prisms) > 0
    assert all([r.ndim == 2 for r in rect_prisms])
    assert all([r.shape[1] == 6 for r in rect_prisms])
    return torch.hstack(
        [
            torch.min(torch.stack([r[:, 0:3] for r in rect_prisms], dim=0), dim=0)[0],
            torch.max(torch.stack([r[:, 3:6] for r in rect_prisms], dim=0), dim=0)[0],
        ]
    )


# def combine_axis_aligned_rect(rect_prism1, rect_prism2):
#     # rect_prism1: torch.tensor with shape (B, 6)
#     #     - per batch: (xmin1, ymin1, zmin1, xmax1, ymax1, zmax1)
#     # rect_prism2: torch.tensor with shape (B, 6)
#     #     - per batch: (xmin2, ymin2, zmin2, xmax2, ymax2, zmax2)
#     # Returns (x_low, y_low, z_low, x_high, y_high, z_high)
#     assert rect_prism1.shape == rect_prism2.shape
#     assert rect_prism1.ndim == 2
#     return torch.hstack([torch.min(rect_prism1[:, 0:3], rect_prism2[:, 0:3]),
#                          torch.max(rect_prism1[:, 3:6], rect_prism2[:, 3:6])])


def axis_aligned_rect_intersect(rect_prism1, rect_prism2):
    # intersect 2 axis-aligned rectangular prisms
    # rect_prism1: torch.tensor with shape (B, 6)
    #     - per batch: (xmin1, ymin1, zmin1, xmax1, ymax1, zmax1)

    conditions = (
        (rect_prism1[:, 0] <= rect_prism2[:, 3]).int()
        + (rect_prism1[:, 3] >= rect_prism2[:, 0]).int()
        + (rect_prism1[:, 1] <= rect_prism2[:, 4]).int()
        + (rect_prism1[:, 4] >= rect_prism2[:, 1]).int()
        + (rect_prism1[:, 2] <= rect_prism2[:, 5]).int()
        + (rect_prism1[:, 5] >= rect_prism2[:, 2]).int()
    )
    return conditions >= 6


def random_transform_rack(
    points_anchor, rot_var=np.pi, trans_var=1, rot_sample_method="axis_angle"
):
    N = points_anchor.shape[0]
    device = points_anchor.device

    # offset = points_anchor.mean(axis=1, keepdim=True)
    # points_anchor = points_anchor - offset

    # # random_se3() seems to have little effect in rotation about the z axis, so do this implicitly
    # axis_angle = torch.tile(torch.tensor([0, 0, 1], device=device, dtype=torch.float), (N, 1))
    # rot_ratio = torch.rand((N,1))*(2*np.pi) / \
    #     torch.norm(axis_angle, dim=1).max().item()
    # constrained_axis_angle = rot_ratio*axis_angle  # max angle is rot_var
    # R = Rotate(axis_angle_to_matrix(constrained_axis_angle))#.translate(torch.zeros(N, 3, device=device))

    # T = random_se3(N, rot_var=rot_var, trans_var=trans_var, device=device, fix_random=False)

    # return T.transform_points(R.transform_points(points_anchor)) + offset

    # Assume the random_se3() function is uniform random in SE(3) now..
    T = random_se3(
        N,
        rot_var=rot_var,
        trans_var=trans_var,
        device=device,
        rot_sample_method=rot_sample_method,
    )
    return T.transform_points(points_anchor)


def get_nonintersecting_rack(
    points_anchor_base,
    rect_prisms_base,
    rot_var=np.pi,
    trans_var=1,
    return_debug=False,
    rot_sample_method="axis_angle",
):
    points_anchor1 = points_anchor_base
    rect_prisms1 = rect_prisms_base

    success = torch.tensor(
        [False] * points_anchor1.shape[0], device=points_anchor1.device
    )
    tries = 0

    points_anchor2 = torch.empty_like(points_anchor1)

    # Empirically, the success rate for any single augmentation for this env is 1/1.3778 = approx 72.5%
    # Success rate for batch size 16 is 0.725^16 = 0.58%
    while not torch.all(success):  # and tries < 10:
        points_anchor2_temp = random_transform_rack(
            points_anchor_base,
            rot_var=rot_var,
            trans_var=trans_var,
            rot_sample_method=rot_sample_method,
        )
        rect_prisms2 = points_to_axis_aligned_rect(points_anchor2_temp)

        intersects_temp = axis_aligned_rect_intersect(rect_prisms1, rect_prisms2)

        points_anchor2[intersects_temp == False] = points_anchor2_temp[
            intersects_temp == False
        ]
        success = torch.logical_or(success, torch.logical_not(intersects_temp))
        tries += 1

    if return_debug:
        rect_prisms2 = points_to_axis_aligned_rect(points_anchor2)
        debug = dict(tries=tries, rect_prisms1=rect_prisms1, rect_prisms2=rect_prisms2)
    else:
        debug = {}

    return points_anchor2, debug


def get_random_distractor_demo(
    points_gripper,
    points_action,
    points_anchor_base,
    rot_var=np.pi,
    trans_var=1,
    transform_base=True,
    return_debug=False,
    rot_sample_method="axis_angle",
):
    debug = {}
    if transform_base:
        N = points_anchor_base.shape[0]
        T = random_se3(
            N,
            rot_var=rot_var,
            trans_var=trans_var,
            device=points_anchor_base.device,
            fix_random=False,
        )

        points_anchor_base = T.transform_points(points_anchor_base)
        points_action = T.transform_points(points_action)
        if points_gripper is not None:
            points_gripper = T.transform_points(points_gripper)
        debug["transform_base_T"] = T
    else:
        debug["transform_base_T"] = None

    points_anchor1 = points_anchor_base

    rect_prisms_base = combine_axis_aligned_rect(
        [
            points_to_axis_aligned_rect(points_anchor_base),
            points_to_axis_aligned_rect(points_action),
        ]
        + (
            [
                points_to_axis_aligned_rect(points_gripper),
            ]
            if points_gripper is not None
            else []
        )
    )
    points_anchor2, debug_temp = get_nonintersecting_rack(
        points_anchor_base,
        rect_prisms_base,
        rot_var=rot_var,
        trans_var=trans_var,
        return_debug=return_debug,
        rot_sample_method=rot_sample_method,
    )
    debug.update(debug_temp)

    if return_debug:
        # If rotate_base=True, these axis aligned rects don't exactly match the intersection checks that were done during
        # the non-intersecting rack generation because a rotation was applied afterwards
        # However, it's a good visualization
        rect_prisms1 = combine_axis_aligned_rect(
            [
                points_to_axis_aligned_rect(points_anchor1),
                points_to_axis_aligned_rect(points_action),
            ]
            + (
                [
                    points_to_axis_aligned_rect(points_gripper),
                ]
                if points_gripper is not None
                else []
            )
        )
        rect_prisms2 = points_to_axis_aligned_rect(points_anchor2)
        debug_temp = dict(
            tries=debug["tries"], rect_prisms1=rect_prisms1, rect_prisms2=rect_prisms2
        )
        debug.update(debug_temp)

    return points_gripper, points_action, points_anchor1, points_anchor2, debug


# Perturbations for the mug on rack environment
class MugOnRackPerturbs:
    RACK_CFG = {
        "RACK_CENTER_X": 0.59,
        "RACK_CENTER_Y": 0.26,
        "RACK_R": (0.61 - 0.56) / 2,
        "ARM_R": (0.6 - 0.58) / 2,
        "ARM_LEN": 0.26 - 0.147 - (0.61 - 0.56) / 2,
        "ARM_Z_BOTTOM": 1.1,
        "ARM_Z_TOP": 1.24,
    }

    @staticmethod
    def duplicate_k_rack_table(
        mug_demo_pts: torch.Tensor,
        rack_demo_pts: torch.Tensor,
        k_range=[2, 2],
        rot_var=np.pi / 180 * 360,
        trans_var=0.15,
    ):
        """
        These racks will all lie along the same xy plane (same height)

        Getting rid of the downsampling. We aren't training things with this :C batch size = 1


        Arguments:
        ==========
        mug_demo_pts: point cloud for the mug while in the demo pose. Shape (B, num_points, 3)
        rack_demo_pts: point cloud for the rack while in the demo pose. One rack only. Shape (B, num_points, 3)
        k_range: number of racks in the output, inclusive
        rot_var: variance for the rotation perturbation
        trans_var: variance for the translation perturbation

        Returns:
        =========
        new_mug_pts: point cloud for the mug on one of the k racks. Shape ((num_points_in_rack+num_points_in_mug)//(k+1), 3)
        all_rack_pts: point clouds for all the racks. Shape ((num_points_in_rack+num_points_in_mug)//(k+1) * k, 3)
        all_transforms: list of transforms for each rack. Length k
        """

        B = mug_demo_pts.shape[0]

        k = torch.randint(k_range[1] + 1 - k_range[0], (1,)) + k_range[0]

        N = int(points_per_cloud[0].item() * ks[0].item())

        all_rack_pts = torch.empty((B, N, 3))

        all_rack_transforms = random_se3(
            ks.sum(), rot_var=rot_var, trans_var=trans_var, device=mug_demo_pts.device
        )

        all_rack_pts = []
        all_mug_pts = []
        index = 0
        all_rack_pts = torch.empty((B, N, 3))
        all_mug_pts = torch.empty((B, mug_demo_pts.shape[1], 3))
        for i, k in enumerate(ks):
            my_transforms = all_rack_transforms[index : index + k]
            # Create the racks
            all_rack_pts[i] = my_transforms.transform_points(rack_demo_pts[i]).reshape(
                -1, 3
            )

            # TODO maybe set a minimum distance between racks so that they dont collide

            # Put the mug on one of the racks
            rack_i = torch.randint(k, (1,)).item()
            all_mug_pts[i] = my_transforms[rack_i].transform_points(
                mug_demo_pts[i].float()
            )

            index += k

        return all_mug_pts, all_rack_pts, all_rack_transforms

    @staticmethod
    def duplicate_k_rack(
        mug_demo_pts: torch.Tensor,
        rack_demo_pts: torch.Tensor,
        k_range=[2, 2],
        rot_var=np.pi / 180 * 360,
        trans_var=0.15,
    ):
        """
        Arguments:
        ==========
        mug_demo_pts: point cloud for the mug while in the demo pose. Shape (B, num_points, 3)
        rack_demo_pts: point cloud for the rack while in the demo pose. One rack only. Shape (B, num_points, 3)
        k_range: number of racks in the output, inclusive
        rot_var: variance for the rotation perturbation
        trans_var: variance for the translation perturbation

        Returns:
        =========
        new_mug_pts: point cloud for the mug on one of the k racks. Shape ((num_points_in_rack+num_points_in_mug)//(k+1), 3)
        all_rack_pts: point clouds for all the racks. Shape ((num_points_in_rack+num_points_in_mug)//(k+1) * k, 3)
        all_transforms: list of transforms for each rack. Length k
        """

        B = mug_demo_pts.shape[0]

        ks = torch.randint(k_range[1] + 1 - k_range[0], (B,)) + k_range[0]

        # Downsample rack
        total_points = rack_demo_pts.shape[1]

        def reduce(A, op):
            r = op.identity  # op = ufunc
            for i in range(len(A)):
                r = op(r, A[i])
                return r

        lcm = torch.tensor(1)
        for i in range(len(ks)):
            lcm = torch.lcm(lcm, ks[i])
        points_per_cloud = total_points // lcm * (lcm / ks)

        # TODO PUT THIS RANDOMNESS BACK
        random_start_point = False
        # sample_farthest_points doesn't do well if there need to be more points than there are in the original point cloud
        # TODO maybe make the rack point cloud bigger so that this isn't an issue

        rack_demo_pts, ids = sample_farthest_points(
            rack_demo_pts, K=points_per_cloud, random_start_point=random_start_point
        )

        N = int(points_per_cloud[0].item() * ks[0].item())

        all_rack_pts = torch.empty((B, N, 3))

        all_rack_transforms = random_se3(
            ks.sum(), rot_var=rot_var, trans_var=trans_var, device=mug_demo_pts.device
        )

        all_rack_pts = []
        all_mug_pts = []
        index = 0
        all_rack_pts = torch.empty((B, N, 3))
        all_mug_pts = torch.empty((B, mug_demo_pts.shape[1], 3))
        for i, k in enumerate(ks):
            my_transforms = all_rack_transforms[index : index + k]
            # Create the racks
            all_rack_pts[i] = my_transforms.transform_points(rack_demo_pts[i]).reshape(
                -1, 3
            )

            # TODO maybe set a minimum distance between racks so that they dont collide

            # Put the mug on one of the racks
            rack_i = torch.randint(k, (1,)).item()
            all_mug_pts[i] = my_transforms[rack_i].transform_points(
                mug_demo_pts[i].float()
            )

            index += k

        return all_mug_pts, all_rack_pts, all_rack_transforms

    @staticmethod
    def duplicate_rack(
        data,
        translation=np.array([-0.3, 0, 0]),
        change_mug_target=False,
        max_points=float("inf"),
        datatype="points_only",
    ):
        # rand_mug_target: if True, randomly place the mug on either the left or right rack
        # translation is the translation of rack 2 relative to the original rack

        if datatype == "points_only":
            pass
        elif datatype == "data_dict":
            data = {k: data[k] for k in data.files}  # convert to dict

            rack_cloud = data["clouds"][data["classes"] == 1]
            rack_cloud = rack_cloud + translation

            rack_classes = np.tile(
                data["classes"][data["classes"] == 1][0], (len(rack_cloud))
            )

            rack_colors = data["colors"][data["classes"] == 1]

            data["clouds"] = np.concatenate([data["clouds"], rack_cloud], axis=0)
            data["classes"] = np.concatenate([data["classes"], rack_classes], axis=0)
            data["colors"] = np.concatenate([data["colors"], rack_cloud], axis=0)

            if change_mug_target:
                # Put the mug on the new rack
                data["clouds"][data["classes"] == 0] = (
                    data["clouds"][data["classes"] == 0] + translation
                )
            if len(data["clouds"]) > max_points:
                idxs = np.random.choice(
                    np.arange(0, len(data["clouds"])), size=(max_points,)
                )
                data["clouds"] = data["clouds"][idxs]
                data["classes"] = data["classes"][idxs]
                data["colors"] = data["colors"][idxs]

            return data

    @staticmethod
    def duplicate_arm(data, rack_cfg=RACK_CFG, change_mug_target=False):
        data = {k: data[k] for k in data.files}  # convert to dict

        THETA = np.pi / 2
        TRANSLATE = np.array([0, 0, 0.1])
        rotation_mat = np.array(
            [
                [np.cos(THETA), -np.sin(THETA), 0],
                [np.sin(THETA), np.cos(THETA), 0],
                [0, 0, 1],
            ]
        )
        center = [
            RACK_CFG["RACK_CENTER_X"],
            RACK_CFG["RACK_CENTER_Y"],
            RACK_CFG["ARM_Z_BOTTOM"],
        ]

        rack_arm_idxs = MugOnRackPerturbs.get_rack_arm_idxs(data, rack_cfg=rack_cfg)
        rack_arm_cloud = data["clouds"][rack_arm_idxs]
        rack_arm_cloud = (
            (rotation_mat @ ((rack_arm_cloud - center).T)).T + center + TRANSLATE
        )

        rack_arm_classes = np.tile(1, (len(rack_arm_idxs)))

        rack_arm_colors = data["colors"][rack_arm_idxs]

        data["clouds"] = np.concatenate([data["clouds"], rack_arm_cloud], axis=0)
        data["classes"] = np.concatenate([data["classes"], rack_arm_classes], axis=0)
        data["colors"] = np.concatenate([data["colors"], rack_arm_cloud], axis=0)

        if change_mug_target:
            # Put the mug on the new rack
            data["clouds"][data["classes"] == 0] = (
                (rotation_mat @ ((data["clouds"][data["classes"] == 0] - center).T)).T
                + center
                + TRANSLATE
            )

        return data

    @staticmethod
    def get_rack_arm_idxs(data, rack_cfg=RACK_CFG):
        RACK_CENTER_X = rack_cfg["RACK_CENTER_X"]
        RACK_CENTER_Y = rack_cfg["RACK_CENTER_Y"]
        RACK_R = rack_cfg["RACK_R"]
        ARM_R = rack_cfg["ARM_R"]
        ARM_LEN = rack_cfg["ARM_LEN"]
        ARM_Z_BOTTOM = rack_cfg["ARM_Z_BOTTOM"]
        ARM_Z_TOP = rack_cfg["ARM_Z_TOP"]

        rack_cloud = data["clouds"][data["classes"] == 1]
        rack_classes = np.tile(
            data["classes"][data["classes"] == 1][0], (len(rack_cloud))
        )
        rack_colors = data["colors"][data["classes"] == 1]

        rack_arm_idxs = np.where(
            np.logical_and.reduce(
                [
                    rack_cloud[:, 0] < RACK_CENTER_X + ARM_R,
                    rack_cloud[:, 0] > RACK_CENTER_X - ARM_R,
                    rack_cloud[:, 1] < RACK_CENTER_Y - RACK_R + 0.01,
                    rack_cloud[:, 1] > RACK_CENTER_Y - RACK_R - ARM_LEN,
                    rack_cloud[:, 2] < ARM_Z_TOP,
                    rack_cloud[:, 2] > ARM_Z_BOTTOM,
                ]
            )
        )

        rack_idxs = np.where(data["classes"] == 1)[0]

        arm_idxs = rack_idxs[rack_arm_idxs]

        return arm_idxs
