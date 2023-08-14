import numpy as np
import open3d as o3d
import torch

from taxpose.utils.color_utils import get_color_sym_dist


def to_np(tensor):
    return tensor.detach().cpu().numpy()


def to_torch(numpy, device_tensor=None):
    tensor = torch.from_numpy(numpy).double()
    if device_tensor != None:
        tensor = tensor.to(device_tensor.device)
    return tensor


def vec_norm(vec):
    """
    vec: (3) torch tensor
    """
    return vec / torch.norm(vec)


def dot_product(x, y):
    """
    x: (3) torch tensor
    y: (3) torch tensor
    """
    assert len(x.shape) == 1, "x should be of shape (3), but got (.., ..)"
    assert len(y.shape) == 1, "y should be of shape (3), but got (.., ..)"
    # norm vec
    return vec_norm(x).T @ vec_norm(y)


def project_to_axis(vector, axis):
    """
    vector: (3)
    axis: 3, can be not unit-normed
    """
    # make axis unit normed
    axis = vec_norm(axis)
    vector = vector.double()
    projected_vec = (vector @ axis) * axis
    return projected_vec


def project_to_xy(vector):
    """
    vector: num_poins, 3
    """
    if len(vector.shape) > 1:
        vector[:, -1] = 0
    elif len(vector.shape) == 1:
        vector[-1] = 0
    return vector


def get_sym_label(
    action_cloud, anchor_cloud, action_class, anchor_class, discrete=True
):
    assert 0 in [
        action_class,
        anchor_class,
    ], "class 0 must be here somewhere as the manipulation object of interest"
    if action_class == 0:
        sym_breaking_class = action_class
        center_class = anchor_class
        points_sym = action_cloud[0]
        points_nonsym = anchor_cloud[0]
    elif anchor_class == 0:
        sym_breaking_class = anchor_class
        center_class = action_class
        points_sym = anchor_cloud[0]
        points_nonsym = action_cloud[0]

    non_sym_center = points_nonsym.mean(axis=0)
    sym_center = points_sym.mean(axis=0)
    sym2nonsym = non_sym_center - sym_center
    sym2nonsym = project_to_xy(sym2nonsym)

    sym_vec = points_sym - sym_center
    sym_vec = project_to_xy(sym_vec)
    if discrete:
        sym_cls = torch.sign(torch.matmul(sym_vec, sym2nonsym)).unsqueeze(
            0
        )  # num_points, 1

    return sym_cls


def color_dist_symmetric_plane_red_blue(cts_cls, points_sym):
    """
    Return the color of each point measured by distance to symmetric plane,
    such that max positive is red and max negative is blue, near zero division line is black.

    @param cts_cls: distance to symmetric normal per point, (num_points)
    @param points_sym: shape of the color_cts which should be, (num_points,3)
    """
    # coloring cts_cls for vis (+: red, -: blue)
    color = cts_cls / torch.abs(torch.max(cts_cls)) * 255
    color_cts = torch.zeros(points_sym.shape).to(points_sym.device).double()
    color_cts[cts_cls >= 0, 0] = color[cts_cls >= 0]
    color_cts[cts_cls < 0, 2] = torch.abs(color[cts_cls < 0])
    color_cts[cts_cls >= 0]
    return color_cts


def color_dist_symmetric_plane_red_green(cts_cls, points_sym):
    """
    Return the color of each point measured by distance to symmetric plane,
    such that max positive is red and max negative is green, near zero division line is yellow.

    color scheme (+: red, -: green, 0: yellow )
    method ref from: https://stackoverflow.com/questions/6394304/algorithm-how-do-i-fade-from-red-to-green-via-yellow-using-rgb-values

    @param cts_cls: distance to symmetric normal per point, (num_points)
    @param points_sym: shape of the color_cts which should be, (num_points,3)
    """
    color = (cts_cls / torch.abs(torch.max(cts_cls)) * (255 / 2)) + (255 / 2)
    color_cts = torch.zeros(points_sym.shape).to(points_sym.device).double()
    color_cts[:, 0] = torch.minimum(
        torch.ones(color.shape).to(color.device) * 255, color * 2
    )
    color_cts[:, 1] = torch.minimum(
        torch.ones(color.shape).to(color.device) * 255, (255 - color) * 2
    )
    return color_cts


def get_sym_label_pca_grasp(
    action_cloud: torch.Tensor,
    anchor_cloud: torch.Tensor,
    action_class,
    anchor_class,
    object_type="bowl",
    color_scheme="red_green",
    normalize_dist=False,
):
    assert object_type in [
        "bowl",
        "bottle",
    ], "object_type should be either bowl or bottle for symmetry breaking"
    assert 0 in [
        action_class,
        anchor_class,
    ], "class 0 must be here somewhere as the manipulation object of interest"
    if action_class == 0:
        sym_breaking_class = action_class
        center_class = anchor_class
        points_sym = action_cloud[0]
        points_nonsym = anchor_cloud[0]
    elif anchor_class == 0:
        sym_breaking_class = anchor_class
        center_class = action_class
        points_sym = anchor_cloud[0]
        points_nonsym = action_cloud[0]

    non_sym_center = points_nonsym.mean(axis=0)
    points_sym_np = to_np(points_sym)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_sym_np)
    pcd_mean, pcd_cov = pcd.compute_mean_and_covariance()
    evals, evecs = np.linalg.eig(pcd_cov)
    evecs = np.transpose(evecs)
    major_axis = evecs[-1]
    if object_type == "bottle":
        major_axis = evecs[0]

    major_axis = to_torch(major_axis, device_tensor=non_sym_center)
    pcd_mean = to_torch(pcd_mean, device_tensor=non_sym_center)
    projected_point = (
        project_to_axis(vector=non_sym_center - pcd_mean, axis=major_axis) + pcd_mean
    )

    air_point = non_sym_center + (pcd_mean - projected_point)
    sym_vec = points_sym - pcd_mean
    sym2nonsym = air_point - pcd_mean

    points_nonsym_np = to_np(points_nonsym)
    pcd_nonsym = o3d.geometry.PointCloud()
    pcd_nonsym.points = o3d.utility.Vector3dVector(points_nonsym_np)
    pcd_nonsym_mean, pcd_nonsym_cov = pcd_nonsym.compute_mean_and_covariance()
    nonsym_evals, nonsym_evecs = np.linalg.eig(pcd_nonsym_cov)
    nonsym_evecs = np.transpose(nonsym_evecs)
    nonsym_major_axis = nonsym_evecs[0]
    nonsym_major_axis = to_torch(
        nonsym_major_axis, device_tensor=non_sym_center
    ).float()
    nonsym_vec = points_nonsym - non_sym_center

    cts_cls_nonsym = torch.matmul(nonsym_vec, nonsym_major_axis) / torch.norm(
        nonsym_major_axis
    )

    if object_type == "bottle":
        sym2nonsym = torch.cross(sym2nonsym, major_axis)
        if torch.matmul(sym2nonsym.float(), nonsym_major_axis) < 0:
            sym2nonsym *= -1

    sym_cls = torch.sign(torch.matmul(sym_vec, sym2nonsym))  # 1, num_points
    cts_cls = torch.matmul(sym_vec, sym2nonsym) / torch.norm(sym2nonsym)

    if color_scheme == "red_blue":
        color_cts = color_dist_symmetric_plane_red_blue(
            cts_cls, points_sym
        )  # (+: red, -: blue)
        color_cts_nonsym = color_dist_symmetric_plane_red_blue(
            cts_cls_nonsym, points_nonsym
        )  # (+: red, -: blue)
    else:
        color_cts = color_dist_symmetric_plane_red_green(
            cts_cls, points_sym
        )  # (+: red, -: green, 0: yellow )
        color_cts_nonsym = color_dist_symmetric_plane_red_green(
            cts_cls_nonsym, points_nonsym
        )  # (+: red, -: blue)
    fig = get_color_sym_dist([points_sym, points_nonsym], [color_cts, color_cts_nonsym])
    if normalize_dist:
        cts_cls = cts_cls / torch.max(torch.abs(cts_cls))
        cts_cls_nonsym = cts_cls_nonsym / torch.max(torch.abs(cts_cls_nonsym))
    # fig = plot_color([points_sym, points_nonsym], [color_cts, None])
    return_dict = {
        "fig": fig,
        "sym_cls": sym_cls.unsqueeze(0),
        "cts_cls": cts_cls.unsqueeze(0).float(),
        "cts_cls_nonsym": cts_cls_nonsym.unsqueeze(0).float(),
    }
    return return_dict


def get_sym_label_pca_place(
    action_cloud: torch.Tensor,
    anchor_cloud: torch.Tensor,
    action_class,
    anchor_class,
    color_scheme="red_green",
    normalize_dist=False,
):
    assert 0 in [
        action_class,
        anchor_class,
    ], "class 0 must be here somewhere as the manipulation object of interest"
    if action_class == 0:
        sym_breaking_class = action_class
        center_class = anchor_class
        points_sym = action_cloud[0]
        points_nonsym = anchor_cloud[0]
    elif anchor_class == 0:
        sym_breaking_class = anchor_class
        center_class = action_class
        points_sym = anchor_cloud[0]
        points_nonsym = action_cloud[0]

    non_sym_center = points_nonsym.mean(axis=0)
    points_sym_np = to_np(points_sym)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_sym_np)
    pcd_mean, pcd_cov = pcd.compute_mean_and_covariance()
    evals, evecs = np.linalg.eig(pcd_cov)
    evecs = np.transpose(evecs)
    major_axis = evecs[0]
    major_axis = to_torch(major_axis, device_tensor=non_sym_center)

    points_nonsym_np = to_np(points_nonsym)
    pcd_nonsym = o3d.geometry.PointCloud()
    pcd_nonsym.points = o3d.utility.Vector3dVector(points_nonsym_np)
    pcd_nonsym_mean, pcd_nonsym_cov = pcd_nonsym.compute_mean_and_covariance()
    nonsym_evals, nonsym_evecs = np.linalg.eig(pcd_nonsym_cov)
    nonsym_evecs = np.transpose(nonsym_evecs)
    nonsym_major_axis = nonsym_evecs[0]
    nonsym_major_axis = to_torch(nonsym_major_axis, device_tensor=non_sym_center)

    pcd_mean = to_torch(pcd_mean, device_tensor=non_sym_center)
    symmetry_plane_normal = nonsym_major_axis
    sym_vec = points_sym - pcd_mean

    sym_cls = torch.sign(torch.matmul(sym_vec, symmetry_plane_normal))  # 1, num_points
    cts_cls = torch.matmul(sym_vec, symmetry_plane_normal) / torch.norm(
        symmetry_plane_normal
    )

    if color_scheme == "red_blue":
        color_cts = color_dist_symmetric_plane_red_blue(
            cts_cls, points_sym
        )  # (+: red, -: blue)
    else:
        color_cts = color_dist_symmetric_plane_red_green(
            cts_cls, points_sym
        )  # (+: red, -: green, 0: yellow )
    fig = get_color_sym_dist([points_sym, points_nonsym], [color_cts, None])
    # fig = plot_color([points_sym, points_nonsym], [color_cts, None])
    if normalize_dist:
        cts_cls = cts_cls / torch.max(torch.abs(cts_cls))
    return_dict = {
        "fig": fig,
        "sym_cls": sym_cls.unsqueeze(0),
        "cts_cls": cts_cls.unsqueeze(0).float(),
    }
    return return_dict


def shift_z(cloud, z_shift, object_type="bowl"):
    """
    shift cloud by z_shift upwards along the object z-axis
    (for bowl, the z-axis open upwards towards the wider opening, for bottle, it points upward towards the bottle cap
    """
    points = cloud[0]
    points_center = points.mean(axis=0)
    points_np = to_np(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd_mean, pcd_cov = pcd.compute_mean_and_covariance()
    evals, evecs = np.linalg.eig(pcd_cov)
    evecs = np.transpose(evecs)
    if object_type == "bowl":
        major_axis = evecs[0]
        radial_pca = evecs[1]
    else:
        major_axis = evecs[0]
        radial_pca = evecs[1]
    major_axis = to_torch(major_axis, device_tensor=points_center)
    pcd_mean = to_torch(pcd_mean, device_tensor=points_center)

    sym_vec = points - pcd_mean
    radial_axis = torch.cross(
        major_axis, to_torch(radial_pca, device_tensor=major_axis)
    )
    # make sure the z-axis always point towards the opening of the bowl
    radial_dist = torch.matmul(sym_vec, radial_axis) / torch.norm(radial_axis)

    if object_type == "bowl":
        max_radial_idx = torch.argmax(radial_dist)
        max_radial_vec = sym_vec[max_radial_idx]
        sign = torch.sign(torch.matmul(max_radial_vec, major_axis))
    elif object_type == "bottle":
        vertial_dist = torch.matmul(sym_vec, major_axis) / torch.norm(major_axis)
        max_radial_idx = torch.argmax(vertial_dist)
        min_radial_idx = torch.argmin(vertial_dist)

        if torch.abs(radial_dist[max_radial_idx]) < torch.abs(
            radial_dist[min_radial_idx]
        ):
            sign = 1
        else:
            sign = -1
    if sign < 0:
        major_axis *= -1

    points += major_axis / torch.norm(major_axis) * z_shift

    return points.unsqueeze(0)


def shift_radial(points_action, points_anchor, ans_dict, radial_shift):
    pred_T_action = ans_dict["pred_T_action"]
    points_action_pred = pred_T_action.transform_points(points_action)

    gripper_pred_center = points_action_pred[0].mean(axis=0)  # 1,3
    anchor = points_anchor[0]
    anchor_center = anchor.mean(axis=0)
    points_sym_np = to_np(anchor)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_sym_np)
    pcd_mean, pcd_cov = pcd.compute_mean_and_covariance()
    evals, evecs = np.linalg.eig(pcd_cov)
    evecs = np.transpose(evecs)
    major_axis = evecs[0]
    major_axis = to_torch(major_axis, device_tensor=gripper_pred_center)

    bottle2gripper_center = gripper_pred_center - anchor_center
    projected_point_on_anchor_axis = (
        torch.matmul(bottle2gripper_center.float(), major_axis.float())
        / torch.norm(major_axis.float())
        * major_axis
        + anchor_center.float()
    )
    radial_vector_towards_gripper = gripper_pred_center - projected_point_on_anchor_axis
    points_anchor -= radial_vector_towards_gripper * radial_shift
    print("points_anchor.shape", points_anchor.shape)

    return points_anchor


def get_sym_label_pca_test(
    action_cloud,
    anchor_cloud,
    action_class,
    anchor_class,
    color_scheme="red_green",
    normalize_dist=True,
    object_type="bowl",
):
    assert 0 in [
        action_class,
        anchor_class,
    ], "class 0 must be here somewhere as the manipulation object of interest"
    if action_class == 0:
        sym_breaking_class = action_class
        center_class = anchor_class
        points_sym = action_cloud[0]
        points_nonsym = anchor_cloud[0]
    elif anchor_class == 0:
        sym_breaking_class = anchor_class
        center_class = action_class
        points_sym = anchor_cloud[0]
        points_nonsym = action_cloud[0]

    non_sym_center = points_nonsym.mean(axis=0)
    points_sym_np = to_np(points_sym)

    points_nonsym_np = to_np(points_nonsym)
    pcd_nonsym = o3d.geometry.PointCloud()
    pcd_nonsym.points = o3d.utility.Vector3dVector(points_nonsym_np)
    pcd_nonsym_mean, pcd_nonsym_cov = pcd_nonsym.compute_mean_and_covariance()
    nonsym_evals, nonsym_evecs = np.linalg.eig(pcd_nonsym_cov)
    nonsym_evecs = np.transpose(nonsym_evecs)
    nonsym_major_axis = nonsym_evecs[0]
    nonsym_major_axis = to_torch(
        nonsym_major_axis, device_tensor=non_sym_center
    ).float()
    nonsym_vec = points_nonsym - non_sym_center

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_sym_np)
    pcd_mean, pcd_cov = pcd.compute_mean_and_covariance()
    evals, evecs = np.linalg.eig(pcd_cov)
    evecs = np.transpose(evecs)
    major_axis = evecs[1]

    major_axis = to_torch(major_axis, device_tensor=non_sym_center)
    pcd_mean = to_torch(pcd_mean, device_tensor=non_sym_center)

    symmetry_plane_normal = major_axis

    sym_vec = points_sym - pcd_mean

    sym_cls = torch.sign(torch.matmul(sym_vec, symmetry_plane_normal))  # 1, num_points
    cts_cls = torch.matmul(sym_vec, symmetry_plane_normal) / torch.norm(
        symmetry_plane_normal
    )
    cts_cls_nonsym = torch.matmul(nonsym_vec, nonsym_major_axis) / torch.norm(
        nonsym_major_axis
    )
    if object_type == "bottle":
        if torch.matmul(symmetry_plane_normal.float(), nonsym_major_axis) < 0:
            symmetry_plane_normal *= -1

    if color_scheme == "red_blue":
        color_cts = color_dist_symmetric_plane_red_blue(
            cts_cls, points_sym
        )  # (+: red, -: blue)
    else:
        color_cts = color_dist_symmetric_plane_red_green(
            cts_cls, points_sym
        )  # (+: red, -: green, 0: yellow )

    # fig = plot_color([points_sym, points_nonsym], [color_cts, None])
    if normalize_dist:
        cts_cls = cts_cls / torch.max(torch.abs(cts_cls))
        cts_cls_nonsym = cts_cls_nonsym / torch.max(torch.abs(cts_cls_nonsym))
    return_dict = {
        "sym_cls": sym_cls.unsqueeze(0),
        "cts_cls": cts_cls.unsqueeze(0).float(),
        "cts_cls_nonsym": cts_cls_nonsym.unsqueeze(0).float(),
    }
    return return_dict


def get_sym_label_pca_test_bottle_graspable(
    rack_cloud,
    action_cloud,
    anchor_cloud,
    action_class,
    anchor_class,
    color_scheme="red_green",
    normalize_dist=True,
    object_type="bowl",
):
    assert 0 in [
        action_class,
        anchor_class,
    ], "class 0 must be here somewhere as the manipulation object of interest"
    if action_class == 0:
        sym_breaking_class = action_class
        center_class = anchor_class
        points_sym = action_cloud[0]
        points_nonsym = anchor_cloud[0]
    elif anchor_class == 0:
        sym_breaking_class = anchor_class
        center_class = action_class
        points_sym = anchor_cloud[0]
        points_nonsym = action_cloud[0]
    points_ref = rack_cloud[0]

    non_sym_center = points_nonsym.mean(axis=0)
    points_sym_np = to_np(points_sym)

    points_nonsym_np = to_np(points_nonsym)
    pcd_nonsym = o3d.geometry.PointCloud()
    pcd_nonsym.points = o3d.utility.Vector3dVector(points_nonsym_np)
    pcd_nonsym_mean, pcd_nonsym_cov = pcd_nonsym.compute_mean_and_covariance()
    nonsym_evals, nonsym_evecs = np.linalg.eig(pcd_nonsym_cov)
    nonsym_evecs = np.transpose(nonsym_evecs)
    nonsym_major_axis = nonsym_evecs[1]
    nonsym_major_axis = to_torch(
        nonsym_major_axis, device_tensor=non_sym_center
    ).float()
    nonsym_vec = points_nonsym - non_sym_center

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_sym_np)
    pcd_mean, pcd_cov = pcd.compute_mean_and_covariance()
    evals, evecs = np.linalg.eig(pcd_cov)
    evecs = np.transpose(evecs)
    major_axis = evecs[1]

    major_axis = to_torch(major_axis, device_tensor=non_sym_center)
    pcd_mean = to_torch(pcd_mean, device_tensor=non_sym_center)

    points_ref_center = points_ref.mean(axis=0)
    points_ref_np = to_np(points_ref)
    pcd_rack = o3d.geometry.PointCloud()
    pcd_rack.points = o3d.utility.Vector3dVector(points_ref_np)
    rack_pcd_mean, rack_pcd_cov = pcd_rack.compute_mean_and_covariance()
    rack_evals, rack_evecs = np.linalg.eig(rack_pcd_cov)
    rack_evecs = np.transpose(rack_evecs)
    rack_major_axis = rack_evecs[0]
    rack_major_axis = to_torch(rack_major_axis, device_tensor=non_sym_center)

    symmetry_plane_normal = rack_major_axis

    sym_vec = points_sym - pcd_mean

    sym_cls = torch.sign(torch.matmul(sym_vec, symmetry_plane_normal))  # 1, num_points
    cts_cls = torch.matmul(sym_vec, symmetry_plane_normal) / torch.norm(
        symmetry_plane_normal
    )
    cts_cls_nonsym = torch.matmul(nonsym_vec, nonsym_major_axis) / torch.norm(
        nonsym_major_axis
    )
    # if object_type == 'bottle':
    #     if torch.matmul(symmetry_plane_normal.float(), nonsym_major_axis) < 0:
    #         symmetry_plane_normal *= -1

    if color_scheme == "red_blue":
        color_cts = color_dist_symmetric_plane_red_blue(
            cts_cls, points_sym
        )  # (+: red, -: blue)
    else:
        color_cts = color_dist_symmetric_plane_red_green(
            cts_cls, points_sym
        )  # (+: red, -: green, 0: yellow )

    # fig = plot_color([points_sym, points_nonsym], [color_cts, None])
    if normalize_dist:
        cts_cls = cts_cls / torch.max(torch.abs(cts_cls))
        cts_cls_nonsym = cts_cls_nonsym / torch.max(torch.abs(cts_cls_nonsym))

    # plot_color([points_sym, points_nonsym, points_ref],
    #            [color_cts, None, None])
    return_dict = {
        "sym_cls": sym_cls.unsqueeze(0),
        "cts_cls": cts_cls.unsqueeze(0).float(),
        "cts_cls_nonsym": cts_cls_nonsym.unsqueeze(0).float(),
    }
    return return_dict
