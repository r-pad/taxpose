import numpy as np
import torch
from colour import Color

# color scheme from https://personal.sron.nl/~pault/
color_scheme = {}
color_scheme["blue"] = np.array([68, 119, 170])
color_scheme["cyan"] = np.array([102, 204, 238])
color_scheme["green"] = np.array([34, 136, 51])
color_scheme["yellow"] = np.array([204, 187, 68])
color_scheme["red"] = np.array([238, 102, 119])
color_scheme["purple"] = np.array([170, 51, 119])
color_scheme["orange"] = np.array([238, 119, 51])


def get_color(tensor_list, color_list, axis=False):
    """
    @param tensor_list: list of tensors of shape (num_points, 3)
    @param color_list: list of strings of color names that should be within color_scheme.keys(), eg, 'red', 'blue'
    @return points_color_stacked: numpy array of shape (num_points*len(tensor_list), 6)
    """

    stacked_list = []
    assert len(tensor_list) == len(
        color_list
    ), "len(tensor_list) should match len(color_list)"
    for i in range(len(tensor_list)):
        tensor = tensor_list[i].detach().cpu().numpy()
        color = color_list[i]
        assert len(tensor.shape) == 2, "tensor should be of shape of (num_points, 3)"
        assert (
            tensor.shape[-1] == 3
        ), "tensor.shape[-1] should be of shape 3, for point coordinates"
        assert (
            color in color_scheme.keys()
        ), "passed in color {} is not in the available color scheme, go to utils/color_utils.py to add any".format(
            color
        )

        color_tensor = torch.from_numpy(color_scheme[color]).unsqueeze(0)  # 1,3
        N = tensor.shape[0]
        color_tensor = (
            torch.repeat_interleave(color_tensor, N, dim=0).detach().cpu().numpy()
        )  # N,3

        points_color = np.concatenate((tensor, color_tensor), axis=-1)  # num_points, 6

        stacked_list.append(points_color)
    points_color_stacked = np.concatenate(
        stacked_list, axis=0
    )  # num_points*len(tensor_list), 6
    if axis:
        axis_pts = create_axis()
        points_color_stacked = np.concatenate([points_color_stacked, axis_pts], axis=0)

    return points_color_stacked


def get_color_sym_dist(tensor_list, color_list):
    """
    @param tensor_list: list of tensors of shape (num_points, 3)
    @param color_list: list of strings of color names that should be within color_scheme.keys(), eg, 'red', 'blue'
    @return points_color_stacked: numpy array of shape (num_points*len(tensor_list), 6)
    """

    stacked_list = []
    assert len(tensor_list) == len(
        color_list
    ), "len(tensor_list) should match len(color_list)"

    for i in range(len(tensor_list)):
        tensor = tensor_list[i].detach().cpu().numpy()
        if color_list[i] == None:
            color = color_scheme["blue"]
            color_tensor = torch.from_numpy(color).unsqueeze(0)  # 1,3
            N = tensor.shape[0]
            color_tensor = (
                torch.repeat_interleave(color_tensor, N, dim=0).detach().cpu().numpy()
            )  # N,3

        else:
            color_tensor = color_list[i]
        assert len(tensor.shape) == 2, "tensor should be of shape of (num_points, 3)"
        assert (
            tensor.shape[-1] == 3
        ), "tensor.shape[-1] should be of shape 3, for point coordinates"
        assert (
            len(color_tensor.shape) == 2
        ), "color_tensor should be of shape of (num_points, 3)"
        assert (
            color_tensor.shape[-1] == 3
        ), "color_tensor.shape[-1] should be of shape 3, for point coordinates"

        points_color = np.concatenate((tensor, color_tensor), axis=-1)  # num_points, 6

        stacked_list.append(points_color)

    points_color_stacked = np.concatenate(
        stacked_list, axis=0
    )  # num_points*len(tensor_list), 6

    return points_color_stacked


def create_axis(length=1.0, num_points=100):
    pts = np.linspace(0, length, num_points)
    x_axis_pts = np.stack([pts, np.zeros(num_points), np.zeros(num_points)], axis=1)
    y_axis_pts = np.stack([np.zeros(num_points), pts, np.zeros(num_points)], axis=1)
    z_axis_pts = np.stack([np.zeros(num_points), np.zeros(num_points), pts], axis=1)

    x_axis_clr = np.tile([255, 0, 0], [num_points, 1])
    y_axis_clr = np.tile([0, 255, 0], [num_points, 1])
    z_axis_clr = np.tile([0, 0, 255], [num_points, 1])

    x_axis = np.concatenate([x_axis_pts, x_axis_clr], axis=1)
    y_axis = np.concatenate([y_axis_pts, y_axis_clr], axis=1)
    z_axis = np.concatenate([z_axis_pts, z_axis_clr], axis=1)

    pts = np.concatenate([x_axis, y_axis, z_axis], axis=0)

    return pts


NUM_GRADIENT_COLORS = 100
gradient_colors = np.array(
    [c.get_rgb() for c in Color("blue").range_to(Color("green"), NUM_GRADIENT_COLORS)]
)


def color_gradient(vals, min_step=1e-6):
    vals = vals.detach().cpu()
    color_idxs = (
        (vals - vals.min(axis=-1, keepdim=True)[0])
        / max(
            min_step,
            vals.max(axis=-1, keepdim=True)[0]
            - vals.min(axis=-1, keepdim=True)[0]
            + 1e-6,
        )
        * NUM_GRADIENT_COLORS
    ).int()
    return (
        np.array(
            [
                gradient_colors[color_idxs_batch]
                for color_idxs_batch in color_idxs.numpy()
            ]
        )
        * 255
    )
