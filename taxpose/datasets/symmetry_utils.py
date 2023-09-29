from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

from taxpose.datasets.ndf import ObjectClass


def scalars_to_rgb(symmetry_labels: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
    # A function which converts a scalar to an RGB color which varies between red (min) and blue (max).

    # Convert the symmetry labels to a color.
    color = np.zeros((symmetry_labels.shape[0], 3), dtype=np.float32)
    color[:, 0] = (symmetry_labels - np.min(symmetry_labels)) / (
        np.max(symmetry_labels) - np.min(symmetry_labels) + 1e-6
    )
    color[:, 2] = 1 - color[:, 0]
    color[:, 1] = 0.5

    # Convert the color to RGB.
    color = color * 255
    color = color.round().astype(np.uint8)

    return color


def gripper_symmetry_labels(
    gripper_pcd: npt.NDArray[np.float32],
) -> Tuple[np.float32, np.float32, np.float32]:
    """Compute gripper symmetry labels.

    Args:
        gripper_pcd (np.NDArray): An N x 3 array of points.

    Returns:
        Tuple[np.float32, np.float32, np.float32]: A tuple containing the symmetry labels, principal axis, and centroid.
        Shapes:
            symmetry_labels: (N, 1)
            principal_axis: (3,)
            centroid: (3,)
    """

    # TODO: since the gripper has some symmetry, we should be able to use this to improve the symmetry labels for the
    # gripper. The gripper has discrete symmetry.

    # Compute the centroid.
    centroid = np.mean(gripper_pcd, axis=0)

    # Do PCA.
    cov = np.cov(gripper_pcd.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    eigvecs = eigvecs.T

    # Find the principal axis.
    principal_axis = eigvecs[np.argmax(eigvals)]

    # Compute the normalized vector from each point to the centroid.
    v_gripper = gripper_pcd - centroid
    v_gripper = v_gripper / np.linalg.norm(v_gripper, axis=1, keepdims=True)

    # Compute the dot product of each vector with the principal axis.
    l_gripper = np.dot(v_gripper, principal_axis)

    # Normalize.
    l_gripper = l_gripper / np.max(np.abs(l_gripper))

    return l_gripper[..., None], principal_axis, centroid


def rotational_symmetry_labels(
    obj_pcd: npt.NDArray[np.float32],
    obj_class: ObjectClass,
    look_at: Optional[npt.NDArray[np.float32]] = None,
    seed: Optional[int] = None,
) -> Tuple[np.float32, np.float32, np.float32, np.float32]:
    """Computes object symmetry labels.

    Args:
        obj_pcd (npt.NDArray[np.float32]): Object point cloud. Shape: (N, 3)
        obj_class (ObjectClass): The object class. Determines which axis is the axis of rotational symmetry.
        look_at (npt.NDArray[np.float32]): A point in 3D space to look at to break symmetry.
            Shape: (3,)
        Returns:
            Tuple[np.float32, np.float32, np.float32]: A tuple containing the symmetry labels, principal axis, secondary axis, and centroid.
            Shapes:
                symmetry_labels: (N, 1)
                principal_axis: (3,)
                s_obj: (3,)
                centroid: (3,)
    """
    # Compute the centroid.
    centroid = np.mean(obj_pcd, axis=0)

    # Centered object.
    v_obj = obj_pcd - centroid

    # Do PCA.
    cov = np.cov(obj_pcd.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    eigvecs = eigvecs.T

    # Find the principal axis. If the object is a bottle, it's the largest principal axis. If it's a bowl, it's the
    # smallest principal axis.
    if obj_class == ObjectClass.BOTTLE:
        principal_axis = eigvecs[np.argmax(eigvals)]
    elif obj_class == ObjectClass.BOWL:
        principal_axis = eigvecs[np.argmin(eigvals)]
    else:
        raise ValueError(f"Symmetry unknown for object class: {obj_class}")

    # If look_at is not provided, error out.
    if look_at is None:
        # Choose a random direction perpendicular to the principal axis.
        rng = np.random.default_rng(seed)
        heading = rng.normal(size=3)
        heading = heading - np.dot(heading, principal_axis) * principal_axis
        heading = heading / np.linalg.norm(heading)
    else:
        # Compute a vector to the look_at point.
        heading = look_at - centroid
        heading = heading / np.linalg.norm(heading)

    # Compute the secondary axis.
    if obj_class == ObjectClass.BOTTLE:
        # See definition of s_bottle in the paper (equation 25 in the appendix).
        # This should separate the bottle into left and right sides.
        s_obj = np.cross(principal_axis, heading)
        s_obj = s_obj / np.linalg.norm(s_obj)
    elif obj_class == ObjectClass.BOWL:
        # See definition of s_bowl in the paper (equation 26 in the appendix).
        # This should separate the bowl into front and back sides.
        s_obj = heading - np.dot(heading, principal_axis) * principal_axis
        s_obj = s_obj / np.linalg.norm(s_obj)
    else:
        raise ValueError(f"Symmetry unknown for object class: {obj_class}")

    # Compute the dot product of each centered point with s_obj.
    l_obj = np.dot(v_obj, s_obj)

    # Normalize.
    l_obj = l_obj / np.max(np.abs(l_obj))

    return l_obj[..., None], principal_axis, s_obj, centroid


def nonsymmetric_labels(obj_pcd) -> Tuple[np.float32, np.float32]:
    """Computes nonsymmetric labels. This should just be ones.

    Args:
        obj_pcd (npt.NDArray[np.float32]): Object point cloud. Shape: (N, 3)

    Returns:
        Tuple[np.float32, np.float32]: A tuple containing the symmetry labels and centroid.
            Shapes:
                symmetry_labels: (N, 1)
                centroid: (3,)

    """
    # Compute the centroid.
    centroid = np.mean(obj_pcd, axis=0)

    l_obj = np.ones((obj_pcd.shape[0], 1), dtype=np.float32)

    return l_obj, centroid
