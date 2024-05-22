import numpy as np
import torch


def sample_uniform_random(radius, num_points):
    points_x = np.random.uniform(-radius, radius, num_points)
    points_y = np.random.uniform(-radius, radius, num_points)
    points_z = np.random.uniform(-radius, radius, num_points)
    point_coords = np.stack([points_x, points_y, points_z], axis=1)

    return point_coords


def sample_uniform_cube(radius, num_dim_points):
    points_x = np.linspace(-radius, radius, num_dim_points)
    points_y = np.linspace(-radius, radius, num_dim_points)
    points_z = np.linspace(-radius, radius, num_dim_points)
    point_coords = np.meshgrid(points_x, points_y, points_z)
    point_coords = np.stack(
        [
            point_coords[0].flatten(),
            point_coords[1].flatten(),
            point_coords[2].flatten(),
        ],
        axis=1,
    )

    return point_coords


def sample_closest_pairs(action_points, anchor_points, top_k=50):
    """
    Identify top_k closest pairs between action_points and anchor_points for each batch.

    Args:
        action_points (torch.Tensor): Point cloud of the action object [B, N, D], where D = 3 (3D coordinates).
        anchor_points (torch.Tensor): Point cloud of the anchor object [B, M, D], where D = 3 (3D coordinates).
        top_k (int): Number of closest pairs to identify.

    Returns:
        A tuple of (action_indices, anchor_indices, distances), where each is a tensor shaped [B, top_k] containing
        indices or distances of the top_k closest pairs for each batch.
    """
    # Ensure action_points and anchor_points are float tensors for computation
    action_points = action_points.float()
    anchor_points = anchor_points.float()

    # Compute squared distances
    squared_distances = (
        (action_points[:, :, None, :3] - anchor_points[:, None, :, :3]) ** 2
    ).sum(dim=-1)

    # Find top_k closest pairs for each batch
    top_k_distances, flat_indices = torch.topk(
        squared_distances.view(squared_distances.shape[0], -1),
        top_k,
        dim=-1,
        largest=False,
        sorted=True,
    )

    # Convert flat indices to pair indices
    num_anchor_points = anchor_points.shape[1]
    top_k_action_indices = flat_indices // num_anchor_points
    top_k_anchor_indices = flat_indices % num_anchor_points

    return top_k_action_indices, top_k_anchor_indices, torch.sqrt(top_k_distances)


def sample_random_pair(top_k_action_indices, top_k_anchor_indices, top_k_distances):
    """
    Randomly sample one pair out of the top K closest pairs for each batch

    Args:
        top_k_action_indices (torch.Tensor): Indices of action points in top K pairs, shape [B, K].
        top_k_anchor_indices (torch.Tensor): Indices of anchor points in top K pairs, shape [B, K].
        top_k_distances (torch.Tensor): Distances of top K pairs, shape [B, K].

    Returns:
        A tuple containing indices of the randomly selected action and anchor points, and their distances.
    """
    batch_size, top_k = top_k_action_indices.shape
    selected_indices = torch.randint(0, top_k, (batch_size,), dtype=torch.long)

    selected_action_indices = top_k_action_indices[
        torch.arange(batch_size), selected_indices
    ]
    selected_anchor_indices = top_k_anchor_indices[
        torch.arange(batch_size), selected_indices
    ]
    selected_distances = top_k_distances[torch.arange(batch_size), selected_indices]

    return selected_action_indices, selected_anchor_indices, selected_distances
