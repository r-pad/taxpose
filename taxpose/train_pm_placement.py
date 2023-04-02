import os
from datetime import datetime

import pytorch3d
import torch
import torch_geometric.loader as tgl
import typer
from rpad.pyg.dataset import CachedByKeyDataset
from tqdm import tqdm

from taxpose.datasets.pm_placement import (
    GoalInferenceDataset,
    PlaceDataset,
    default_scenes,
    scenes_by_location,
)
from taxpose.models.taxpose import BrianChuerLoss, SE3LossTheirs
from taxpose.models.taxpose import TAXPoseModel as Model


def create_datasets(
    root: str,
    dataset: str,
    randomize_camera: bool = True,
    rotate_anchor: bool = True,
    snap_to_surface: bool = True,
    full_obj: bool = True,
    even_downsample: bool = True,
    n_repeat: int = 50,
    n_workers: int = 30,
    n_proc_per_worker: int = 2,
):
    if dataset == "single":
        obs_train_scene_ids = [("11299", "ell", "0", "in")]
        goal_train_scene_ids = [("11299", "ell", "0", "in")]
        obs_test_scene_ids = [("11299", "ell", "0", "in")]
        goal_test_scene_ids = [("11299", "ell", "0", "in")]
    elif dataset == "dishwasher_top":
        obs_train_scene_ids = default_scenes("train", "obs", "dishwasher", "3")
        goal_train_scene_ids = default_scenes("train", "goal", "dishwasher", "3")
        obs_test_scene_ids = default_scenes("test", "obs", "dishwasher", "3")
        goal_test_scene_ids = default_scenes("test", "goal", "dishwasher", "3")

        obs_train_scene_ids = [(t[0], t[1], t[2], "top") for t in obs_train_scene_ids]
        goal_train_scene_ids = [(t[0], t[1], t[2], "top") for t in goal_train_scene_ids]
        obs_test_scene_ids = [(t[0], t[1], t[2], "top") for t in obs_test_scene_ids]
        goal_test_scene_ids = [(t[0], t[1], t[2], "top") for t in goal_test_scene_ids]
    elif dataset == "dishwasher_in":
        obs_train_scene_ids = default_scenes("train", "obs", "dishwasher", "0")
        goal_train_scene_ids = default_scenes("train", "goal", "dishwasher", "0")
        obs_test_scene_ids = default_scenes("test", "obs", "dishwasher", "0")
        goal_test_scene_ids = default_scenes("test", "goal", "dishwasher", "0")

        obs_train_scene_ids = [(t[0], t[1], t[2], "in") for t in obs_train_scene_ids]
        goal_train_scene_ids = [(t[0], t[1], t[2], "in") for t in goal_train_scene_ids]
        obs_test_scene_ids = [(t[0], t[1], t[2], "in") for t in obs_test_scene_ids]
        goal_test_scene_ids = [(t[0], t[1], t[2], "in") for t in goal_test_scene_ids]
    elif dataset in {"in", "top", "left", "right", "under"}:
        obs_train_scene_ids = scenes_by_location("train", "obs", dataset)
        goal_train_scene_ids = scenes_by_location("train", "goal", dataset)
        obs_test_scene_ids = scenes_by_location("test", "obs", dataset)
        goal_test_scene_ids = scenes_by_location("test", "goal", dataset)

        obs_train_scene_ids = [(t[0], t[1], t[2], dataset) for t in obs_train_scene_ids]
        goal_train_scene_ids = [
            (t[0], t[1], t[2], dataset) for t in goal_train_scene_ids
        ]
        obs_test_scene_ids = [(t[0], t[1], t[2], dataset) for t in obs_test_scene_ids]
        goal_test_scene_ids = [(t[0], t[1], t[2], dataset) for t in goal_test_scene_ids]
    elif dataset == "gc":
        obs_train_scene_ids = []
        goal_train_scene_ids = []
        obs_test_scene_ids = []
        goal_test_scene_ids = []
        for loc in {"in", "top", "left", "right", "under"}:
            otr = scenes_by_location("train", "obs", loc)
            gtr = scenes_by_location("train", "goal", loc)
            ote = scenes_by_location("test", "obs", loc)
            gte = scenes_by_location("test", "goal", loc)

            obs_train_scene_ids.extend([(t[0], t[1], t[2], loc) for t in otr])
            goal_train_scene_ids.extend([(t[0], t[1], t[2], loc) for t in gtr])
            obs_test_scene_ids.extend([(t[0], t[1], t[2], loc) for t in ote])
            goal_test_scene_ids.extend([(t[0], t[1], t[2], loc) for t in gte])

    else:
        raise ValueError("bad dataset name")

    # A helper to create individual datasets, since many of the
    # arguments are the same.
    def _create_place_dset(mode, ids, seed):
        dset = CachedByKeyDataset(
            dset_cls=PlaceDataset,
            dset_kwargs={
                "root": root,
                "randomize_camera": randomize_camera,
                "snap_to_surface": snap_to_surface,
                "full_obj": full_obj,
                "even_downsample": even_downsample,
                "rotate_anchor": rotate_anchor,
                "scene_ids": ids,
                "mode": mode,
            },
            data_keys=ids,
            root=root,
            processed_dirname=PlaceDataset.processed_dir_name(
                mode,
                randomize_camera,
                snap_to_surface,
                full_obj,
                even_downsample,
            ),
            n_repeat=n_repeat,
            n_workers=n_workers,
            n_proc_per_worker=n_proc_per_worker,
            seed=seed,
        )

        return dset

    # The goal transfer dataset is a combination of two place datasets.
    train_dset = GoalInferenceDataset(
        obs_dset=_create_place_dset("obs", obs_train_scene_ids, 123456),
        goal_dset=_create_place_dset("goal", goal_train_scene_ids, 654321),
        rotate_anchor=rotate_anchor,
    )
    test_dset = GoalInferenceDataset(
        obs_dset=_create_place_dset("obs", obs_test_scene_ids, 123456),
        goal_dset=_create_place_dset("goal", goal_test_scene_ids, 654321),
        rotate_anchor=rotate_anchor,
    )

    return train_dset, test_dset


def main(
    dataset: str,
    arch="brianchuer",
    root="/home/beisner/datasets/partnet-mobility",
    batch_size: int = 16,
    use_bc_loss: bool = True,
    n_epochs: int = 50,
    lr: float = 0.001,
    n_repeat: int = 50,
    embedding_dim: int = 512,
    n_workers: int = 30,
    n_proc_per_worker: int = 2,
):
    torch.autograd.set_detect_anomaly(True)
    n_print = 1

    device = "cuda:0"

    train_dset, test_dset = create_datasets(
        root=root,
        dataset=dataset,
        randomize_camera=True,
        rotate_anchor=True,
        snap_to_surface=True,
        full_obj=True,
        even_downsample=True,
        n_repeat=n_repeat,
        n_workers=n_workers,
        n_proc_per_worker=n_proc_per_worker,
    )

    train_dset[0]

    train_loader = tgl.DataLoader(
        train_dset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    model = Model(arch, attn=True, embedding_dim=embedding_dim).to(device)

    # opt = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    if use_bc_loss:
        crit = BrianChuerLoss()
    else:
        crit = SE3LossTheirs()

    d = f"checkpoints/manual/{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}"
    os.makedirs(d, exist_ok=True)

    for i in range(1, n_epochs + 1):
        pbar = tqdm(train_loader)
        for action, anchor, _, _ in pbar:
            action = action.to(device)
            anchor = anchor.to(device)

            opt.zero_grad()

            R_gt = action.R_action_anchor.reshape(-1, 3, 3)
            t_gt = action.t_action_anchor
            mat = torch.zeros(action.num_graphs, 4, 4).to(device)
            mat[:, :3, :3] = R_gt
            mat[:, :3, 3] = t_gt
            mat[:, 3, 3] = 1

            R_pred, t_pred, _, pred_T_action, Fx, Fy = model(action, anchor)

            gt_T_action = pytorch3d.transforms.Transform3d(
                device=device, matrix=mat.transpose(-1, -2)
            )

            # T_action_anchor_gt = (
            #     pytorch3d.transforms.Transform3d().rotate(R_gt).translate(t_gt)
            # ).to(device)

            if use_bc_loss:
                loss = crit(
                    action.pos.reshape(action.num_graphs, -1, 3),
                    pred_T_action,
                    gt_T_action,
                    Fx,
                )
            else:
                loss, R_loss, t_loss = crit(R_pred, R_gt, t_pred, t_gt)

            loss.backward()
            opt.step()

            if i % n_print == 0:
                if use_bc_loss:
                    desc = f"Epoch {i:03d}:  Loss:{loss.item():.3f}"

                else:
                    desc = f"Epoch {i:03d}:  Loss:{loss.item():.3f}, R_loss:{R_loss.item():.3f}, t_loss:{t_loss.item():.3f}"

                pbar.set_description(desc)
                # print(
                #     f"{i}\tLoss:{loss.item()}\tR_loss:{R_loss.item()}\tt_loss:{t_loss.item()}"
                # )
        if dataset != "single":
            torch.save(model.state_dict(), os.path.join(d, f"weights_{i:03d}.pt"))

    # dcp_sg_plot(action.pos, anchor.pos, t_gt, t_pred, R_gt, R_pred, None).show()


if __name__ == "__main__":
    typer.run(main)
