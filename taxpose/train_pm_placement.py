import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytorch3d
import torch
import torch_geometric.loader as tgl
import typer
import wandb
from tqdm import tqdm

from taxpose.datasets.pm_placement import (
    CATEGORIES,
    DatasetSplit,
    DatasetType,
    PlaceDataset,
    create_goal_inference_dataset,
)
from taxpose.models.taxpose import BrianChuerLoss, SE3LossTheirs
from taxpose.models.taxpose import TAXPoseModel as Model

app = typer.Typer()


def theta_err(R_pred: torch.FloatTensor, R_gt: torch.FloatTensor):
    # Normalize. This prevents NaNs.
    def normalize(x):
        return x / torch.norm(x, dim=-1, keepdim=True)

    R_pred = normalize(R_pred)
    dR = R_pred.transpose(-2, -1) @ R_gt

    # Batched trace. Ugly.
    # In future versions of torch, we can just vmap.
    t_dR = torch.diagonal(dR, offset=0, dim1=-2, dim2=-1).sum(-1)

    th = torch.arccos((torch.clamp(t_dR, -1, 3) - 1) / 2.0)

    if torch.isnan(th).any():
        raise ValueError("NaN in theta error")

    return torch.rad2deg(th)


def t_err(t_pred, t_gt):
    return torch.norm(t_pred - t_gt, dim=-1)


def classwise_mean(
    results: Dict[str, List[Dict[str, float]]]
) -> Dict[str, Dict[str, float]]:
    classwise: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for k, ds in results.items():
        for d in ds:
            for m, v in d.items():
                classwise[m][CATEGORIES[k]].append(v)

    return {k: {m: np.mean(v) for m, v in d.items()} for k, d in classwise.items()}  # type: ignore


def global_mean(results: Dict[str, List[Dict[str, float]]]) -> Dict[str, float]:
    metricswise = defaultdict(list)
    for ds in results.values():
        for d in ds:
            for m, v in d.items():
                metricswise[m].append(v)
    return {m: np.mean(v) for m, v in metricswise.items()}  # type: ignore


def class_weighted_mean(
    classwise_means: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    return {k: np.mean(list(v.values())) for k, v in classwise_means.items()}  # type: ignore


def global_mean_to_pandas(means: Dict[str, float], methodname="ours") -> pd.DataFrame:
    return pd.DataFrame.from_dict(means, orient="index", columns=[methodname]).T


def classwise_mean_to_pandas(
    classwise_means: Dict[str, Dict[str, float]], methodname="ours"
) -> pd.DataFrame:
    return {
        k: pd.DataFrame.from_dict(v, orient="index", columns=[methodname]).T
        for k, v in classwise_means.items()
    }


@torch.no_grad()
def run_eval(
    model, dset: PlaceDataset, batch_size: int, device: str, num_workers: int = 0
) -> Tuple[
    Dict[str, float],
    Dict[str, Dict[str, float]],
    Dict[str, List[Dict[str, float]]],
]:
    """Run evaluation on a dataset.

    Args:
        model: Model to predict relative poses.
        dset (PlaceDataset): Dataset to evaluate on.
        batch_size (int): Batch size.
        device (str): Device.
        num_workers (int, optional): Loader workers. Defaults to 0.

    Returns:
        Tuple[Dict[str, float], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
            global metrics, classwise metrics, per-object metrics.
    """
    loader = tgl.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    obj_metrics: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    for action, anchor in tqdm(loader):
        action = action.to(device)
        anchor = anchor.to(device)

        R_gt = action.R_action_anchor.reshape(-1, 3, 3)
        t_gt = action.t_action_anchor

        R_pred, t_pred, aws, _, _, _ = model(action, anchor)

        R_errs = theta_err(R_pred, R_gt)
        t_errs = t_err(t_pred, t_gt)

        # Aggregate results.
        for j in range(len(anchor)):
            obj_metrics[anchor.id[j]].append(
                {
                    "R_err": R_errs[j].item(),
                    "t_err": t_errs[j].item(),
                }
            )

    class_means = classwise_mean(obj_metrics)
    global_means = global_mean(obj_metrics)
    return global_means, class_means, obj_metrics


@app.command()
def train(
    dataset: DatasetType = typer.Option(...),
    arch="brianchuer-gc",
    root="/home/beisner/datasets/partnet-mobility",
    batch_size: int = 16,
    use_bc_loss: bool = True,
    n_epochs: int = 50,
    lr: float = 0.0001,
    n_repeat: int = 50,
    embedding_dim: int = 512,
    n_workers: int = 30,
    n_proc_per_worker: int = 2,
):
    torch.autograd.set_detect_anomaly(True)

    device = "cuda:0"

    # Get the dataset.
    train_dset = create_goal_inference_dataset(
        pm_root=root,
        dataset=dataset,
        split=DatasetSplit.TRAIN,
        randomize_camera=True,
        rotate_anchor=True,
        snap_to_surface=True,
        full_obj=True,
        even_downsample=True,
        n_repeat=n_repeat,
        n_workers=n_workers,
        n_proc_per_worker=n_proc_per_worker,
        seed=123456,
    )

    # Two smaller datasets for validation.
    eval_train_dset = create_goal_inference_dataset(
        pm_root=root,
        dataset=dataset,
        split=DatasetSplit.TRAIN,
        randomize_camera=True,
        rotate_anchor=True,
        snap_to_surface=True,
        full_obj=True,
        even_downsample=True,
        n_repeat=1,
        n_workers=n_workers,
        n_proc_per_worker=n_proc_per_worker,
        seed=123456,
    )
    eval_test_dset = create_goal_inference_dataset(
        pm_root=root,
        dataset=dataset,
        split=DatasetSplit.TEST,
        randomize_camera=True,
        rotate_anchor=True,
        snap_to_surface=True,
        full_obj=True,
        even_downsample=True,
        n_repeat=1,
        n_workers=n_workers,
        n_proc_per_worker=n_proc_per_worker,
        seed=654321,
    )

    train_loader = tgl.DataLoader(
        train_dset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    model = Model(arch, attn=True, embedding_dim=embedding_dim).to(device)

    # opt = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    if use_bc_loss:
        crit = BrianChuerLoss()
    else:
        crit = SE3LossTheirs()  # type: ignore

    d = f"checkpoints/manual/{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}"
    os.makedirs(d, exist_ok=True)

    wandb.init(
        project="taxpose",
        entity="r-pad",
        config={"dataset": dataset, "arch": arch, "lr": lr, "batch_size": batch_size},
    )

    for i in range(1, n_epochs + 1):
        # Run the evals.
        model.eval()
        for name, dset in [("train", eval_train_dset), ("test", eval_test_dset)]:
            global_means, class_means, obj_metrics = run_eval(
                model, dset, batch_size, device
            )
            wandb.log(
                {f"{name}/{k}": v for k, v in global_means.items()},
                commit=False,
            )

        pbar = tqdm(train_loader)
        model.train()
        for action, anchor in pbar:
            action = action.to(device)
            anchor = anchor.to(device)

            opt.zero_grad()

            R_gt = action.R_action_anchor.reshape(-1, 3, 3)
            t_gt = action.t_action_anchor
            mat = torch.zeros(len(action), 4, 4).to(device)
            mat[:, :3, :3] = R_gt
            mat[:, :3, 3] = t_gt
            mat[:, 3, 3] = 1

            R_pred, t_pred, _, pred_T_action, Fx, Fy = model(action, anchor)

            gt_T_action = pytorch3d.transforms.Transform3d(
                device=device, matrix=mat.transpose(-1, -2)
            )

            if use_bc_loss:
                loss = crit(
                    action.pos.reshape(len(action), -1, 3),
                    pred_T_action,
                    gt_T_action,
                    Fx,
                )
            else:
                loss, R_loss, t_loss = crit(R_pred, R_gt, t_pred, t_gt)

            loss.backward()
            opt.step()
            wandb.log({"loss": loss.item()})

            if use_bc_loss:
                desc = f"Epoch {i:03d}:  Loss:{loss.item():.3f}"

            else:
                desc = f"Epoch {i:03d}:  Loss:{loss.item():.3f}, R_loss:{R_loss.item():.3f}, t_loss:{t_loss.item():.3f}"

            pbar.set_description(desc)

        if dataset != "single":
            torch.save(model.state_dict(), os.path.join(d, f"weights_{i:03d}.pt"))


@app.command()
def evaluate(
    dataset: DatasetType = typer.Option(...),
    split: DatasetSplit = typer.Option(...),
    weights: str = typer.Option(...),
    arch="brianchuer-gc",
    root=os.path.expanduser("~/datasets/partnet-mobility"),
    batch_size: int = 32,  # Higher because we're not doing backprop.
    embedding_dim: int = 512,
    n_repeat: int = 50,
    results_dir: str = "results",
):
    device = "cuda:0"

    results_dir = os.path.join(results_dir, dataset.value, split.value, arch)
    os.makedirs(results_dir, exist_ok=True)

    # Load the model.
    model = Model(arch, attn=True, embedding_dim=embedding_dim).to(device)
    model.load_state_dict(torch.load(weights))
    model.eval()

    # Get the dataset.
    dset = create_goal_inference_dataset(
        pm_root=root,
        dataset=dataset,
        split=split,
        randomize_camera=True,
        rotate_anchor=True,
        snap_to_surface=True,
        full_obj=True,
        even_downsample=True,
        n_repeat=n_repeat,
        n_workers=30,
        n_proc_per_worker=2,
        seed=123456 if split == "train" else 654321,
    )

    # Run the evaluation.
    global_means, class_means, obj_metrics = run_eval(model, dset, batch_size, device)

    # Print the results.
    print("Global metrics: - PER OBJECT")
    gm = global_mean_to_pandas(global_means)
    gm.to_csv(os.path.join(results_dir, "global_metrics.csv"))
    print(gm)

    print("Global metrics: - CLASS-WEIGHTED")
    gm_cw = global_mean_to_pandas(class_weighted_mean(class_means))
    gm_cw.to_csv(os.path.join(results_dir, "global_metrics_classweighted.csv"))
    print(gm_cw)

    print("Classwise metrics:")
    for k, v in classwise_mean_to_pandas(class_means).items():
        v.to_csv(os.path.join(results_dir, f"classwise_{k}.csv"))
        print(k)
        print(v)


if __name__ == "__main__":
    app()
