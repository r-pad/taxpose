import logging

import hydra
import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import OmegaConf
from rpad.visualize_3d.plots import pointcloud_fig, segmentation_fig
from tqdm import tqdm

from taxpose.datasets.point_cloud_data_module import MultiviewDataModule
from taxpose.nets.transformer_flow import create_network
from taxpose.training.flow_equivariance_training_module_nocentering import (
    EquivarianceTrainingModule,
)
from taxpose.utils.load_model import get_weights_path


def visualize_preds(
    points_action_trans,
    points_anchor_trans,
    res,
    points_action,
    points_anchor,
    action_symmetry_rgb,
    anchor_symmetry_rgb,
):
    # Evaluate the model on the prediction.

    fig = segmentation_fig(
        torch.cat(
            [
                points_action_trans[0].cpu(),
                points_anchor_trans[0].cpu(),
                res["pred_points_action"][0].cpu(),
            ],
            dim=0,
        ).numpy(),
        torch.cat(
            [
                torch.ones(points_action_trans.shape[1]),
                2 * torch.ones(points_anchor_trans.shape[1]),
                3 * torch.ones(res["pred_points_action"].shape[1]),
            ],
            dim=0,
        )
        .int()
        .numpy(),
        labelmap={1: "mug", 2: "rack", 3: "pred"},
    )
    fig.show()
    fig = pointcloud_fig(
        np.concatenate(
            [
                points_action[0].cpu(),
                points_anchor[0].cpu(),
            ]
        ),
        downsample=1,
        colors=np.concatenate(
            [
                action_symmetry_rgb[0].cpu().numpy(),
                anchor_symmetry_rgb[0].cpu().numpy(),
            ]
        ),
    )
    fig.show()

    fig = pointcloud_fig(
        np.concatenate(
            [
                res["pred_points_action"][0].cpu(),
                points_anchor_trans[0].cpu(),
            ]
        ),
        downsample=1,
        colors=np.concatenate(
            [
                action_symmetry_rgb[0].cpu().numpy(),
                anchor_symmetry_rgb[0].cpu().numpy(),
            ]
        ),
    )
    fig.show()


@torch.no_grad()
@hydra.main(config_path="../configs", config_name="eval_metrics", version_base="1.1")
def main(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    pl.seed_everything(cfg.seed)

    dm = MultiviewDataModule(
        batch_size=cfg.inference.batch_size,
        num_workers=cfg.resources.num_workers,
        cfg=cfg.dm,
    )

    network = create_network(cfg.model)

    model = EquivarianceTrainingModule(
        model=network,
        weight_normalize=cfg.task.phase.weight_normalize,
        softmax_temperature=cfg.task.phase.softmax_temperature,
        sigmoid_on=cfg.inference.sigmoid_on,
    )

    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        group=cfg.wandb.group,
        job_type=cfg.job_type,
        save_code=True,
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
    )

    if cfg.checkpoint is not None:
        ckpt_file = get_weights_path(cfg.checkpoint, cfg.wandb, run)
        weights = torch.load(ckpt_file)["state_dict"]
        model.load_state_dict(weights)
        logging.info(f"Loaded checkpoint from {ckpt_file}")

    model.cuda()
    model.eval()

    # model = torch.compile(model)

    print("setting up train")
    dm.setup(stage="train")
    train_dataloader = dm.train_dataloader()
    print("setting up val")
    dm.setup(stage="val")
    val_dataloader = dm.val_dataloader()

    csv_strs = [cfg.task.phase.name]

    for name, loader in zip(["train", "val"], [train_dataloader, val_dataloader]):
        # for name, loader in zip(["val"], [val_dataloader]):
        metrics = []

        for batch in tqdm(loader):
            points_anchor = batch["points_anchor"].cuda()
            points_action = batch["points_action"].cuda()
            points_action_trans = batch["points_action_trans"].cuda()
            points_anchor_trans = batch["points_anchor_trans"].cuda()
            action_features = (
                batch["action_features"].cuda() if "action_features" in batch else None
            )
            anchor_features = (
                batch["anchor_features"].cuda() if "anchor_features" in batch else None
            )
            phase_onehot = (
                batch["phase_onehot"].cuda() if "phase_onehot" in batch else None
            )

            res = model(
                points_action_trans,
                points_anchor_trans,
                action_features,
                anchor_features,
                phase_onehot,
            )

            if "sampled_ixs_action" in res:
                ixs_action = res["sampled_ixs_action"]
                points_action = torch.take_along_dim(
                    points_action, ixs_action.unsqueeze(-1), dim=1
                )

            # Apply the transform to the original transform.
            T0 = batch["T0"].transpose(1, 2)
            T1 = batch["T1"].transpose(1, 2)

            T_gt = torch.matmul(T1, torch.inverse(T0))

            R_gt = T_gt[:, :3, :3].double()
            t_gt = T_gt[:, :3, 3]

            T_pred = res["pred_T_action"].get_matrix().cpu().transpose(1, 2)

            R_pred = T_pred[:, :3, :3].double()
            t_pred = T_pred[:, :3, 3]

            # Compute the angular error between transforms

            trace = torch.diagonal(
                torch.matmul(R_gt.transpose(1, 2), R_pred), offset=0, dim1=-1, dim2=-2
            ).sum(-1)
            cos_angle = (trace - 1) / 2
            angle = torch.acos(torch.clamp(cos_angle, -1, 1))
            angle_deg = angle * 180 / np.pi

            # Normalize so that the angle is between 0 and 180 degrees
            angle_deg = torch.min(angle_deg, 180 - angle_deg)

            # Compute the translation error between transforms
            # t_err = torch.norm(t_gt - t_pred, dim=1)

            # Actually, let's try the distance between the centroids of the clouds.
            R_T1, t_T1 = T1[:, :3, :3], T1[:, :3, 3]
            gt_action_pts = torch.matmul(
                R_T1, batch["points_action"].transpose(1, 2)
            ) + t_T1.unsqueeze(-1)

            pred_action_pts = torch.matmul(
                R_pred.float(), batch["points_action_trans"].transpose(1, 2)
            ) + t_pred.unsqueeze(-1)

            gt_action_centroid = gt_action_pts.mean(dim=2)
            pred_action_centroid = pred_action_pts.mean(dim=2)

            t_err = torch.norm(gt_action_centroid - pred_action_centroid, dim=1)

            metrics.append(
                {
                    "angle_err": angle_deg.cpu().numpy(),
                    "t_err": t_err.cpu().numpy(),
                }
            )

            fig = segmentation_fig(
                torch.cat(
                    [
                        points_action[0].cpu(),
                        points_anchor[0].cpu(),
                    ],
                    dim=0,
                ).numpy(),
                torch.cat(
                    [
                        torch.ones(points_action.shape[1]),
                        2 * torch.ones(points_anchor.shape[1]),
                    ],
                    dim=0,
                )
                .int()
                .numpy(),
                labelmap={1: "demo_action", 2: "demo_anchor"},
            )
            fig.show()

            # Prediction.
            fig = segmentation_fig(
                torch.cat(
                    [
                        points_action_trans[0].cpu(),
                        points_anchor_trans[0].cpu(),
                        res["pred_points_action"][0].cpu(),
                    ],
                    dim=0,
                ).numpy(),
                torch.cat(
                    [
                        torch.ones(points_action_trans.shape[1]),
                        2 * torch.ones(points_anchor_trans.shape[1]),
                        3 * torch.ones(res["pred_points_action"].shape[1]),
                    ],
                    dim=0,
                )
                .int()
                .numpy(),
                labelmap={1: "action", 2: "anchor", 3: "pred"},
            )
            fig.show()
            breakpoint()

        metrics = {
            k: np.concatenate([m[k] for m in metrics]) for k in metrics[0].keys()
        }
        print("Results for {}".format(name))
        print("Mean angle error: {}".format(metrics["angle_err"].mean()))
        print("Mean translation error: {}".format(metrics["t_err"].mean()))

        # Save the metrics to wandb as a table.
        ang_err_mean = metrics["angle_err"].mean()
        t_err_mean = metrics["t_err"].mean()

        csv_strs.append(f"{ang_err_mean},{t_err_mean}")

        metrics_table = wandb.Table(
            columns=["angle_err", "t_err"],
            data=[
                [ang_err_mean, t_err_mean],
            ],
        )

        wandb.log({f"{name}_metrics": metrics_table})

    print("CSV string: ", ",".join(csv_strs))


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
