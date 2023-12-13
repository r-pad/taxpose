import logging
import os

import hydra
import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

from taxpose.datasets.point_cloud_data_module import MultiviewDataModule
from taxpose.nets.transformer_flow import ResidualFlow_DiffEmbTransformer
from taxpose.training.flow_equivariance_training_module_nocentering import (
    EquivarianceTrainingModule,
)
from taxpose.utils.load_model import get_weights_path


def load_emb_weights(checkpoint_reference, wandb_cfg=None, run=None):
    if checkpoint_reference.startswith(wandb_cfg.entity):
        artifact_dir = os.path.join(wandb_cfg.artifact_dir, checkpoint_reference)
        artifact = run.use_artifact(checkpoint_reference)
        checkpoint_path = artifact.get_path("model.ckpt").download(root=artifact_dir)
        weights = torch.load(checkpoint_path)["state_dict"]
        # remove "model.emb_nn" prefix from keys
        weights = {k.replace("model.emb_nn.", ""): v for k, v in weights.items()}
        return weights
    else:
        return torch.load(hydra.utils.to_absolute_path(checkpoint_reference))[
            "embnn_state_dict"
        ]


def maybe_load_from_wandb(checkpoint_reference, wandb_cfg, run):
    if checkpoint_reference.startswith(wandb_cfg.entity):
        # download checkpoint locally (if not already cached)
        artifact_dir = os.path.join(wandb_cfg.artifact_dir, checkpoint_reference)
        artifact = run.use_artifact(checkpoint_reference)
        ckpt_file = artifact.get_path("model.ckpt").download(root=artifact_dir)
    else:
        ckpt_file = checkpoint_reference

    return ckpt_file


@torch.no_grad()
@hydra.main(config_path="../configs", config_name="eval_metrics")
def main(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    pl.seed_everything(cfg.seed)

    dm = MultiviewDataModule(
        batch_size=cfg.inference.batch_size,
        num_workers=cfg.resources.num_workers,
        cfg=cfg.dm,
    )

    dm.setup(stage=cfg.split)

    network = ResidualFlow_DiffEmbTransformer(
        emb_dims=cfg.model.emb_dims,
        emb_nn=cfg.model.emb_nn,
        return_flow_component=cfg.model.return_flow_component,
        center_feature=cfg.model.center_feature,
        pred_weight=cfg.model.pred_weight,
        multilaterate=cfg.model.multilaterate,
        sample=cfg.model.mlat_sample,
        mlat_nkps=cfg.model.mlat_nkps,
        break_symmetry=cfg.model.break_symmetry,
    )

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
        # ckpt_file = maybe_load_from_wandb(cfg.checkpoint, cfg.wandb, run)
        ckpt_file = get_weights_path(cfg.checkpoint, cfg.wandb, run)
        try:
            weights = torch.load(ckpt_file)["state_dict"]
            model.load_state_dict(weights)
        except RuntimeError:
            # This is an "older" style model, so we need to load the weights
            # manually.
            model.model.load_state_dict(weights)

        logging.info(f"Loaded checkpoint from {ckpt_file}")

    model.cuda()
    model.eval()

    if cfg.split == "train":
        loader = dm.train_dataloader()
    elif cfg.split == "val":
        loader = dm.val_dataloader()
    elif cfg.split == "test":
        loader = dm.test_dataloader()

    dm.setup(stage="train")
    train_dataloader = dm.train_dataloader()
    dm.setup(stage="val")
    val_dataloader = dm.val_dataloader()

    for name, loader in zip(["train", "val"], [train_dataloader, val_dataloader]):
        # for name, loader in zip(["val"], [val_dataloader]):
        metrics = []

        for batch in tqdm(loader):
            points_anchor = batch["points_anchor"].cuda()
            points_action = batch["points_action"].cuda()
            points_action_trans = batch["points_action_trans"].cuda()
            points_anchor_trans = batch["points_anchor_trans"].cuda()
            action_symmetry_features = batch["action_symmetry_features"].cuda()
            anchor_symmetry_features = batch["anchor_symmetry_features"].cuda()
            action_symmetry_rgb = batch["action_symmetry_rgb"].cuda()
            anchor_symmetry_rgb = batch["anchor_symmetry_rgb"].cuda()

            res = model(
                points_action_trans,
                points_anchor_trans,
                action_symmetry_features,
                anchor_symmetry_features,
            )

            if "sampled_ixs_action" in res:
                ixs_action = res["sampled_ixs_action"]
                points_action = torch.take_along_dim(
                    points_action, ixs_action.unsqueeze(-1), dim=1
                )
                action_symmetry_rgb = torch.take_along_dim(
                    action_symmetry_rgb, ixs_action.unsqueeze(-1), dim=1
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

            # Evaluate the model on the prediction.

            # fig = segmentation_fig(
            #     torch.cat(
            #         [
            #             points_action_trans[0].cpu(),
            #             points_anchor_trans[0].cpu(),
            #             res["pred_points_action"][0].cpu(),
            #         ],
            #         dim=0,
            #     ).numpy(),
            #     torch.cat(
            #         [
            #             torch.ones(points_action_trans.shape[1]),
            #             2 * torch.ones(points_anchor_trans.shape[1]),
            #             3 * torch.ones(res["pred_points_action"].shape[1]),
            #         ],
            #         dim=0,
            #     )
            #     .int()
            #     .numpy(),
            #     labelmap={1: "mug", 2: "rack", 3: "pred"},
            # )
            # fig.show()
            # fig = pointcloud_fig(
            #     np.concatenate(
            #         [
            #             points_action[0].cpu(),
            #             points_anchor[0].cpu(),
            #         ]
            #     ),
            #     downsample=1,
            #     colors=np.concatenate(
            #         [
            #             action_symmetry_rgb[0].cpu().numpy(),
            #             anchor_symmetry_rgb[0].cpu().numpy(),
            #         ]
            #     ),
            # )
            # fig.show()

            # fig = pointcloud_fig(
            #     np.concatenate(
            #         [
            #             res["pred_points_action"][0].cpu(),
            #             points_anchor_trans[0].cpu(),
            #         ]
            #     ),
            #     downsample=1,
            #     colors=np.concatenate(
            #         [
            #             action_symmetry_rgb[0].cpu().numpy(),
            #             anchor_symmetry_rgb[0].cpu().numpy(),
            #         ]
            #     ),
            # )
            # fig.show()

        metrics = {
            k: np.concatenate([m[k] for m in metrics]) for k in metrics[0].keys()
        }
        print("Results for {}".format(name))
        print("Mean angle error: {}".format(metrics["angle_err"].mean()))
        print("Mean translation error: {}".format(metrics["t_err"].mean()))

        # Save the metrics to wandb as a table.
        ang_err_mean = metrics["angle_err"].mean()
        t_err_mean = metrics["t_err"].mean()

        metrics_table = wandb.Table(
            columns=["angle_err", "t_err"],
            data=[
                [ang_err_mean, t_err_mean],
            ],
        )

        wandb.log({f"{name}_metrics": metrics_table})


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
