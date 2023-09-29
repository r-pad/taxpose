import logging
import os

import hydra
import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import OmegaConf
from rpad.visualize_3d.plots import pointcloud_fig

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
@hydra.main(config_path="../configs", config_name="eval")
def main(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    pl.seed_everything(cfg.seed)

    dm = MultiviewDataModule(
        dataset_root=hydra.utils.to_absolute_path(cfg.train_data_dir),
        test_dataset_root=hydra.utils.to_absolute_path(cfg.test_data_dir),
        dataset_index=cfg.dataset_index,
        action_class=cfg.task.action_class,
        anchor_class=cfg.task.anchor_class,
        dataset_size=cfg.dataset_size,
        rotation_variance=np.pi / 180 * cfg.rotation_variance,
        translation_variance=cfg.translation_variance,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        cloud_type=cfg.task.cloud_type,
        num_points=cfg.num_points,
        overfit=cfg.overfit,
        num_demo=cfg.num_demo,
        cfg=cfg.dm,
    )

    dm.setup(stage=cfg.split)

    network = ResidualFlow_DiffEmbTransformer(
        emb_dims=cfg.emb_dims,
        emb_nn=cfg.emb_nn,
        return_flow_component=cfg.return_flow_component,
        center_feature=cfg.center_feature,
        pred_weight=cfg.pred_weight,
        multilaterate=cfg.multilaterate,
        sample=cfg.mlat_sample,
        mlat_nkps=cfg.mlat_nkps,
        break_symmetry=cfg.break_symmetry,
    )

    # reasoning_module = TAXPoseReasoning(
    #     network,
    #     TAXPoseReasoningConfig(
    #         loop=1,
    #         weight_normalize=cfg.task.weight_normalize,
    #         softmax_temperature=cfg.task.softmax_temperature,
    #     ),
    # )

    # model = TAXPoseInferenceModule(
    #     reasoning_module,
    #     symmetry_cfg=SymmetryConfig(
    #         action_class=cfg.task.action_class,
    #         anchor_class=cfg.task.anchor_class,
    #         object_type=cfg.object_class.name,
    #         normalize_dist=True,
    #         action=cfg.relationship.name,
    #     ),
    # )

    model = EquivarianceTrainingModule(
        model=network,
        weight_normalize=cfg.task.weight_normalize,
        softmax_temperature=cfg.task.softmax_temperature,
        sigmoid_on=cfg.sigmoid_on,
        flow_supervision="both",
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

    for batch in loader:
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

        breakpoint()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
