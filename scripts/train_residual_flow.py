import os

import hydra
import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from taxpose.datasets.point_cloud_data_module import MultiviewDataModule
from taxpose.nets.transformer_flow import ResidualFlow_DiffEmbTransformer
from taxpose.training.flow_equivariance_training_module_nocentering import (
    EquivarianceTrainingModule,
)


def write_to_file(file_name, string):
    with open(file_name, "a") as f:
        f.writelines(string)
        f.write("\n")
    f.close()


def load_emb_weights(checkpoint_reference, wandb_cfg=None, run=None):
    if checkpoint_reference.startswith(wandb_cfg.entity):
        artifact_dir = os.path.join(wandb_cfg.artifact_dir, checkpoint_reference)
        if run is None or not isinstance(run, wandb.sdk.wandb_run.Run):
            # Download without a run
            api = wandb.Api()
            artifact = api.artifact(checkpoint_reference, type="model")
        else:
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
        artifact_dir = wandb_cfg.artifact_dir
        artifact = run.use_artifact(checkpoint_reference)
        ckpt_file = artifact.get_path("model.ckpt").download(root=artifact_dir)
    else:
        ckpt_file = checkpoint_reference


@hydra.main(config_path="../configs", config_name="train_ndf")
def main(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    # breakpoint()

    # torch.set_float32_matmul_precision("medium")
    pl.seed_everything(cfg.seed)
    logger = WandbLogger(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        group=cfg.wandb.group,
        save_dir=cfg.wandb.save_dir,
        job_type=cfg.job_type,
        save_code=True,
        log_model=True,
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
    )
    # logger.log_hyperparams(cfg)
    # logger.log_hyperparams({"working_dir": os.getcwd()})
    trainer = pl.Trainer(
        logger=logger,
        accelerator="gpu",
        devices=[0],
        log_every_n_steps=cfg.training.log_every_n_steps,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        # reload_dataloaders_every_n_epochs=1,
        # callbacks=[SaverCallbackModel(), SaverCallbackEmbnnActionAnchor()],
        callbacks=[
            # This checkpoint callback saves the latest model during training, i.e. so we can resume if it crashes.
            # It saves everything, and you can load by referencing last.ckpt.
            ModelCheckpoint(
                dirpath=cfg.lightning.checkpoint_dir,
                filename="{epoch}-{step}",
                monitor="step",
                mode="max",
                save_weights_only=False,
                save_last=True,
                every_n_epochs=1,
            ),
            # This checkpoint will get saved to WandB. The Callback mechanism in lightning is poorly designed, so we have to put it last.
            ModelCheckpoint(
                dirpath=cfg.lightning.checkpoint_dir,
                filename="{epoch}-{step}-{train_loss:.2f}-weights-only",
                monitor="val_loss",
                mode="min",
                save_weights_only=True,
            ),
        ],
        max_epochs=cfg.max_epochs,
    )
    log_txt_file = cfg.log_txt_file
    write_to_file(log_txt_file, "-----------------------")
    write_to_file(log_txt_file, "Project: {}".format(logger._project))
    write_to_file(log_txt_file, "Experiment: {}".format(logger.experiment.name))
    write_to_file(log_txt_file, "working_dir: {}".format(os.getcwd()))
    write_to_file(log_txt_file, "")
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

    dm.setup()

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

    model = EquivarianceTrainingModule(
        network,
        lr=cfg.lr,
        image_log_period=cfg.image_logging_period,
        displace_loss_weight=cfg.displace_loss_weight,
        consistency_loss_weight=cfg.consistency_loss_weight,
        direct_correspondence_loss_weight=cfg.direct_correspondence_loss_weight,
        weight_normalize=cfg.task.weight_normalize,
        sigmoid_on=cfg.sigmoid_on,
        softmax_temperature=cfg.task.softmax_temperature,
        flow_supervision=cfg.flow_supervision,
    )

    model.cuda()
    model.train()
    if cfg.load_from_checkpoint:
        print("loaded checkpoint from")
        print(cfg.checkpoint_file)
        model.load_state_dict(
            torch.load(hydra.utils.to_absolute_path(cfg.checkpoint_file))["state_dict"]
        )

    else:
        if cfg.pretraining.checkpoint_file_action is not None:
            # # Check to see if it's a wandb checkpoint.
            # TODO: need to retrain a few things... checkpoint didn't stick...
            emb_nn_action_state_dict = load_emb_weights(
                cfg.pretraining.checkpoint_file_action, cfg.wandb, logger.experiment
            )
            # checkpoint_file_fn = maybe_load_from_wandb(
            #     cfg.pretraining.checkpoint_file_action, cfg.wandb, logger.experiment.run
            # )

            model.model.emb_nn_action.load_state_dict(emb_nn_action_state_dict)
            print(
                "-----------------------Pretrained EmbNN Action Model Loaded!-----------------------"
            )
            print(
                "Loaded Pretrained EmbNN Action: {}".format(
                    cfg.pretraining.checkpoint_file_action
                )
            )
        if cfg.pretraining.checkpoint_file_anchor is not None:
            emb_nn_anchor_state_dict = load_emb_weights(
                cfg.pretraining.checkpoint_file_anchor, cfg.wandb, logger.experiment
            )
            model.model.emb_nn_anchor.load_state_dict(emb_nn_anchor_state_dict)
            print(
                "-----------------------Pretrained EmbNN Anchor Model Loaded!-----------------------"
            )
            print(
                "Loaded Pretrained EmbNN Anchor: {}".format(
                    cfg.pretraining.checkpoint_file_anchor
                )
            )
    trainer.fit(model, dm)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
