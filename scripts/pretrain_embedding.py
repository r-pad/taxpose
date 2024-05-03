import json
import os

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from taxpose.datasets.pretraining_point_cloud_data_module import (
    PretrainingMultiviewDataModule,
)
from taxpose.nets.transformer_flow import EquivariantFeatureEmbeddingNetwork
from taxpose.training.equivariant_feature_pretraining_module import (
    EquivariancePreTrainingModule,
)


@hydra.main(version_base="1.1", config_path="../configs", config_name="pretraining")
def main(cfg):
    print(
        json.dumps(
            omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
            sort_keys=True,
            indent=4,
        )
    )

    ######################################################################
    # Torch settings.
    ######################################################################

    # Make deterministic + reproducible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Since most of us are training on 3090s+, we can use mixed precision.
    torch.set_float32_matmul_precision("medium")

    pl.seed_everything(cfg.seed)

    TRAINING = "PYTEST_CURRENT_TEST" not in os.environ

    logger = WandbLogger(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        group=cfg.wandb.group,
        save_dir=cfg.wandb.save_dir,
        job_type=cfg.job_type,
        save_code=True,
        log_model=True,
        config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
    )

    trainer = pl.Trainer(
        logger=logger if TRAINING else None,
        accelerator="gpu",
        devices=[0],
        log_every_n_steps=cfg.training.log_every_n_steps,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        max_epochs=cfg.training.epochs,
        callbacks=(
            [
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
                    monitor="train_loss",
                    mode="min",
                    save_weights_only=True,
                ),
            ]
            if TRAINING
            else []
        ),
        fast_dev_run=5 if "PYTEST_CURRENT_TEST" in os.environ else False,
    )

    dm = PretrainingMultiviewDataModule(
        cfg=cfg.dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.resources.num_workers,
    )

    network = EquivariantFeatureEmbeddingNetwork(encoder_cfg=cfg.encoder)
    model = EquivariancePreTrainingModule(
        network,
        lr=cfg.training.lr,
        image_log_period=cfg.training.image_logging_period,
        l2_reg_weight=cfg.training.l2_reg_weight,
        normalize_features=cfg.training.normalize_features,
        temperature=cfg.training.temperature,
        con_weighting=cfg.training.con_weighting,
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    # torch.cuda.empty_cache()
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
