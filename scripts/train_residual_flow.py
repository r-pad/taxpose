import os

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

from taxpose.datasets.point_cloud_data_module import MultiviewDataModule
from taxpose.models.transformer_flow import ResidualFlow_DiffEmbTransformer
from taxpose.training.flow_equivariance_training_module_nocentering import (
    EquivarianceTrainingModule,
)
from taxpose.utils.callbacks import SaverCallbackEmbnnActionAnchor, SaverCallbackModel


@hydra.main(config_path="../configs", config_name="train_mug_residual")
def main(cfg):
    pl.seed_everything(cfg.seed)
    logger = WandbLogger(project=cfg.experiment, entity="r-pad")
    logger.log_hyperparams(cfg)
    logger.log_hyperparams({"working_dir": os.getcwd()})
    trainer = pl.Trainer(
        logger=logger,
        gpus=1,
        reload_dataloaders_every_n_epochs=1,
        callbacks=[SaverCallbackModel(), SaverCallbackEmbnnActionAnchor()],
    )

    dm = MultiviewDataModule(
        dataset_root=cfg.dataset_root,
        test_dataset_root=cfg.test_dataset_root,
        dataset_index=cfg.dataset_index,
        action_class=cfg.action_class,
        anchor_class=cfg.anchor_class,
        dataset_size=cfg.dataset_size,
        rotation_variance=np.pi / 180 * cfg.rotation_variance,
        translation_variance=cfg.translation_variance,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        cloud_type=cfg.cloud_type,
        num_points=cfg.num_points,
        overfit=cfg.overfit,
        synthetic_occlusion=cfg.synthetic_occlusion,
        ball_radius=cfg.ball_radius,
    )

    dm.setup()

    network = ResidualFlow_DiffEmbTransformer(
        emb_dims=cfg.emb_dims,
        emb_nn=cfg.emb_nn,
        return_flow_component=cfg.return_flow_component,
        center_feature=cfg.center_feature,
        inital_sampling_ratio=cfg.inital_sampling_ratio,
        pred_weight=cfg.pred_weight,
    )

    model = EquivarianceTrainingModule(
        network,
        lr=cfg.lr,
        image_log_period=cfg.image_logging_period,
        point_loss_type=cfg.point_loss_type,
        rotation_weight=cfg.rotation_weight,
        weight_normalize=cfg.weight_normalize,
        consistency_weight=cfg.consistency_weight,
        smoothness_weight=cfg.smoothness_weight,
        sigmoid_on=cfg.sigmoid_on,
        softmax_temperature=cfg.softmax_temperature,
    )

    model.cuda()
    model.train()
    if cfg.checkpoint_file is not None:
        print("loaded checkpoint from")
        print(cfg.checkpoint_file)
        model.load_state_dict(torch.load(cfg.checkpoint_file)["state_dict"])

    else:
        if cfg.checkpoint_file_action is not None:
            model.model.emb_nn_action.load_state_dict(
                torch.load(cfg.checkpoint_file_action)["embnn_state_dict"]
            )
            print(
                "-----------------------Pretrained EmbNN Action Model Loaded!-----------------------"
            )
        if cfg.checkpoint_file_anchor is not None:
            model.model.emb_nn_anchor.load_state_dict(
                torch.load(cfg.checkpoint_file_anchor)["embnn_state_dict"]
            )
            print(
                "-----------------------Pretrained EmbNN Anchor Model Loaded!-----------------------"
            )

    trainer.fit(model, dm)


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
