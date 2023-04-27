import os

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

from taxpose.datasets.point_cloud_data_module import MultiviewDataModule
from taxpose.nets.transformer_flow import ResidualFlow_DiffEmbTransformer
from taxpose.training.flow_equivariance_training_module_nocentering import (
    EquivarianceTrainingModule,
)
from taxpose.utils.callbacks import SaverCallbackEmbnnActionAnchor, SaverCallbackModel


def write_to_file(file_name, string):
    with open(file_name, "a") as f:
        f.writelines(string)
        f.write("\n")
    f.close()


@hydra.main(config_path="../configs", config_name="train_mug_residual")
def main(cfg):
    pl.seed_everything(cfg.seed)
    logger = WandbLogger(project=cfg.experiment)
    logger.log_hyperparams(cfg)
    logger.log_hyperparams({"working_dir": os.getcwd()})
    trainer = pl.Trainer(
        logger=logger,
        gpus=1,
        reload_dataloaders_every_n_epochs=1,
        callbacks=[SaverCallbackModel(), SaverCallbackEmbnnActionAnchor()],
        max_epochs=cfg.max_epochs,
    )
    log_txt_file = cfg.log_txt_file
    if cfg.mode == "train":
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
        synthetic_occlusion=cfg.synthetic_occlusion,
        ball_radius=cfg.ball_radius,
        ball_occlusion=cfg.ball_occlusion,
        plane_standoff=cfg.plane_standoff,
        plane_occlusion=cfg.plane_occlusion,
        num_demo=cfg.num_demo,
        occlusion_class=cfg.occlusion_class,
    )

    dm.setup()

    network = ResidualFlow_DiffEmbTransformer(
        emb_dims=cfg.emb_dims,
        emb_nn=cfg.emb_nn,
        return_flow_component=cfg.return_flow_component,
        center_feature=cfg.center_feature,
        pred_weight=cfg.pred_weight,
        multilaterate=cfg.multilaterate,
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
        if cfg.task.checkpoint_file_action is not None:
            model.model.emb_nn_action.load_state_dict(
                torch.load(
                    hydra.utils.to_absolute_path(cfg.task.checkpoint_file_action)
                )["embnn_state_dict"]
            )
            print(
                "-----------------------Pretrained EmbNN Action Model Loaded!-----------------------"
            )
            print(
                "Loaded Pretrained EmbNN Action: {}".format(
                    cfg.task.checkpoint_file_action
                )
            )
        if cfg.task.checkpoint_file_anchor is not None:
            model.model.emb_nn_anchor.load_state_dict(
                torch.load(
                    hydra.utils.to_absolute_path(cfg.task.checkpoint_file_anchor)
                )["embnn_state_dict"]
            )
            print(
                "-----------------------Pretrained EmbNN Anchor Model Loaded!-----------------------"
            )
            print(
                "Loaded Pretrained EmbNN Anchor: {}".format(
                    cfg.task.checkpoint_file_anchor
                )
            )
    if cfg.mode == "train":
        trainer.fit(model, dm)
    elif cfg.mode == "train_val":
        pl.seed_everything(123456)
        model.eval()

        with torch.no_grad():
            results = trainer.validate(model, dm.train_dataloader())
    elif cfg.mode == "validate":
        pl.seed_everything(123456)
        model.eval()

        with torch.no_grad():
            results = trainer.validate(model, dm)
    elif cfg.mode == "test":
        pl.seed_everything(123456)
        model.eval()

        with torch.no_grad():
            results = trainer.test(model, dm)
    else:
        raise ValueError("Mode not recognized")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
