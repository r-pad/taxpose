import os
import pathlib

import torch
from pytorch_lightning.callbacks import Callback


class SaverCallbackModel(Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(self):
        self.save_freq = 1000
        self.prev_path = None

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Check if we should save a checkpoint after every train batch"""
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_freq == 0 and global_step > 100:
            filename = f"epoch_{epoch}_global_step_{global_step}.ckpt"
            ckpt_path_embnn = os.path.join(
                trainer.checkpoint_callback.dirpath, filename
            )
            if not os.path.isdir(trainer.checkpoint_callback.dirpath):
                os.makedirs(trainer.checkpoint_callback.dirpath)
            torch.save({"state_dict": pl_module.state_dict()}, ckpt_path_embnn)
            if self.prev_path is not None:
                self.prev_path.unlink()
                self.prev_path = pathlib.Path(ckpt_path_embnn)


class SaverCallbackEmbnn(Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(self):
        self.save_freq = 100
        self.prev_path = None

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Check if we should save a checkpoint after every train batch"""
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_freq == 0:
            filename = f"embnn_callback_epoch_{epoch}_global_step_{global_step}.ckpt"
            ckpt_path_embnn = os.path.join(
                trainer.checkpoint_callback.dirpath, filename
            )
            if not os.path.isdir(trainer.checkpoint_callback.dirpath):
                os.makedirs(trainer.checkpoint_callback.dirpath)
            torch.save(
                {"embnn_state_dict": pl_module.model.emb_nn.state_dict()},
                ckpt_path_embnn,
            )
            if self.prev_path is not None:
                self.prev_path.unlink()
                self.prev_path = pathlib.Path(ckpt_path_embnn)


class SaverCallbackEmbnnActionAnchor(Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(self):
        self.save_freq = 1000
        self.prev_path = None

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Check if we should save a checkpoint after every train batch"""
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_freq == 0:
            filename = f"embnn_callback_epoch_{epoch}_global_step_{global_step}.ckpt"
            ckpt_path_embnn = os.path.join(
                trainer.checkpoint_callback.dirpath, filename
            )
            if not os.path.isdir(trainer.checkpoint_callback.dirpath):
                os.makedirs(trainer.checkpoint_callback.dirpath)
            torch.save(
                {
                    "embnn_action_state_dict": pl_module.model.emb_nn_action.state_dict(),
                    "embnn_anchor_state_dict": pl_module.model.emb_nn_anchor.state_dict(),
                },
                ckpt_path_embnn,
            )
            if self.prev_path is not None:
                self.prev_path.unlink()
                self.prev_path = pathlib.Path(ckpt_path_embnn)


class SaverCallbackRefinement(Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(self):
        self.save_freq = 1000
        self.prev_path = None

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Check if we should save a checkpoint after every train batch"""
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_freq == 0:
            filename = (
                f"refinement_callback_epoch_{epoch}_global_step_{global_step}.ckpt"
            )
            ckpt_path_embnn = os.path.join(
                trainer.checkpoint_callback.dirpath, filename
            )
            if not os.path.isdir(trainer.checkpoint_callback.dirpath):
                os.mkdir(trainer.checkpoint_callback.dirpath)
            torch.save(
                {"refinement_state_dict": pl_module.refinement_model.state_dict()},
                ckpt_path_embnn,
            )
            if self.prev_path is not None:
                self.prev_path.unlink()
                self.prev_path = pathlib.Path(ckpt_path_embnn)
