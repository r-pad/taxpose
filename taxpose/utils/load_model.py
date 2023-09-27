import os

import wandb


def get_weights_path(
    checkpoint_reference, wandb_cfg, run=None, weights_file="model.ckpt"
):
    if checkpoint_reference.startswith(wandb_cfg.entity):
        # download checkpoint locally (if not already cached)
        artifact_dir = os.path.join(wandb_cfg.artifact_dir, checkpoint_reference)

        if isinstance(run, wandb.sdk.wandb_run.Run):
            artifact = run.use_artifact(checkpoint_reference)
        else:
            api = wandb.Api()
            artifact = api.artifact(checkpoint_reference)

        try:
            ckpt_file = artifact.get_path(weights_file).download(root=artifact_dir)
        except KeyError:
            # I re-uploaded a few failed runs...
            ckpt_file = artifact.get_path("last.ckpt").download(root=artifact_dir)
    else:
        ckpt_file = checkpoint_reference

    return ckpt_file
