import wandb

run_id = "uszuglfm"
path = "/opt/logs/train_mug_place/2023-08-14/16-50-36/checkpoints/last.ckpt"


with wandb.init(id=run_id, resume="allow") as run:
    artifact = wandb.Artifact(f"model-{run_id}", "model")
    artifact.add_file(path)
    run.log_artifact(artifact)
