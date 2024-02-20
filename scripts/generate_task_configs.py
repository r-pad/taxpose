import os
import shutil

import typer
import yaml
from rpad.rlbench_utils.task_info import TASK_DICT

DRY_RUN = False


def write_file(path: str, contents: str, dry_run: bool = False):
    if not dry_run:
        with open(path, "w") as f:
            f.write(contents)
    else:
        print(f"Writing to {path}")
        print(contents)


def make_dirs(path: str, dry_run: bool = False):
    if not dry_run:
        os.makedirs(path, exist_ok=True)
    else:
        print(f"Making dirs at {path}")


def copy_file(src: str, dst: str, dry_run: bool = False):
    if not dry_run:
        shutil.copyfile(src, dst)
    else:
        print(f"Copying {src} to {dst}")


def generate_task_configs(task_name: str, config_root: str, dry_run: bool = False):
    task_dir = os.path.join(config_root, "task", "rlbench", task_name)

    content_dict = {}

    # First, make the task directory structure.
    make_dirs(task_dir, dry_run)
    make_dirs(os.path.join(task_dir, "phase"), dry_run)

    # Next, generate a task config file for each phase.
    for phase in TASK_DICT[task_name]["phase_order"]:
        content = {
            "name": phase,
            "action_class": TASK_DICT[task_name]["phase"][phase]["action_pose_name"],
            "anchor_class": TASK_DICT[task_name]["phase"][phase]["anchor_pose_name"],
            "softmax_temperature": 0.1,
            "weight_normalize": "softmax",
        }
        content_str = yaml.dump(content)
        content_str = "# @package task\n\n" + content_str
        content_dict[f"phase/{phase}.yaml"] = content_str

    # Write an "all" config file that includes all phases.
    content = {
        "name": "all",
        "action_class": None,
        "anchor_class": None,
        "softmax_temperature": 0.1,
        "weight_normalize": "softmax",
    }
    content_str = yaml.dump(content)
    content_str = "# @package task\n\n" + content_str
    content_dict["phase/all.yaml"] = content_str

    # Write a task-level file.
    content = {
        "name": task_name,
        "defaults": [
            {f"phase@phases.{phase}": phase}
            for phase in TASK_DICT[task_name]["phase_order"]
        ],
    }
    content_str = yaml.dump(content)
    content_dict["task.yaml"] = content_str

    for fn, contents in content_dict.items():
        write_file(os.path.join(task_dir, fn), contents, dry_run=dry_run)


def generate_dataset_configs(task_name: str, config_root: str, dry_run: bool = False):
    dataset_dir = os.path.join(config_root, "dataset", "rlbench", task_name)
    default_dir = os.path.join(config_root, "dataset", "rlbench", "_default")
    make_dirs(dataset_dir, dry_run)

    # Copy the file _default.yaml from the default directory to the task directory, using shutil.
    default_file = os.path.join(default_dir, "_default.yaml")
    task_file = os.path.join(dataset_dir, "_default.yaml")
    copy_file(default_file, task_file, dry_run=dry_run)

    for phase in TASK_DICT[task_name]["phase_order"] + ["all"]:
        content = {
            "defaults": [
                f"rlbench/{task_name}/_default@_here_",
                "_self_",
            ]
        }
        content_str = yaml.dump(content)
        write_file(
            os.path.join(dataset_dir, f"{phase}.yaml"), content_str, dry_run=dry_run
        )


def generate_command_configs(task_name: str, config_root: str, dry_run: bool = False):
    commands_dir = os.path.join(config_root, "commands", "rlbench", task_name)
    default_dir = os.path.join(config_root, "commands", "rlbench", "_default")

    make_dirs(commands_dir, dry_run=dry_run)

    # Find and replace the string "DEFAULT" with the task name in the file train_taxpose_all.yaml.
    default_file = os.path.join(default_dir, "train_taxpose_all.yaml")
    task_file = os.path.join(commands_dir, "train_taxpose_all.yaml")

    with open(default_file, "r") as f:
        contents = f.read()
    contents = contents.replace("DEFAULT", task_name)
    write_file(task_file, contents, dry_run=dry_run)


def generate_checkpoint_configs(
    task_name: str, config_root: str, dry_run: bool = False
):
    checkpoint_dir = os.path.join(config_root, "checkpoints", "rlbench", task_name)

    make_dirs(checkpoint_dir, dry_run=dry_run)
    make_dirs(os.path.join(checkpoint_dir, "pretraining"), dry_run=dry_run)

    # Create empty files for the pretraining checkpoints.
    for phase in TASK_DICT[task_name]["phase_order"] + ["all"]:
        phase_file = os.path.join(checkpoint_dir, "pretraining", f"{phase}.yaml")
        write_file(phase_file, "", dry_run=dry_run)

    # Create a taxpose folder in the task directory.
    taxpose_dir = os.path.join(checkpoint_dir, "taxpose")
    make_dirs(taxpose_dir, dry_run=dry_run)

    # For each phase, create a file called "phase.yaml" in the taxpose directory.
    for phase in TASK_DICT[task_name]["phase_order"]:
        phase_file = os.path.join(taxpose_dir, f"{phase}.yaml")
        contents = {"ckpt_file": "???"}
        contents = yaml.dump(contents)
        write_file(phase_file, contents, dry_run=dry_run)

    # Create a file called "taxpose.yaml" in the task directory.
    taxpose_file = os.path.join(checkpoint_dir, "taxpose.yaml")
    contents = {
        "defaults": [
            {f"{task_name}/taxpose@{phase}": phase}
            for phase in TASK_DICT[task_name]["phase_order"]
        ]
    }
    contents_str = yaml.dump(contents)
    write_file(taxpose_file, contents_str, dry_run=dry_run)


def main(task_name: str, dry_run: bool = False):
    print(f"Generating task configs for {task_name}")

    config_root = "configs"

    generate_task_configs(task_name, config_root, dry_run=dry_run)

    generate_dataset_configs(task_name, config_root, dry_run=dry_run)

    generate_command_configs(task_name, config_root, dry_run=dry_run)

    generate_checkpoint_configs(task_name, config_root, dry_run=dry_run)


if __name__ == "__main__":
    typer.run(main)
