import os
import shutil
from typing import List

import typer
import yaml
from rpad.rlbench_utils.task_info import TASK_DICT

DRY_RUN = False


def write_file(path: str, contents: str, dry_run: bool = False, overwrite: bool = True):
    if os.path.exists(path) and not overwrite:
        print(f"File {path} already exists and overwrite is False. Skipping.")
        return
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


def generate_training_command_configs(
    task_name: str, config_root: str, dry_run: bool = False
):
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


def generate_precision_eval_command_configs(
    task_name: str, config_root: str, method_name: str, dry_run: bool = False
):
    commands_dir = os.path.join(
        config_root, "commands", "rlbench", task_name, method_name, "precision_eval"
    )

    make_dirs(commands_dir, dry_run=dry_run)

    # For each phase, create a file called "phase.yaml" in the taxpose directory, which is a copy of
    # commands/rlbench/task_name/train_taxpose_all.yaml. Replace "_train" with "_eval", and replace
    # "/phase: all" with "/phase: phase".
    for phase in TASK_DICT[task_name]["phase_order"]:
        default_file = os.path.join(
            config_root, "commands", "rlbench", task_name, "train_taxpose_all.yaml"
        )
        task_file = os.path.join(commands_dir, f"{phase}.yaml")

        with open(default_file, "r") as f:
            contents = f.read()
        contents = contents.replace("DEFAULT", task_name)
        contents = contents.replace("_train", "_eval")
        contents = contents.replace("/phase: all", f"/phase: {phase}")
        contents = contents.replace("/model: taxpose\n", f"/model: {method_name}\n")
        write_file(task_file, contents, dry_run=dry_run)


def generate_rlbench_eval_command_configs(
    task_name: str, config_root: str, method_name: str, dry_run: bool = False
):
    commands_dir = os.path.join(
        config_root, "commands", "rlbench", task_name, method_name
    )

    make_dirs(commands_dir, dry_run=dry_run)

    # Copy the "train_taxpose_all.yaml" file from the task directory to the method directory.
    default_file = os.path.join(
        config_root, "commands", "rlbench", task_name, "train_taxpose_all.yaml"
    )

    task_file = os.path.join(commands_dir, "eval_rlbench.yaml")
    copy_file(default_file, task_file, dry_run=dry_run)

    # Parse the task_file as a yaml file.
    with open(task_file, "r") as f:
        contents = yaml.load(f, Loader=yaml.FullLoader)
    contents["defaults"][0] = "/eval_rlbench.yaml"
    contents["defaults"][1] = {"override /model": method_name}
    del contents["defaults"][3]

    # Set the model number of points
    if "dm" in contents:
        contents["model"]["num_points"] = contents["dm"]["train_dset"]["demo_dset"][
            "num_points"
        ]
    contents["model"]["z_offset"] = 0.0
    contents["model"]["break_symmetry"] = contents["break_symmetry"]

    # Write the contents back to the file.
    contents_str = yaml.dump(contents)

    # Add a line to the top of the file that says "# @package _global_".
    contents_str = "# @package _global_\n\n" + contents_str
    write_file(task_file, contents_str, dry_run=dry_run)


def generate_precision_eval_script(
    task_name: str, config_root: str, method_name: str, dry_run: bool = False
):
    script_dir = os.path.join(
        config_root, "commands", "rlbench", task_name, method_name, "precision_eval"
    )
    make_dirs(script_dir, dry_run=dry_run)

    # Create a file called "precision_eval.sh" in the precision_eval directory.
    script_file = os.path.join(script_dir, "precision_eval.sh")

    commands = [
        (
            phase,
            f"python scripts/eval_metrics.py --config-name commands/rlbench/{task_name}/{method_name}/precision_eval/{phase} data_root=/data wandb.group=rlbench_{task_name}",
        )
        for phase in TASK_DICT[task_name]["phase_order"]
    ]

    contents = "#!/bin/bash\n\n"
    # We want to pass all arguments provided to the script to each command.
    contents += "set -e\n\n"
    contents += "echo 'Running precision eval'\n\n"
    for phase, command in commands:
        contents += 'echo "' + "-" * 80 + '"\n'
        contents += f"echo 'Evaluating {phase}'\n"
        contents += 'echo "' + "-" * 80 + '"\n'
        contents += 'echo "' + command + ' $@"\n\n'

        # Pass the arguments to the command.
        contents += command + " $@\n\n"

    write_file(script_file, contents, dry_run=dry_run)

    # Chmod the file to make it executable.
    if not dry_run:
        os.chmod(script_file, 0o777)


def generate_checkpoint_configs(
    task_name: str, config_root: str, method_name: str, dry_run: bool = False
):
    checkpoint_dir = os.path.join(config_root, "checkpoints", "rlbench", task_name)

    make_dirs(checkpoint_dir, dry_run=dry_run)
    make_dirs(os.path.join(checkpoint_dir, "pretraining"), dry_run=dry_run)

    # Create empty files for the pretraining checkpoints.
    for phase in TASK_DICT[task_name]["phase_order"] + ["all"]:
        phase_file = os.path.join(checkpoint_dir, "pretraining", f"{phase}.yaml")
        write_file(phase_file, "", dry_run=dry_run)


def generate_method_checkpoint_configs(
    task_name: str, config_root: str, method_name: str, dry_run: bool = False
):
    checkpoint_dir = os.path.join(config_root, "checkpoints", "rlbench", task_name)

    # Create a method folder in the task directory.
    method_dir = os.path.join(checkpoint_dir, method_name)
    make_dirs(method_dir, dry_run=dry_run)

    # For each phase, create a file called "phase.yaml" in the method directory.
    for phase in TASK_DICT[task_name]["phase_order"]:
        phase_file = os.path.join(method_dir, f"{phase}.yaml")
        # contents = {"ckpt_file": "r-pad/taxpose/model-???:v0"}
        contents = {
            "defaults": [
                f"/checkpoints/rlbench/{task_name}/{method_name}/_model@_here_"
            ]
        }
        contents = yaml.dump(contents)

        # Prepend the contents with "# @package checkpoints.{task_name}.{method_name}".
        # contents = (
        #     f"# @package checkpoints.rlbench.{task_name}.{method_name}\n\n" + contents
        # )
        write_file(phase_file, contents, dry_run=dry_run, overwrite=True)

    # Create a file called _model.yaml
    model_file = os.path.join(method_dir, "_model.yaml")
    contents = {"ckpt_file": f"r-pad/{method_name}/model-???:v0"}
    contents = yaml.dump(contents)
    write_file(model_file, contents, dry_run=dry_run, overwrite=False)

    # Create a file called method_name in the task directory.
    method_file = os.path.join(checkpoint_dir, f"{method_name}.yaml")
    contents = {
        "defaults": [
            {f"{task_name}/{method_name}@{phase}": phase}
            for phase in TASK_DICT[task_name]["phase_order"]
        ]
    }
    contents_str = yaml.dump(contents)
    write_file(method_file, contents_str, dry_run=dry_run)


def main(task_names: List[str], dry_run: bool = False, evals: bool = True):
    for task_name in task_names:
        print(f"Generating task configs for {task_name}")

        config_root = "configs"

        if not evals:
            generate_task_configs(task_name, config_root, dry_run=dry_run)

            generate_dataset_configs(task_name, config_root, dry_run=dry_run)

            generate_training_command_configs(task_name, config_root, dry_run=dry_run)

            generate_checkpoint_configs(
                task_name, config_root, "taxpose_all", dry_run=dry_run
            )

        else:
            generate_method_checkpoint_configs(
                task_name, config_root, "taxpose_all", dry_run=dry_run
            )
            generate_precision_eval_command_configs(
                task_name, config_root, "taxpose_all", dry_run=dry_run
            )

            generate_precision_eval_script(
                task_name, config_root, "taxpose_all", dry_run=dry_run
            )

            generate_rlbench_eval_command_configs(
                task_name, config_root, "taxpose_all", dry_run=dry_run
            )


if __name__ == "__main__":
    typer.run(main)
