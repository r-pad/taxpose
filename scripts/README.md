# Scripts for your project

If you write some scripts which are meant to be run stand-alone, and not imported as part of the library, put them in this directory.

# Generate the RLBench dataset (using my custom fork).

ACTUALLY - the docker doesn't work because we have local changes....

```
docker run --gpus "device=1" -it -v /usr/share/glvnd/egl_vendor.d/10_nvidia.json:/usr/share/glvnd/egl_vendor.d/10_nvidia.json -v "/home/beisner/code/RLBench:/opt/baeisner/code" -v "/data:/opt/data" beisner/taxpose:latest /bin/bash
```

Training set.

```
taskset -c 0-50 python tools/dataset_generator.py --tasks=pick_and_lift,pick_up_cup,put_knife_on_chopping_board,put_money_in_safe,push_button,reach_target,slide_block_to_target,stack_wine,take_money_out_safe,take_umbrella_out_of_umbrella_stand --save_path="/opt/data/rlbench10_collisions" --image_size=256,256 --processes=10 --episodes_per_task=100 --variations=1
```

Val set.

```
taskset -c 0-50 python tools/dataset_generator.py --tasks=pick_and_lift,pick_up_cup,put_knife_on_chopping_board,put_money_in_safe,push_button,reach_target,slide_block_to_target,stack_wine,take_money_out_safe,take_umbrella_out_of_umbrella_stand --save_path="/opt/data/rlbench10_collisions_val" --image_size=256,256 --processes=10 --episodes_per_task=10 --variations=1
```

High-resolution full set.

```
taskset -c 0-50 python tools/dataset_generator.py --tasks=pick_and_lift,pick_up_cup,put_knife_on_chopping_board,put_money_in_safe,push_button,reach_target,slide_block_to_target,stack_wine,take_money_out_safe,take_umbrella_out_of_umbrella_stand --save_path="/opt/data/rlbench10_highres" --image_size=640,640 --processes=10 --episodes_per_task=110 --variations=1
```

And for generating the tar files:

```
tar --exclude='rlbench10_collisions/.cache' -czvf rlbench10_collisions.tar.gz rlbench10_collisions
tar --exclude='rlbench10_collisions_val/.cache' -czvf rlbench10_collisions_val.tar.gz rlbench10_collisions_val
```

And for copying:
```
rsync --progress -avP rlbench10_collisions.tar.gz  baeisner@autobot.vision.cs.cmu.edu:/project_data/held/baeisner/rlbench10_collisions.tar.gz
rsync --progress -avP rlbench10_collisions_val.tar.gz  baeisner@autobot.vision.cs.cmu.edu:/project_data/held/baeisner/rlbench10_collisions_val.tar.gz
```

Here are some scratch commands for now.


# New RLBench.

# Cool thing I'm trying...

<!-- pick_and_lift -->
mkdir -p configs/checkpoints/rlbench/pick_and_lift/pretraining && touch configs/checkpoints/rlbench/pick_and_lift/pretraining/all.yaml
./scripts/train_eval.sh \
    "./launch.sh autobot 0 python scripts/train_residual_flow.py --config-name commands/rlbench/pick_and_lift/train_taxpose_all.yaml dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512 wandb.group=rlbench_pick_and_lift resources.num_workers=10"  \
    "./launch.sh autobot 0 ./configs/commands/rlbench/pick_and_lift/taxpose_all/precision_eval/precision_eval.sh dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512" \
    echo

<!-- pick_up_cup -->
mkdir -p configs/checkpoints/rlbench/pick_up_cup/pretraining && touch configs/checkpoints/rlbench/pick_up_cup/pretraining/all.yaml
./scripts/train_eval.sh \
    "./launch.sh autobot 1 python scripts/train_residual_flow.py --config-name commands/rlbench/pick_up_cup/train_taxpose_all.yaml dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512 wandb.group=rlbench_pick_up_cup resources.num_workers=10"  \
    "./launch.sh autobot 1 ./configs/commands/rlbench/pick_up_cup/taxpose_all/precision_eval/precision_eval.sh dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512" \
    echo

<!-- put_knife_on_chopping_board -->
mkdir -p configs/checkpoints/rlbench/put_knife_on_chopping_board/pretraining && touch configs/checkpoints/rlbench/put_knife_on_chopping_board/pretraining/all.yaml
./scripts/train_eval.sh \
    "./launch.sh autobot 2 python scripts/train_residual_flow.py --config-name commands/rlbench/put_knife_on_chopping_board/train_taxpose_all.yaml dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512 wandb.group=rlbench_put_knife_on_chopping_board resources.num_workers=10"  \
    "./launch.sh autobot 2 ./configs/commands/rlbench/put_knife_on_chopping_board/taxpose_all/precision_eval/precision_eval.sh dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512" \
    echo

<!-- put_money_in_safe -->
mkdir -p configs/checkpoints/rlbench/put_money_in_safe/pretraining && touch configs/checkpoints/rlbench/put_money_in_safe/pretraining/all.yaml
./scripts/train_eval.sh \
    "./launch.sh autobot 3 python scripts/train_residual_flow.py --config-name commands/rlbench/put_money_in_safe/train_taxpose_all.yaml dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512 wandb.group=rlbench_put_money_in_safe resources.num_workers=10"  \
    "./launch.sh autobot 3 ./configs/commands/rlbench/put_money_in_safe/taxpose_all/precision_eval/precision_eval.sh dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512" \
    echo

<!-- push_button -->
mkdir -p configs/checkpoints/rlbench/push_button/pretraining && touch configs/checkpoints/rlbench/push_button/pretraining/all.yaml
./scripts/train_eval.sh \
    "./launch.sh autobot 4 python scripts/train_residual_flow.py --config-name commands/rlbench/push_button/train_taxpose_all.yaml dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512 wandb.group=rlbench_push_button resources.num_workers=10"  \
    "./launch.sh autobot 4 ./configs/commands/rlbench/push_button/taxpose_all/precision_eval/precision_eval.sh dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512" \
    echo

<!-- reach_target -->
EXTRA_PARAMS="" \
    ./scripts/train_eval.sh \
    "./launch.sh local-docker 0 python scripts/train_residual_flow.py --config-name commands/rlbench/reach_target/train_taxpose_tc.yaml wandb.group=rlbench_reach_target resources.num_workers=10 ${EXTRA_PARAMS}"  \
    "./launch.sh local-docker 0 ./configs/commands/rlbench/reach_target/taxpose_tc/precision_eval/precision_eval.sh ${EXTRA_PARAMS}" \
    "./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/reach_target/taxpose_all/eval_rlbench.yaml num_trials=100 resources.num_workers=10 wandb.group=rlbench_reach_target headless=True ${EXTRA_PARAMS}"

EXTRA_PARAMS="dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512" \
    ./scripts/train_eval.sh \
    "./launch.sh autobot 0 python scripts/train_residual_flow.py --config-name commands/rlbench/reach_target/train_taxpose_all.yaml wandb.group=rlbench_reach_target resources.num_workers=10 ${EXTRA_PARAMS}"  \
    "./launch.sh autobot 0 ./configs/commands/rlbench/reach_target/taxpose_all/precision_eval/precision_eval.sh ${EXTRA_PARAMS}" \
    "./launch.sh autobot 0 python scripts/eval_rlbench.py --config-name commands/rlbench/reach_target/taxpose_all/eval_rlbench.yaml num_trials=100 resources.num_workers=10 wandb.group=rlbench_reach_target headless=True ${EXTRA_PARAMS}"

<!-- slide_block_to_target -->
mkdir -p configs/checkpoints/rlbench/slide_block_to_target/pretraining && touch configs/checkpoints/rlbench/slide_block_to_target/pretraining/all.yaml
./scripts/train_eval.sh \
    "./launch.sh autobot 6 python scripts/train_residual_flow.py --config-name commands/rlbench/slide_block_to_target/train_taxpose_all.yaml dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512 wandb.group=rlbench_slide_block_to_target resources.num_workers=10"  \
    "./launch.sh autobot 6 ./configs/commands/rlbench/slide_block_to_target/taxpose_all/precision_eval/precision_eval.sh dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512" \
    echo

<!-- stack_wine -->
mkdir -p configs/checkpoints/rlbench/stack_wine/pretraining && touch configs/checkpoints/rlbench/stack_wine/pretraining/all.yaml
./scripts/train_eval.sh \
    "./launch.sh autobot 7 python scripts/train_residual_flow.py --config-name commands/rlbench/stack_wine/train_taxpose_all.yaml dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512 wandb.group=rlbench_stack_wine resources.num_workers=10"  \
    "./launch.sh autobot 7 ./configs/commands/rlbench/stack_wine/taxpose_all/precision_eval/precision_eval.sh dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512" \
    echo

<!-- take_money_out_safe -->
mkdir -p configs/checkpoints/rlbench/take_money_out_safe/pretraining && touch configs/checkpoints/rlbench/take_money_out_safe/pretraining/all.yaml
./scripts/train_eval.sh \
    "./launch.sh autobot 0 python scripts/train_residual_flow.py --config-name commands/rlbench/take_money_out_safe/train_taxpose_all.yaml dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512 wandb.group=rlbench_take_money_out_safe resources.num_workers=10 data_root=/home/beisner/datasets"  \
    "./launch.sh autobot 0 ./configs/commands/rlbench/take_money_out_safe/taxpose_all/precision_eval/precision_eval.sh dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512 data_root=/home/beisner/datasets" \
    echo

<!-- take_umbrella_out_of_umbrella_stand -->
mkdir -p configs/checkpoints/rlbench/take_umbrella_out_of_umbrella_stand/pretraining && touch configs/checkpoints/rlbench/take_umbrella_out_of_umbrella_stand/pretraining/all.yaml
./scripts/train_eval.sh \
    "./launch.sh autobot 1 python scripts/train_residual_flow.py --config-name commands/rlbench/take_umbrella_out_of_umbrella_stand/train_taxpose_all.yaml dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512 wandb.group=rlbench_take_umbrella_out_of_umbrella_stand resources.num_workers=10 data_root=/home/beisner/datasets"  \
    "./launch.sh autobot 1 ./configs/commands/rlbench/take_umbrella_out_of_umbrella_stand/taxpose_all/precision_eval/precision_eval.sh dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512 data_root=/home/beisner/datasets" \
    echo


### Redo the evals...
./launch.sh autobot 0 ./configs/commands/rlbench/pick_and_lift/taxpose_all/precision_eval/precision_eval.sh dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512 checkpoints.ckpt_file=r-pad/taxpose/model-a8vhdn5b:v0

./launch.sh autobot 1 ./configs/commands/rlbench/pick_up_cup/taxpose_all/precision_eval/precision_eval.sh dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512 checkpoints.ckpt_file=r-pad/taxpose/model-nay0eueg:v0

./launch.sh autobot 2 ./configs/commands/rlbench/put_knife_on_chopping_board/taxpose_all/precision_eval/precision_eval.sh dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512 checkpoints.ckpt_file=r-pad/taxpose/model-opjzx3mi:v0

./launch.sh autobot 3 ./configs/commands/rlbench/put_money_in_safe/taxpose_all/precision_eval/precision_eval.sh dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512 checkpoints.ckpt_file=r-pad/taxpose/model-m2eb9u1s:v0

./launch.sh autobot 4 ./configs/commands/rlbench/push_button/taxpose_all/precision_eval/precision_eval.sh dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512 checkpoints.ckpt_file=r-pad/taxpose/model-gjzsgkkh:v0

./launch.sh autobot 5 ./configs/commands/rlbench/reach_target/taxpose_all/precision_eval/precision_eval.sh dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512 checkpoints.ckpt_file=r-pad/taxpose/model-yz9e7iz8:v0

./launch.sh autobot 6 ./configs/commands/rlbench/slide_block_to_target/taxpose_all/precision_eval/precision_eval.sh dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512 checkpoints.ckpt_file=r-pad/taxpose/model-kpbx8qf4:v0

./launch.sh autobot 7 ./configs/commands/rlbench/stack_wine/taxpose_all/precision_eval/precision_eval.sh dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512 checkpoints.ckpt_file=r-pad/taxpose/model-nixx3qii:v0

./launch.sh autobot 8 ./configs/commands/rlbench/take_money_out_safe/taxpose_all/precision_eval/precision_eval.sh dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512 checkpoints.ckpt_file=r-pad/taxpose/model-1gmi342v:v0

./launch.sh local 1 ./configs/commands/rlbench/take_umbrella_out_of_umbrella_stand/taxpose_all/precision_eval/precision_eval.sh dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512 checkpoints.ckpt_file=r-pad/taxpose/model-1y7t5g4o:v0 data_root=/home/beisner/datasets




# RGB training + evals.

RLBENCH_10_TASKS = [
    "pick_and_lift",
    "pick_up_cup",
    "put_knife_on_chopping_board",
    "put_money_in_safe",
    "push_button",
    "reach_target",
    "slide_block_to_target",
    "stack_wine",
    "take_money_out_safe",
    "take_umbrella_out_of_umbrella_stand",
]

## Training


./launch.sh autobot 0 python scripts/train_residual_flow.py --config-name commands/rlbench/pick_and_lift/train_taxpose_tc.yaml dm.train_dset.include_rgb_features=True model.feature_channels=3 benchmark.dataset_root=/data/rlbench10_collisions

./launch.sh autobot 1 python scripts/train_residual_flow.py --config-name commands/rlbench/pick_up_cup/train_taxpose_tc.yaml dm.train_dset.include_rgb_features=True model.feature_channels=3 benchmark.dataset_root=/data/rlbench10_collisions

./launch.sh autobot 2 python scripts/train_residual_flow.py --config-name commands/rlbench/put_knife_on_chopping_board/train_taxpose_tc.yaml dm.train_dset.include_rgb_features=True model.feature_channels=3 benchmark.dataset_root=/data/rlbench10_collisions

./launch.sh autobot 3 python scripts/train_residual_flow.py --config-name commands/rlbench/put_money_in_safe/train_taxpose_tc.yaml dm.train_dset.include_rgb_features=True model.feature_channels=3 benchmark.dataset_root=/data/rlbench10_collisions

./launch.sh autobot 4 python scripts/train_residual_flow.py --config-name commands/rlbench/push_button/train_taxpose_tc.yaml dm.train_dset.include_rgb_features=True model.feature_channels=3 benchmark.dataset_root=/data/rlbench10_collisions

./launch.sh autobot 5 python scripts/train_residual_flow.py --config-name commands/rlbench/reach_target/train_taxpose_tc.yaml dm.train_dset.include_rgb_features=True model.feature_channels=3 benchmark.dataset_root=/data/rlbench10_collisions

./launch.sh autobot 6 python scripts/train_residual_flow.py --config-name commands/rlbench/slide_block_to_target/train_taxpose_tc.yaml dm.train_dset.include_rgb_features=True model.feature_channels=3 benchmark.dataset_root=/data/rlbench10_collisions

./launch.sh autobot 7 python scripts/train_residual_flow.py --config-name commands/rlbench/stack_wine/train_taxpose_tc.yaml dm.train_dset.include_rgb_features=True model.feature_channels=3 benchmark.dataset_root=/data/rlbench10_collisions

./launch.sh autobot 8 python scripts/train_residual_flow.py --config-name commands/rlbench/take_money_out_safe/train_taxpose_tc.yaml dm.train_dset.include_rgb_features=True model.feature_channels=3 benchmark.dataset_root=/data/rlbench10_collisions

./launch.sh autobot 9 python scripts/train_residual_flow.py --config-name commands/rlbench/take_umbrella_out_of_umbrella_stand/train_taxpose_tc.yaml dm.train_dset.include_rgb_features=True model.feature_channels=3 benchmark.dataset_root=/data/rlbench10_collisions

## Checkpoints from this training run

pick_and_lift:                          r-pad/taxpose/model-9tx1uje9:v0
pick_up_cup:                            r-pad/taxpose/model-9pfjeq0j:v0
put_knife_on_chopping_board:            r-pad/taxpose/model-c9u0u4np:v0
put_money_in_safe:                      r-pad/taxpose/model-lq4b953m:v0
push_button:                            r-pad/taxpose/model-0kxpww1x:v0
reach_target:                           r-pad/taxpose/model-w5kjqoph:v0
slide_block_to_target:                  r-pad/taxpose/model-fct8vbrq:v0
stack_wine:                             r-pad/taxpose/model-cbe1hgx4:v0
take_money_out_safe:                    r-pad/taxpose/model-j3swo5k7:v0
take_umbrella_out_of_umbrella_stand:    r-pad/taxpose/model-txvpna0v:v0


## Precision Eval.


```
# pick_and_lift

./configs/commands/rlbench/pick_and_lift/taxpose_tc/precision_eval/precision_eval.sh dm.train_dset.include_rgb_features=True model.feature_channels=3 checkpoint=r-pad/taxpose/model-9tx1uje9:v0 benchmark.dataset_root=/data/rlbench10_collisions

# pick_up_cup

./configs/commands/rlbench/pick_up_cup/taxpose_tc/precision_eval/precision_eval.sh dm.train_dset.include_rgb_features=True model.feature_channels=3 checkpoint=r-pad/taxpose/model-9pfjeq0j:v0 benchmark.dataset_root=/data/rlbench10_collisions

# put_knife_on_chopping_board

./configs/commands/rlbench/put_knife_on_chopping_board/taxpose_tc/precision_eval/precision_eval.sh dm.train_dset.include_rgb_features=True model.feature_channels=3 checkpoint=r-pad/taxpose/model-c9u0u4np:v0 benchmark.dataset_root=/data/rlbench10_collisions

# put_money_in_safe

./configs/commands/rlbench/put_money_in_safe/taxpose_tc/precision_eval/precision_eval.sh dm.train_dset.include_rgb_features=True model.feature_channels=3 checkpoint=r-pad/taxpose/model-lq4b953m:v0 benchmark.dataset_root=/data/rlbench10_collisions

# push_button

./configs/commands/rlbench/push_button/taxpose_tc/precision_eval/precision_eval.sh dm.train_dset.include_rgb_features=True model.feature_channels=3 checkpoint=r-pad/taxpose/model-0kxpww1x:v0 benchmark.dataset_root=/data/rlbench10_collisions

# reach_target

./configs/commands/rlbench/reach_target/taxpose_tc/precision_eval/precision_eval.sh dm.train_dset.include_rgb_features=True model.feature_channels=3 checkpoint=r-pad/taxpose/model-w5kjqoph:v0 benchmark.dataset_root=/data/rlbench10_collisions

# slide_block_to_target

./configs/commands/rlbench/slide_block_to_target/taxpose_tc/precision_eval/precision_eval.sh dm.train_dset.include_rgb_features=True model.feature_channels=3 checkpoint=r-pad/taxpose/model-fct8vbrq:v0 benchmark.dataset_root=/data/rlbench10_collisions

# stack_wine

./configs/commands/rlbench/stack_wine/taxpose_tc/precision_eval/precision_eval.sh dm.train_dset.include_rgb_features=True model.feature_channels=3 checkpoint=r-pad/taxpose/model-cbe1hgx4:v0 benchmark.dataset_root=/data/rlbench10_collisions

# take_money_out_safe

./configs/commands/rlbench/take_money_out_safe/taxpose_tc/precision_eval/precision_eval.sh dm.train_dset.include_rgb_features=True model.feature_channels=3 checkpoint=r-pad/taxpose/model-j3swo5k7:v0 benchmark.dataset_root=/data/rlbench10_collisions

# take_umbrella_out_of_umbrella_stand

./configs/commands/rlbench/take_umbrella_out_of_umbrella_stand/taxpose_tc/precision_eval/precision_eval.sh dm.train_dset.include_rgb_features=True model.feature_channels=3 checkpoint=r-pad/taxpose/model-txvpna0v:v0 benchmark.dataset_root=/data/rlbench10_collisions

```



## RLBench Eval.

```
# pick_and_lift

./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/pick_and_lift/taxpose_tc/eval_rlbench.yaml num_trials=100 policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-9tx1uje9:v0 wandb.group=rlbench_pick_and_lift

# pick_up_cup

./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/pick_up_cup/taxpose_tc/eval_rlbench.yaml num_trials=100 policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-9pfjeq0j:v0 wandb.group=rlbench_pick_up_cup

# put_knife_on_chopping_board

./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/put_knife_on_chopping_board/taxpose_tc/eval_rlbench.yaml num_trials=100 policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-c9u0u4np:v0 wandb.group=rlbench_put_knife_on_chopping_board

# put_money_in_safe

./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/put_money_in_safe/taxpose_tc/eval_rlbench.yaml num_trials=100 policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-lq4b953m:v0 wandb.group=rlbench_put_money_in_safe

# push_button

./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/push_button/taxpose_tc/eval_rlbench.yaml num_trials=100 policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-0kxpww1x:v0 wandb.group=rlbench_push_button

# reach_target

./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/reach_target/taxpose_tc/eval_rlbench.yaml num_trials=100 policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-w5kjqoph:v0 wandb.group=rlbench_reach_target

# slide_block_to_target

./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/slide_block_to_target/taxpose_tc/eval_rlbench.yaml num_trials=100 policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-fct8vbrq:v0 wandb.group=rlbench_slide_block_to_target

# stack_wine

./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/stack_wine/taxpose_tc/eval_rlbench.yaml num_trials=100 policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-cbe1hgx4:v0 wandb.group=rlbench_stack_wine

# take_money_out_safe

./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/take_money_out_safe/taxpose_tc/eval_rlbench.yaml num_trials=100 policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-j3swo5k7:v0 wandb.group=rlbench_take_money_out_safe

# take_umbrella_out_of_umbrella_stand

./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/take_umbrella_out_of_umbrella_stand/taxpose_tc/eval_rlbench.yaml num_trials=100 policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-txvpna0v:v0 wandb.group=rlbench_take_umbrella_out_of_umbrella_stand

```
