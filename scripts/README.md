# Scripts for your project

If you write some scripts which are meant to be run stand-alone, and not imported as part of the library, put them in this directory.

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
mkdir -p configs/checkpoints/rlbench/reach_target/pretraining && touch configs/checkpoints/rlbench/reach_target/pretraining/all.yaml
EXTRA_PARAMS="dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512" \
    ./scripts/train_eval.sh \
    "./launch.sh local-docker 0 python scripts/train_residual_flow.py --config-name commands/rlbench/reach_target/train_taxpose_all.yaml wandb.group=rlbench_reach_target resources.num_workers=10 ${EXTRA_PARAMS}"  \
    "./launch.sh local-docker 0 ./configs/commands/rlbench/reach_target/taxpose_all/precision_eval/precision_eval.sh ${EXTRA_PARAMS}" \
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
