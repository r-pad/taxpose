# !/bin/bash

NUM_TRIALS=100

# pick_and_lift
# taskset -c 0-50 python scripts/eval_rlbench.py --config-name commands/rlbench/pick_and_lift/taxpose_all/eval_rlbench.yaml num_trials=$NUM_TRIALS resources.num_workers=10 wandb.group=rlbench_pick_and_lift headless=True task.action_mode=gripper_and_object task.anchor_mode=background_robot_removed model.num_points=512

# pick_up_cup
# taskset -c 0-50 python scripts/eval_rlbench.py --config-name commands/rlbench/pick_up_cup/taxpose_all/eval_rlbench.yaml num_trials=$NUM_TRIALS resources.num_workers=10 wandb.group=rlbench_pick_up_cup headless=True task.action_mode=gripper_and_object task.anchor_mode=background_robot_removed model.num_points=512

# put_knife_on_chopping_board
taskset -c 0-50 python scripts/eval_rlbench.py --config-name commands/rlbench/put_knife_on_chopping_board/taxpose_all/eval_rlbench.yaml num_trials=$NUM_TRIALS resources.num_workers=10 wandb.group=rlbench_put_knife_on_chopping_board headless=True task.action_mode=gripper_and_object task.anchor_mode=background_robot_removed model.num_points=512

# put_money_in_safe
taskset -c 0-50 python scripts/eval_rlbench.py --config-name commands/rlbench/put_money_in_safe/taxpose_all/eval_rlbench.yaml num_trials=$NUM_TRIALS resources.num_workers=5 wandb.group=rlbench_put_money_in_safe headless=True task.action_mode=gripper_and_object task.anchor_mode=background_robot_removed model.num_points=512

# push_button
taskset -c 0-50 python scripts/eval_rlbench.py --config-name commands/rlbench/push_button/taxpose_all/eval_rlbench.yaml num_trials=$NUM_TRIALS resources.num_workers=10 wandb.group=rlbench_push_button headless=True task.action_mode=gripper_and_object task.anchor_mode=background_robot_removed model.num_points=512

# reach_target
taskset -c 0-50 python scripts/eval_rlbench.py --config-name commands/rlbench/reach_target/taxpose_all/eval_rlbench.yaml num_trials=$NUM_TRIALS resources.num_workers=10 wandb.group=rlbench_reach_target headless=True task.action_mode=gripper_and_object task.anchor_mode=background_robot_removed model.num_points=512

# slide_block_to_target
taskset -c 0-50 python scripts/eval_rlbench.py --config-name commands/rlbench/slide_block_to_target/taxpose_all/eval_rlbench.yaml num_trials=$NUM_TRIALS resources.num_workers=10 wandb.group=rlbench_slide_block_to_target headless=True task.action_mode=gripper_and_object task.anchor_mode=background_robot_removed model.num_points=512

# stack_wine
taskset -c 0-50 python scripts/eval_rlbench.py --config-name commands/rlbench/stack_wine/taxpose_all/eval_rlbench.yaml num_trials=$NUM_TRIALS resources.num_workers=10 wandb.group=rlbench_stack_wine headless=True task.action_mode=gripper_and_object task.anchor_mode=background_robot_removed model.num_points=512

# take_money_out_safe
taskset -c 0-50 python scripts/eval_rlbench.py --config-name commands/rlbench/take_money_out_safe/taxpose_all/eval_rlbench.yaml num_trials=$NUM_TRIALS resources.num_workers=5 wandb.group=rlbench_take_money_out_safe headless=True task.action_mode=gripper_and_object task.anchor_mode=background_robot_removed model.num_points=512

# take_umbrella_out_of_umbrella_stand
taskset -c 0-50 python scripts/eval_rlbench.py --config-name commands/rlbench/take_umbrella_out_of_umbrella_stand/taxpose_all/eval_rlbench.yaml num_trials=$NUM_TRIALS resources.num_workers=10 wandb.group=rlbench_take_umbrella_out_of_umbrella_stand headless=True task.action_mode=gripper_and_object task.anchor_mode=background_robot_removed model.num_points=512
