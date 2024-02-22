# !/bin/bash

NUM_TRIALS=100

taskset -c 0-50 python scripts/eval_rlbench.py --config-name commands/rlbench/pick_and_lift/taxpose_all/eval_rlbench.yaml num_trials=$NUM_TRIALS resources.num_workers=10 wandb.group=rlbench_pick_and_lift headless=True

taskset -c 0-50 python scripts/eval_rlbench.py --config-name commands/rlbench/pick_up_cup/taxpose_all/eval_rlbench.yaml num_trials=$NUM_TRIALS resources.num_workers=10 wandb.group=rlbench_pick_up_cup headless=True

taskset -c 0-50 python scripts/eval_rlbench.py --config-name commands/rlbench/put_knife_on_chopping_board/taxpose_all/eval_rlbench.yaml num_trials=$NUM_TRIALS resources.num_workers=10 wandb.group=rlbench_put_knife_on_chopping_board headless=True

taskset -c 0-50 python scripts/eval_rlbench.py --config-name commands/rlbench/put_money_in_safe/taxpose_all/eval_rlbench.yaml num_trials=$NUM_TRIALS resources.num_workers=10 wandb.group=rlbench_put_money_in_safe headless=True

taskset -c 0-50 python scripts/eval_rlbench.py --config-name commands/rlbench/push_button/taxpose_all/eval_rlbench.yaml num_trials=$NUM_TRIALS resources.num_workers=10 wandb.group=rlbench_push_button headless=True

taskset -c 0-50 python scripts/eval_rlbench.py --config-name commands/rlbench/reach_target/taxpose_all/eval_rlbench.yaml num_trials=$NUM_TRIALS resources.num_workers=10 wandb.group=rlbench_reach_target headless=True

taskset -c 0-50 python scripts/eval_rlbench.py --config-name commands/rlbench/slide_block_to_target/taxpose_all/eval_rlbench.yaml num_trials=$NUM_TRIALS resources.num_workers=10 wandb.group=rlbench_slide_block_to_target headless=True

taskset -c 0-50 python scripts/eval_rlbench.py --config-name commands/rlbench/stack_wine/taxpose_all/eval_rlbench.yaml num_trials=$NUM_TRIALS resources.num_workers=10 wandb.group=rlbench_stack_wine headless=True

taskset -c 0-50 python scripts/eval_rlbench.py --config-name commands/rlbench/take_money_out_safe/taxpose_all/eval_rlbench.yaml num_trials=$NUM_TRIALS resources.num_workers=10 wandb.group=rlbench_take_money_out_safe headless=True

taskset -c 0-50 python scripts/eval_rlbench.py --config-name commands/rlbench/take_umbrella_out_of_umbrella_stand/taxpose_all/eval_rlbench.yaml num_trials=$NUM_TRIALS resources.num_workers=10 wandb.group=rlbench_take_umbrella_out_of_umbrella_stand headless=True
