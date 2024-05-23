# !/bin/bash

########################### RGB experiments #############################


# # pick_and_lift

# ./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/pick_and_lift/taxpose_tc/eval_rlbench.yaml num_trials=100 policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-9tx1uje9:v0 wandb.group=rlbench_pick_and_lift

# # pick_up_cup

# ./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/pick_up_cup/taxpose_tc/eval_rlbench.yaml num_trials=100 policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-9pfjeq0j:v0 wandb.group=rlbench_pick_up_cup

# # put_knife_on_chopping_board

# ./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/put_knife_on_chopping_board/taxpose_tc/eval_rlbench.yaml num_trials=100 policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-c9u0u4np:v0 wandb.group=rlbench_put_knife_on_chopping_board

# # put_money_in_safe

# ./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/put_money_in_safe/taxpose_tc/eval_rlbench.yaml num_trials=100 policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-lq4b953m:v0 wandb.group=rlbench_put_money_in_safe

# # push_button

# ./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/push_button/taxpose_tc/eval_rlbench.yaml num_trials=100 policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-0kxpww1x:v0 wandb.group=rlbench_push_button

# # reach_target

# ./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/reach_target/taxpose_tc/eval_rlbench.yaml num_trials=100 policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-w5kjqoph:v0 wandb.group=rlbench_reach_target

# # slide_block_to_target

# ./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/slide_block_to_target/taxpose_tc/eval_rlbench.yaml num_trials=100 policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-fct8vbrq:v0 wandb.group=rlbench_slide_block_to_target

# # stack_wine

# ./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/stack_wine/taxpose_tc/eval_rlbench.yaml num_trials=100 policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-cbe1hgx4:v0 wandb.group=rlbench_stack_wine

# # take_money_out_safe

# ./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/take_money_out_safe/taxpose_tc/eval_rlbench.yaml num_trials=100 policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-j3swo5k7:v0 wandb.group=rlbench_take_money_out_safe

# # take_umbrella_out_of_umbrella_stand

# ./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/take_umbrella_out_of_umbrella_stand/taxpose_tc/eval_rlbench.yaml num_trials=100 policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-txvpna0v:v0 wandb.group=rlbench_take_umbrella_out_of_umbrella_stand


########################### No gripper, wrist cam #############################

# pick_and_lift: hbdcvydg
# pick_up_cup: gwfjqpfk
# push_button: mmc5fhzu
# put_knife_on_chopping_board: rjscih24
# put_money_in_safe: gj3h3o3c
# reach_target: v8vv53tx
# slide_block_to_target: sejd7pz0
# stack_wine: 3hyo3r7q
# take_money_out_safe: 69gka1ew
# take_umbrella_out_of_umbrella_stand: b48mz8e1

# pick_and_lift

./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/pick_and_lift/taxpose_tc/eval_rlbench.yaml num_trials=100 policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-hbdcvydg:v0 wandb.group=rlbench_pick_and_lift

# pick_up_cup

./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/pick_up_cup/taxpose_tc/eval_rlbench.yaml num_trials=100 policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-gwfjqpfk:v0 wandb.group=rlbench_pick_up_cup

# put_knife_on_chopping_board

./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/put_knife_on_chopping_board/taxpose_tc/eval_rlbench.yaml num_trials=100 policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-rjscih24:v0 wandb.group=rlbench_put_knife_on_chopping_board

# put_money_in_safe

./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/put_money_in_safe/taxpose_tc/eval_rlbench.yaml num_trials=100 policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-gj3h3o3c:v0 wandb.group=rlbench_put_money_in_safe

# push_button

./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/push_button/taxpose_tc/eval_rlbench.yaml num_trials=100 policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-mmc5fhzu:v0 wandb.group=rlbench_push_button

# reach_target

./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/reach_target/taxpose_tc/eval_rlbench.yaml num_trials=100 policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-v8vv53tx:v0 wandb.group=rlbench_reach_target

# slide_block_to_target

./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/slide_block_to_target/taxpose_tc/eval_rlbench.yaml num_trials=100 policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-sejd7pz0:v0 wandb.group=rlbench_slide_block_to_target

# stack_wine

./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/stack_wine/taxpose_tc/eval_rlbench.yaml num_trials=100 policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-3hyo3r7q:v0 wandb.group=rlbench_stack_wine

# take_money_out_safe

./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/take_money_out_safe/taxpose_tc/eval_rlbench.yaml num_trials=100 policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-69gka1ew:v0 wandb.group=rlbench_take_money_out_safe

# take_umbrella_out_of_umbrella_stand

./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/take_umbrella_out_of_umbrella_stand/taxpose_tc/eval_rlbench.yaml num_trials=100 policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-b48mz8e1:v0 wandb.group=rlbench_take_umbrella_out_of_umbrella_stand
