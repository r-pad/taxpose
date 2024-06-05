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

# pick_and_lift: 8pyp8zaw
# pick_up_cup: 9m4fflcx
# push_button: mmc5fhzu
# put_knife_on_chopping_board: rjscih24
# put_money_in_safe: tpuvxzyg
# reach_target: ffp30alr
# slide_block_to_target: sejd7pz0
# stack_wine: 3hyo3r7q
# take_money_out_safe: u4bpi2bf
# take_umbrella_out_of_umbrella_stand: b48mz8e1

# pick_and_lift

NUM_TRIALS=500


# # pick_up_cup

# ./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/pick_up_cup/taxpose_tc/eval_rlbench.yaml num_trials=$NUM_TRIALS policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-9m4fflcx:v0 wandb.group=rlbench_pick_up_cup

# # put_knife_on_chopping_board

# ./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/put_knife_on_chopping_board/taxpose_tc/eval_rlbench.yaml num_trials=$NUM_TRIALS policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-rjscih24:v0 wandb.group=rlbench_put_knife_on_chopping_board

# # put_money_in_safe

# ./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/put_money_in_safe/taxpose_tc/eval_rlbench.yaml num_trials=$NUM_TRIALS policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-tpuvxzyg:v0 wandb.group=rlbench_put_money_in_safe

# # push_button

# ./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/push_button/taxpose_tc/eval_rlbench.yaml num_trials=$NUM_TRIALS policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-mmc5fhzu:v0 wandb.group=rlbench_push_button

# # reach_target

# ./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/reach_target/taxpose_tc/eval_rlbench.yaml num_trials=$NUM_TRIALS policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-ffp30alr:v0 wandb.group=rlbench_reach_target

# # slide_block_to_target

# ./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/slide_block_to_target/taxpose_tc/eval_rlbench.yaml num_trials=$NUM_TRIALS policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-sejd7pz0:v0 wandb.group=rlbench_slide_block_to_target

# # stack_wine

# ./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/stack_wine/taxpose_tc/eval_rlbench.yaml num_trials=$NUM_TRIALS policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-3hyo3r7q:v0 wandb.group=rlbench_stack_wine

# # take_money_out_safe

# ./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/take_money_out_safe/taxpose_tc/eval_rlbench.yaml num_trials=$NUM_TRIALS policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-u4bpi2bf:v0 wandb.group=rlbench_take_money_out_safe

# # take_umbrella_out_of_umbrella_stand

# ./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/take_umbrella_out_of_umbrella_stand/taxpose_tc/eval_rlbench.yaml num_trials=$NUM_TRIALS policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-b48mz8e1:v0 wandb.group=rlbench_take_umbrella_out_of_umbrella_stand

# ./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/pick_and_lift/taxpose_tc/eval_rlbench.yaml num_trials=$NUM_TRIALS policy_spec.include_rgb_features=True model.feature_channels=3 checkpoints.ckpt_file=r-pad/taxpose/model-8pyp8zaw:v0 wandb.group=rlbench_pick_and_lift

########################### No gripper, wrist cam, trained for 10x #############################

# pick_and_lift: 0kaohhnd
# pick_up_cup: wm2awuy3
# push_button: 8pjlad6b
# put_knife_on_chopping_board: qnrs3d24
# put_money_in_safe: 17ohuoyi
# reach_target: 58mr6cyt
# slide_block_to_target: 68eod49e
# stack_wine: k42u3jsg
# take_money_out_safe: ry299c41
# take_umbrella_out_of_umbrella_stand: uqslnslg


# pick_and_lift

# ./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/pick_and_lift/taxpose_tc/eval_rlbench.yaml num_trials=$NUM_TRIALS checkpoints.ckpt_file=r-pad/taxpose/model-0kaohhnd:v0 wandb.group=rlbench_pick_and_lift

# # pick_up_cup

# ./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/pick_up_cup/taxpose_tc/eval_rlbench.yaml num_trials=$NUM_TRIALS checkpoints.ckpt_file=r-pad/taxpose/model-wm2awuy3:v0 wandb.group=rlbench_pick_up_cup

# # put_knife_on_chopping_board

# ./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/put_knife_on_chopping_board/taxpose_tc/eval_rlbench.yaml num_trials=$NUM_TRIALS checkpoints.ckpt_file=r-pad/taxpose/model-qnrs3d24:v0 wandb.group=rlbench_put_knife_on_chopping_board

# # put_money_in_safe

# ./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/put_money_in_safe/taxpose_tc/eval_rlbench.yaml num_trials=$NUM_TRIALS checkpoints.ckpt_file=r-pad/taxpose/model-17ohuoyi:v0 wandb.group=rlbench_put_money_in_safe

# # push_button

# ./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/push_button/taxpose_tc/eval_rlbench.yaml num_trials=$NUM_TRIALS checkpoints.ckpt_file=r-pad/taxpose/model-8pjlad6b:v0 wandb.group=rlbench_push_button

# # reach_target

# ./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/reach_target/taxpose_tc/eval_rlbench.yaml num_trials=$NUM_TRIALS checkpoints.ckpt_file=r-pad/taxpose/model-58mr6cyt:v0 wandb.group=rlbench_reach_target

# # slide_block_to_target

# ./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/slide_block_to_target/taxpose_tc/eval_rlbench.yaml num_trials=$NUM_TRIALS checkpoints.ckpt_file=r-pad/taxpose/model-68eod49e:v0 wandb.group=rlbench_slide_block_to_target

# # stack_wine

# ./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/stack_wine/taxpose_tc/eval_rlbench.yaml num_trials=$NUM_TRIALS checkpoints.ckpt_file=r-pad/taxpose/model-k42u3jsg:v0 wandb.group=rlbench_stack_wine

# # take_money_out_safe

# ./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/take_money_out_safe/taxpose_tc/eval_rlbench.yaml num_trials=$NUM_TRIALS checkpoints.ckpt_file=r-pad/taxpose/model-ry299c41:v0 wandb.group=rlbench_take_money_out_safe

# # take_umbrella_out_of_umbrella_stand

# ./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/take_umbrella_out_of_umbrella_stand/taxpose_tc/eval_rlbench.yaml num_trials=$NUM_TRIALS checkpoints.ckpt_file=r-pad/taxpose/model-uqslnslg:v0 wandb.group=rlbench_take_umbrella_out_of_umbrella_stand

########################### Remove conditional vector #############################

# pick_and_lift: cralwmdq
# pick_up_cup: hrw54l6b
# push_button: qehsvvq3
# put_knife_on_chopping_board: qnzr4gu2
# put_money_in_safe: ilos0cn0
# reach_target: u6o56yqc
# slide_block_to_target: i9r6v1dw
# stack_wine: dazcchr3
# take_money_out_safe: 49cmezw7
# take_umbrella_out_of_umbrella_stand: cw90yjba

# pick_and_lift

./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/pick_and_lift/taxpose_tc/eval_rlbench.yaml num_trials=$NUM_TRIALS checkpoints.ckpt_file=r-pad/taxpose/model-cralwmdq:v0 wandb.group=rlbench_pick_and_lift model.conditional=False

# pick_up_cup

./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/pick_up_cup/taxpose_tc/eval_rlbench.yaml num_trials=$NUM_TRIALS checkpoints.ckpt_file=r-pad/taxpose/model-hrw54l6b:v0 wandb.group=rlbench_pick_up_cup model.conditional=False

# put_knife_on_chopping_board

./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/put_knife_on_chopping_board/taxpose_tc/eval_rlbench.yaml num_trials=$NUM_TRIALS checkpoints.ckpt_file=r-pad/taxpose/model-qnzr4gu2:v0 wandb.group=rlbench_put_knife_on_chopping_board model.conditional=False

# put_money_in_safe

./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/put_money_in_safe/taxpose_tc/eval_rlbench.yaml num_trials=$NUM_TRIALS checkpoints.ckpt_file=r-pad/taxpose/model-ilos0cn0:v0 wandb.group=rlbench_put_money_in_safe model.conditional=False

# push_button

./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/push_button/taxpose_tc/eval_rlbench.yaml num_trials=$NUM_TRIALS checkpoints.ckpt_file=r-pad/taxpose/model-qehsvvq3:v0 wandb.group=rlbench_push_button model.conditional=False

# reach_target

./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/reach_target/taxpose_tc/eval_rlbench.yaml num_trials=$NUM_TRIALS checkpoints.ckpt_file=r-pad/taxpose/model-u6o56yqc:v0 wandb.group=rlbench_reach_target model.conditional=False

# slide_block_to_target

./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/slide_block_to_target/taxpose_tc/eval_rlbench.yaml num_trials=$NUM_TRIALS checkpoints.ckpt_file=r-pad/taxpose/model-i9r6v1dw:v0 wandb.group=rlbench_slide_block_to_target model.conditional=False

# stack_wine

./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/stack_wine/taxpose_tc/eval_rlbench.yaml num_trials=$NUM_TRIALS checkpoints.ckpt_file=r-pad/taxpose/model-dazcchr3:v0 wandb.group=rlbench_stack_wine model.conditional=False

# take_money_out_safe

./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/take_money_out_safe/taxpose_tc/eval_rlbench.yaml num_trials=$NUM_TRIALS checkpoints.ckpt_file=r-pad/taxpose/model-49cmezw7:v0 wandb.group=rlbench_take_money_out_safe model.conditional=False

# take_umbrella_out_of_umbrella_stand

./launch.sh local-docker 0 python scripts/eval_rlbench.py --config-name commands/rlbench/take_umbrella_out_of_umbrella_stand/taxpose_tc/eval_rlbench.yaml num_trials=$NUM_TRIALS checkpoints.ckpt_file=r-pad/taxpose/model-cw90yjba:v0 wandb.group=rlbench_take_umbrella_out_of_umbrella_stand model.conditional=False
