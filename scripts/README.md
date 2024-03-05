# Scripts for your project

If you write some scripts which are meant to be run stand-alone, and not imported as part of the library, put them in this directory.


## Full pretraining.

### Vector neurons.
./launch_autobot.sh 0 python scripts/pretrain_embedding.py --config-name=commands/pretrain_vn_dgcnn_bottle wandb.group=pretrain_vn
./launch_autobot.sh 1 python scripts/pretrain_embedding.py --config-name=commands/pretrain_vn_dgcnn_bowl wandb.group=pretrain_vn
./launch_autobot.sh 2 python scripts/pretrain_embedding.py --config-name=commands/pretrain_vn_dgcnn_gripper wandb.group=pretrain_vn
./launch_autobot.sh 3 python scripts/pretrain_embedding.py --config-name=commands/pretrain_vn_dgcnn_mug wandb.group=pretrain_vn
./launch_autobot.sh 4 python scripts/pretrain_embedding.py --config-name=commands/pretrain_vn_dgcnn_rack wandb.group=pretrain_vn
./launch_autobot.sh 5 python scripts/pretrain_embedding.py --config-name=commands/pretrain_vn_dgcnn_table wandb.group=pretrain_vn

### Normal DGCNN.
./launch_autobot.sh 0 python scripts/pretrain_embedding.py --config-name=commands/pretrain_dgcnn_gripper wandb.group=pretrain
./launch_autobot.sh 1 python scripts/pretrain_embedding.py --config-name=commands/pretrain_dgcnn_mug wandb.group=pretrain
./launch_autobot.sh 2 python scripts/pretrain_embedding.py --config-name=commands/pretrain_dgcnn_rack wandb.group=pretrain
./launch_autobot.sh 3 python scripts/pretrain_embedding.py --config-name=commands/pretrain_dgcnn_table wandb.group=pretrain
./launch_autobot.sh 4 python scripts/pretrain_embedding.py --config-name=commands/pretrain_dgcnn_bowl wandb.group=pretrain
./launch_autobot.sh 5 python scripts/pretrain_embedding.py --config-name=commands/pretrain_dgcnn_bottle wandb.group=pretrain


## Full training.

### Taxpose.
./launch_autobot.sh 1 python scripts/train_residual_flow.py --config-name commands/train_taxpose_mug_grasp +mode=train wandb.group=taxpose
./launch_autobot.sh 2 python scripts/train_residual_flow.py --config-name commands/train_taxpose_mug_place +mode=train wandb.group=taxpose
./launch_autobot.sh 4 python scripts/train_residual_flow.py --config-name commands/train_taxpose_bottle_grasp +mode=train wandb.group=taxpose
./launch_autobot.sh 5 python scripts/train_residual_flow.py --config-name commands/train_taxpose_bottle_place +mode=train wandb.group=taxpose
./launch_autobot.sh 6 python scripts/train_residual_flow.py --config-name commands/train_taxpose_bowl_grasp +mode=train wandb.group=taxpose
./launch_autobot.sh 7 python scripts/train_residual_flow.py --config-name commands/train_taxpose_bowl_place +mode=train wandb.group=taxpose

### Multilateration.
./launch_autobot.sh 1 python scripts/train_residual_flow.py --config-name commands/train_mlat_mug_grasp +mode=train wandb.group=mlat_s256_vnn
./launch_autobot.sh 2 python scripts/train_residual_flow.py --config-name commands/train_mlat_mug_place +mode=train wandb.group=mlat_s256_vnn
./launch_autobot.sh 4 python scripts/train_residual_flow.py --config-name commands/train_mlat_bottle_grasp +mode=train wandb.group=mlat_s256_vnn
./launch_autobot.sh 5 python scripts/train_residual_flow.py --config-name commands/train_mlat_bottle_place +mode=train wandb.group=mlat_s256_vnn
./launch_autobot.sh 6 python scripts/train_residual_flow.py --config-name commands/train_mlat_bowl_grasp +mode=train wandb.group=mlat_s256_vnn
./launch_autobot.sh 7 python scripts/train_residual_flow.py --config-name commands/train_mlat_bowl_place +mode=train wandb.group=mlat_s256_vnn


## Ablations

### SE(3)

#### TAX-pose

./launch_autobot.sh 0 python scripts/train_residual_flow.py --config-name commands/ablations/se3_augmentation/train_taxpose_mug_grasp.yaml +mode=train wandb.group=ablate_se3_taxpose

./launch_autobot.sh 1 python scripts/train_residual_flow.py --config-name commands/ablations/se3_augmentation/train_taxpose_mug_place.yaml +mode=train wandb.group=ablate_se3_taxpose


####

./launch_autobot.sh 2 python scripts/train_residual_flow.py --config-name commands/ablations/se3_augmentation/train_mlat_mug_grasp.yaml +mode=train wandb.group=ablate_se3_mlat_s256_vnn

./launch_autobot.sh 3 python scripts/train_residual_flow.py --config-name commands/ablations/se3_augmentation/train_mlat_mug_place.yaml +mode=train wandb.group=ablate_se3_mlat_s256_vnn


## RLBench

### Taxpose

./launch_autobot.sh 0 python scripts/train_residual_flow.py --config-name commands/rlbench/train_taxpose_rlbench_stack_wine_place.yaml wandb.group=rlbench_taxpose
./launch_autobot.sh 1 python scripts/train_residual_flow.py --config-name commands/rlbench/train_taxpose_rlbench_insert_onto_square_peg_place.yaml wandb.group=rlbench_taxpose
./launch_autobot.sh 2 python scripts/train_residual_flow.py --config-name commands/rlbench/train_taxpose_rlbench_phone_on_base_place.yaml wandb.group=rlbench_taxpose
./launch_autobot.sh 3 python scripts/train_residual_flow.py --config-name commands/rlbench/train_taxpose_rlbench_put_toilet_roll_on_stand_place.yaml wandb.group=rlbench_taxpose
./launch_autobot.sh 4 python scripts/train_residual_flow.py --config-name commands/rlbench/train_taxpose_rlbench_solve_puzzle_place.yaml wandb.group=rlbench_taxpose
./launch_autobot.sh 5 python scripts/train_residual_flow.py --config-name commands/rlbench/train_taxpose_rlbench_place_hanger_on_rack_place.yaml wandb.group=rlbench_taxpose

./launch.sh ${RPAD_PLATFORM} 0 python scripts/train_residual_flow.py --config-name commands/rlbench/put_toilet_roll_on_stand/train_taxpose_grasp.yaml wandb.group=rlbench_taxpose
./launch.sh ${RPAD_PLATFORM} 1 python scripts/train_residual_flow.py --config-name commands/rlbench/phone_on_base/train_taxpose_grasp.yaml wandb.group=rlbench_taxpose
./launch.sh ${RPAD_PLATFORM} 2 python scripts/train_residual_flow.py --config-name commands/rlbench/place_hanger_on_rack/train_taxpose_grasp.yaml wandb.group=rlbench_taxpose
./launch.sh ${RPAD_PLATFORM} 3 python scripts/train_residual_flow.py --config-name commands/rlbench/stack_wine/train_taxpose_grasp.yaml wandb.group=rlbench_taxpose
./launch.sh ${RPAD_PLATFORM} 0 python scripts/train_residual_flow.py --config-name commands/rlbench/insert_onto_square_peg/train_taxpose_grasp.yaml wandb.group=rlbench_taxpose

### Mlat

./launch_autobot.sh 0 python scripts/train_residual_flow.py --config-name commands/rlbench/train_mlat_rlbench_stack_wine_place.yaml wandb.group=rlbench_mlat
./launch_autobot.sh 1 python scripts/train_residual_flow.py --config-name commands/rlbench/train_mlat_rlbench_insert_onto_square_peg_place.yaml wandb.group=rlbench_mlat
./launch_autobot.sh 2 python scripts/train_residual_flow.py --config-name commands/rlbench/train_mlat_rlbench_phone_on_base_place.yaml wandb.group=rlbench_mlat
./launch_autobot.sh 3 python scripts/train_residual_flow.py --config-name commands/rlbench/train_mlat_rlbench_put_toilet_roll_on_stand_place.yaml wandb.group=rlbench_mlat
./launch_autobot.sh 4 python scripts/train_residual_flow.py --config-name commands/rlbench/train_mlat_rlbench_solve_puzzle_place.yaml wandb.group=rlbench_mlat resources.num_workers=0
./launch_autobot.sh 5 python scripts/train_residual_flow.py --config-name commands/rlbench/train_mlat_rlbench_place_hanger_on_rack_place.yaml wandb.group=rlbench_mlat resources.num_workers=0


### Ablations

#### Taxpose
python scripts/train_residual_flow.py --config-name commands/rlbench/ablations/sample_efficiency/train_taxpose_stack_wine_place_1 wandb.group=rlbench_ablations_taxpose
python scripts/train_residual_flow.py --config-name commands/rlbench/ablations/sample_efficiency/train_taxpose_stack_wine_place_5 wandb.group=rlbench_ablations_taxpose
python scripts/train_residual_flow.py --config-name commands/rlbench/ablations/sample_efficiency/train_taxpose_stack_wine_place_10 wandb.group=rlbench_ablations_taxpose

#### Mlat
./launch_autobot.sh 0 python scripts/train_residual_flow.py --config-name commands/rlbench/ablations/sample_efficiency/train_mlat_stack_wine_place_1 wandb.group=rlbench_ablations_mlat resources.num_workers=0
./launch_autobot.sh 1 python scripts/train_residual_flow.py --config-name commands/rlbench/ablations/sample_efficiency/train_mlat_stack_wine_place_5 wandb.group=rlbench_ablations_mlat
./launch_autobot.sh 2 python scripts/train_residual_flow.py --config-name commands/rlbench/ablations/sample_efficiency/train_mlat_stack_wine_place_10 wandb.group=rlbench_ablations_mlat

##### Evals

python scripts/eval_metrics.py --config-name=commands/rlbench/ablations/sample_efficiency/evals/taxpose_stack_wine_place_1 wandb.group=rlbench_ablations_taxpose
python scripts/eval_metrics.py --config-name=commands/rlbench/ablations/sample_efficiency/evals/taxpose_stack_wine_place_5 wandb.group=rlbench_ablations_taxpose
python scripts/eval_metrics.py --config-name=commands/rlbench/ablations/sample_efficiency/evals/taxpose_stack_wine_place_10 wandb.group=rlbench_ablations_taxpose

python scripts/eval_metrics.py --config-name=commands/rlbench/ablations/sample_efficiency/evals/mlat_stack_wine_place_1 wandb.group=rlbench_ablations_mlat
python scripts/eval_metrics.py --config-name=commands/rlbench/ablations/sample_efficiency/evals/mlat_stack_wine_place_5 wandb.group=rlbench_ablations_mlat
python scripts/eval_metrics.py --config-name=commands/rlbench/ablations/sample_efficiency/evals/mlat_stack_wine_place_10 wandb.group=rlbench_ablations_mlat

## RLBench Evals

### Taxpose
python scripts/eval_metrics.py --config-name=commands/rlbench/evals/taxpose_stack_wine_place wandb.group=rlbench_taxpose
python scripts/eval_metrics.py --config-name=commands/rlbench/evals/taxpose_put_toilet_roll_on_stand_place wandb.group=rlbench_taxpose
python scripts/eval_metrics.py --config-name=commands/rlbench/evals/taxpose_place_hanger_on_rack_place wandb.group=rlbench_taxpose
python scripts/eval_metrics.py --config-name=commands/rlbench/evals/taxpose_phone_on_base_place wandb.group=rlbench_taxpose
python scripts/eval_metrics.py --config-name=commands/rlbench/evals/taxpose_insert_onto_square_peg_place wandb.group=rlbench_taxpose

### Mlat
python scripts/eval_metrics.py --config-name=commands/rlbench/evals/mlat_stack_wine_place wandb.group=rlbench_mlat
python scripts/eval_metrics.py --config-name=commands/rlbench/evals/mlat_put_toilet_roll_on_stand_place wandb.group=rlbench_mlat
python scripts/eval_metrics.py --config-name=commands/rlbench/evals/mlat_place_hanger_on_rack_place wandb.group=rlbench_mlat
python scripts/eval_metrics.py --config-name=commands/rlbench/evals/mlat_phone_on_base_place wandb.group=rlbench_mlat
python scripts/eval_metrics.py --config-name=commands/rlbench/evals/mlat_insert_onto_square_peg_place wandb.group=rlbench_mlat


## NDF evals.
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0.0 python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf_evals/taxpose_mug_upright seed=10 wandb.group=taxpose
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0.0 python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf_evals/taxpose_mug_arbitrary seed=10 wandb.group=taxpose
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0.0 python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf_evals/taxpose_bottle_upright seed=10 wandb.group=taxpose
CUDA_VISIBLE_DEVICES=1 DISPLAY=:0.0 python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf_evals/taxpose_bottle_arbitrary seed=10 wandb.group=taxpose
CUDA_VISIBLE_DEVICES=1 DISPLAY=:0.0 python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf_evals/taxpose_bowl_upright seed=10 wandb.group=taxpose
CUDA_VISIBLE_DEVICES=1 DISPLAY=:0.0 python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf_evals/taxpose_bowl_arbitrary seed=10 wandb.group=taxpose

CUDA_VISIBLE_DEVICES=0 DISPLAY=:0.0 python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf_evals/mlat_mug_upright seed=10 wandb.group=mlat_s256_vnn
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0.0 python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf_evals/mlat_mug_arbitrary seed=10 wandb.group=mlat_s256_vnn
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0.0 python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf_evals/mlat_bottle_upright seed=10 wandb.group=mlat_s256_vnn
CUDA_VISIBLE_DEVICES=1 DISPLAY=:0.0 python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf_evals/mlat_bottle_arbitrary seed=10 wandb.group=mlat_s256_vnn
CUDA_VISIBLE_DEVICES=1 DISPLAY=:0.0 python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf_evals/mlat_bowl_upright seed=10 wandb.group=mlat_s256_vnn
CUDA_VISIBLE_DEVICES=1 DISPLAY=:0.0 python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf_evals/mlat_bowl_arbitrary seed=10 wandb.group=mlat_s256_vnn

# New RLBench.

## Stack Wine.

./launch.sh local 0 python scripts/train_residual_flow.py --config-name commands/rlbench/stack_wine/train_taxpose_pregrasp.yaml wandb.group=rlbench_taxpose
./launch.sh local 0 python scripts/train_residual_flow.py --config-name commands/rlbench/stack_wine/train_taxpose_grasp.yaml wandb.group=rlbench_taxpose
./launch.sh local 1 python scripts/train_residual_flow.py --config-name commands/rlbench/stack_wine/train_taxpose_lift.yaml wandb.group=rlbench_taxpose
./launch.sh local 1 python scripts/train_residual_flow.py --config-name commands/rlbench/stack_wine/train_taxpose_preplace.yaml wandb.group=rlbench_taxpose
./launch.sh local 1 python scripts/train_residual_flow.py --config-name commands/rlbench/stack_wine/train_taxpose_place.yaml wandb.group=rlbench_taxpose

## Put knife on chopping board

./launch.sh local 0 python scripts/train_residual_flow.py --config-name commands/rlbench/put_knife_on_chopping_board/train_taxpose_pregrasp.yaml wandb.group=rlbench_put_knife_on_chopping_board
./launch.sh local 0 python scripts/train_residual_flow.py --config-name commands/rlbench/put_knife_on_chopping_board/train_taxpose_grasp.yaml wandb.group=rlbench_put_knife_on_chopping_board
./launch.sh local 1 python scripts/train_residual_flow.py --config-name commands/rlbench/put_knife_on_chopping_board/train_taxpose_lift.yaml wandb.group=rlbench_put_knife_on_chopping_board
./launch.sh local 1 python scripts/train_residual_flow.py --config-name commands/rlbench/put_knife_on_chopping_board/train_taxpose_place.yaml wandb.group=rlbench_put_knife_on_chopping_board
./launch.sh local 1 python scripts/train_residual_flow.py --config-name commands/rlbench/put_knife_on_chopping_board/train_taxpose_place.yaml wandb.group=all

### Eval.


#### TAXPOSE
python scripts/eval_metrics.py --config-name commands/rlbench/put_knife_on_chopping_board/eval_taxpose_pregrasp data_root=/data wandb.group=rlbench_put_knife_on_chopping_board_taxpose
python scripts/eval_metrics.py --config-name commands/rlbench/put_knife_on_chopping_board/eval_taxpose_grasp data_root=/data wandb.group=rlbench_put_knife_on_chopping_board_taxpose
python scripts/eval_metrics.py --config-name commands/rlbench/put_knife_on_chopping_board/eval_taxpose_lift data_root=/data wandb.group=rlbench_put_knife_on_chopping_board_taxpose dm.train_dset.anchor_rotation_variance=1e-5 dm.train_dset.action_rot_sample_method=axis_angle_uniform_z dm.train_dset.anchor_rot_sample_method=axis_angle_uniform_z
python scripts/eval_metrics.py --config-name commands/rlbench/put_knife_on_chopping_board/eval_taxpose_place data_root=/data wandb.group=rlbench_put_knife_on_chopping_board_taxpose dm.train_dset.anchor_rotation_variance=1e-5 dm.train_dset.action_rot_sample_method=axis_angle_uniform_z dm.train_dset.anchor_rot_sample_method=axis_angle_uniform_z

#### TAXPOSE-ALL
python scripts/eval_metrics.py --config-name commands/rlbench/put_knife_on_chopping_board/eval_taxpose_all_pregrasp data_root=/data wandb.group=rlbench_put_knife_on_chopping_board_taxpose_all dm.train_dset.anchor_rotation_variance=1e-5 dm.train_dset.action_rot_sample_method=axis_angle_uniform_z dm.train_dset.anchor_rot_sample_method=axis_angle_uniform_z
python scripts/eval_metrics.py --config-name commands/rlbench/put_knife_on_chopping_board/eval_taxpose_all_grasp data_root=/data wandb.group=rlbench_put_knife_on_chopping_board_taxpose_all dm.train_dset.anchor_rotation_variance=1e-5 dm.train_dset.action_rot_sample_method=axis_angle_uniform_z dm.train_dset.anchor_rot_sample_method=axis_angle_uniform_z
python scripts/eval_metrics.py --config-name commands/rlbench/put_knife_on_chopping_board/eval_taxpose_all_lift data_root=/data wandb.group=rlbench_put_knife_on_chopping_board_taxpose_all dm.train_dset.anchor_rotation_variance=1e-5 dm.train_dset.action_rot_sample_method=axis_angle_uniform_z dm.train_dset.anchor_rot_sample_method=axis_angle_uniform_z
python scripts/eval_metrics.py --config-name commands/rlbench/put_knife_on_chopping_board/eval_taxpose_all_place data_root=/data wandb.group=rlbench_put_knife_on_chopping_board_taxpose_all dm.train_dset.anchor_rotation_variance=1e-5 dm.train_dset.action_rot_sample_method=axis_angle_uniform_z dm.train_dset.anchor_rot_sample_method=axis_angle_uniform_z

### RLBench Eval.
taskset -c 0-40 python scripts/eval_rlbench.py --config-name commands/rlbench/put_knife_on_chopping_board/eval_rlbench_taxpose_all num_trials=100 resources.num_workers=10 wandb.group=rlbench_put_knife_on_chopping_board headless=True

## Reach Target

### Train

./launch.sh local 1 python scripts/train_residual_flow.py --config-name commands/rlbench/reach_target/train_taxpose_all.yaml wandb.group=rlbench_reach_target

### Eval
python scripts/eval_metrics.py --config-name commands/rlbench/reach_target/eval_taxpose_all_reach data_root=/data wandb.group=rlbench_reach_target

### RLBench Eval
taskset -c 0-40 python scripts/eval_rlbench.py --config-name commands/rlbench/reach_target/eval_rlbench_taxpose_all num_trials=100 resources.num_workers=10 wandb.group=rlbench_reach_target headless=True



## RLBench Single eval.
<!-- - [ ] "pick_and_lift",
- [ ] "pick_up_cup",
- [ ] "put_knife_on_chopping_board",
- [ ] "put_money_in_safe",
- [ ] "push_button",
- [ ] "reach_target",
- [ ] "slide_block_to_target",
- [ ] "stack_wine",
- [ ] "take_money_out_safe",
- [ ] "take_umbrella_out_of_umbrella_stand",
 -->
WANDB_MODE=disabled taskset -c 0-50 python scripts/eval_rlbench.py --config-name commands/rlbench/pick_and_lift/taxpose_all/eval_rlbench.yaml num_trials=1 resources.num_workers=0 wandb.group=rlbench_pick_and_lift headless=True

WANDB_MODE=disabled taskset -c 0-50 python scripts/eval_rlbench.py --config-name commands/rlbench/pick_up_cup/taxpose_all/eval_rlbench.yaml num_trials=1 resources.num_workers=0 wandb.group=rlbench_pick_up_cup headless=True

WANDB_MODE=disabled taskset -c 0-50 python scripts/eval_rlbench.py --config-name commands/rlbench/put_knife_on_chopping_board/taxpose_all/eval_rlbench.yaml num_trials=1 resources.num_workers=0 wandb.group=rlbench_put_knife_on_chopping_board headless=True

WANDB_MODE=disabled taskset -c 0-50 python scripts/eval_rlbench.py --config-name commands/rlbench/put_money_in_safe/taxpose_all/eval_rlbench.yaml num_trials=1 resources.num_workers=0 wandb.group=rlbench_put_money_in_safe headless=True

WANDB_MODE=disabled taskset -c 0-50 python scripts/eval_rlbench.py --config-name commands/rlbench/push_button/taxpose_all/eval_rlbench.yaml num_trials=1 resources.num_workers=0 wandb.group=rlbench_push_button headless=True

WANDB_MODE=disabled taskset -c 0-50 python scripts/eval_rlbench.py --config-name commands/rlbench/reach_target/taxpose_all/eval_rlbench.yaml num_trials=1 resources.num_workers=0 wandb.group=rlbench_reach_target headless=True

WANDB_MODE=disabled taskset -c 0-50 python scripts/eval_rlbench.py --config-name commands/rlbench/slide_block_to_target/taxpose_all/eval_rlbench.yaml num_trials=1 resources.num_workers=0 wandb.group=rlbench_slide_block_to_target headless=True

WANDB_MODE=disabled taskset -c 0-50 python scripts/eval_rlbench.py --config-name commands/rlbench/stack_wine/taxpose_all/eval_rlbench.yaml num_trials=1 resources.num_workers=0 wandb.group=rlbench_stack_wine headless=True

WANDB_MODE=disabled taskset -c 0-50 python scripts/eval_rlbench.py --config-name commands/rlbench/take_money_out_safe/taxpose_all/eval_rlbench.yaml num_trials=1 resources.num_workers=0 wandb.group=rlbench_take_money_out_safe headless=True

WANDB_MODE=disabled taskset -c 0-50 python scripts/eval_rlbench.py --config-name commands/rlbench/take_umbrella_out_of_umbrella_stand/taxpose_all/eval_rlbench.yaml num_trials=1 resources.num_workers=0 wandb.group=rlbench_take_umbrella_out_of_umbrella_stand headless=True


### Try using whole scene

WANDB_MODE=disabled python scripts/eval_metrics.py --config-name commands/rlbench/put_knife_on_chopping_board/taxpose_all/precision_eval/lift.yaml data_root=/data wandb.group=rlbench_put_knife_on_chopping_board dm.train_dset.demo_dset.anchor_mode="raw" checkpoints.ckpt_file=r-pad/taxpose/model-2t9piqya:v0


### Try various partial scenes...
./launch.sh local 1 python scripts/train_residual_flow.py --config-name commands/rlbench/put_knife_on_chopping_board/train_taxpose_all.yaml dm.train_dset.demo_dset.anchor_mode="background_robot_removed" wandb.group=rlbench_reach_target



./launch.sh local 1 python scripts/train_residual_flow.py --config-name commands/rlbench/put_knife_on_chopping_board/train_taxpose_all.yaml dm.train_dset.demo_dset.anchor_mode="background_robot_removed" dm.train_dset.demo_dset.action_mode="gripper_and_object" dm.train_dset.demo_dset.num_points=512  wandb.group=rlbench_reach_target



# Cool thing I'm trying...

<!--
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
] -->

<!-- pick_and_lift -->
mkdir -p configs/checkpoints/rlbench/pick_and_lift/pretraining && touch configs/checkpoints/rlbench/pick_and_lift/pretraining/all.yaml
./scripts/train_eval.sh \
    "./launch.sh autobot 0 python scripts/train_residual_flow.py --config-name commands/rlbench/pick_and_lift/train_taxpose_all.yaml dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512 wandb.group=rlbench_pick_and_lift resources.num_workers=20"  \
    "./launch.sh autobot 0 ./configs/commands/rlbench/pick_and_lift/taxpose_all/precision_eval/precision_eval.sh dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512" \
    echo

<!-- pick_up_cup -->
mkdir -p configs/checkpoints/rlbench/pick_up_cup/pretraining && touch configs/checkpoints/rlbench/pick_up_cup/pretraining/all.yaml
./scripts/train_eval.sh \
    "./launch.sh autobot 1 python scripts/train_residual_flow.py --config-name commands/rlbench/pick_up_cup/train_taxpose_all.yaml dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512 wandb.group=rlbench_pick_up_cup resources.num_workers=20"  \
    "./launch.sh autobot 1 ./configs/commands/rlbench/pick_up_cup/taxpose_all/precision_eval/precision_eval.sh dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512" \
    echo

<!-- put_knife_on_chopping_board -->
mkdir -p configs/checkpoints/rlbench/put_knife_on_chopping_board/pretraining && touch configs/checkpoints/rlbench/put_knife_on_chopping_board/pretraining/all.yaml
./scripts/train_eval.sh \
    "./launch.sh autobot 2 python scripts/train_residual_flow.py --config-name commands/rlbench/put_knife_on_chopping_board/train_taxpose_all.yaml dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512 wandb.group=rlbench_put_knife_on_chopping_board resources.num_workers=20"  \
    "./launch.sh autobot 2 ./configs/commands/rlbench/put_knife_on_chopping_board/taxpose_all/precision_eval/precision_eval.sh dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512" \
    echo

<!-- put_money_in_safe -->
mkdir -p configs/checkpoints/rlbench/put_money_in_safe/pretraining && touch configs/checkpoints/rlbench/put_money_in_safe/pretraining/all.yaml
./scripts/train_eval.sh \
    "./launch.sh autobot 3 python scripts/train_residual_flow.py --config-name commands/rlbench/put_money_in_safe/train_taxpose_all.yaml dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512 wandb.group=rlbench_put_money_in_safe resources.num_workers=20"  \
    "./launch.sh autobot 3 ./configs/commands/rlbench/put_money_in_safe/taxpose_all/precision_eval/precision_eval.sh dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512" \
    echo

<!-- push_button -->
mkdir -p configs/checkpoints/rlbench/push_button/pretraining && touch configs/checkpoints/rlbench/push_button/pretraining/all.yaml
./scripts/train_eval.sh \
    "./launch.sh autobot 4 python scripts/train_residual_flow.py --config-name commands/rlbench/push_button/train_taxpose_all.yaml dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512 wandb.group=rlbench_push_button resources.num_workers=20"  \
    "./launch.sh autobot 4 ./configs/commands/rlbench/push_button/taxpose_all/precision_eval/precision_eval.sh dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512" \
    echo

<!-- reach_target -->
mkdir -p configs/checkpoints/rlbench/reach_target/pretraining && touch configs/checkpoints/rlbench/reach_target/pretraining/all.yaml
./scripts/train_eval.sh \
    "./launch.sh autobot 5 python scripts/train_residual_flow.py --config-name commands/rlbench/reach_target/train_taxpose_all.yaml dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512 wandb.group=rlbench_reach_target resources.num_workers=20"  \
    "./launch.sh autobot 5 ./configs/commands/rlbench/reach_target/taxpose_all/precision_eval/precision_eval.sh dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512" \
    echo

<!-- slide_block_to_target -->
mkdir -p configs/checkpoints/rlbench/slide_block_to_target/pretraining && touch configs/checkpoints/rlbench/slide_block_to_target/pretraining/all.yaml
./scripts/train_eval.sh \
    "./launch.sh autobot 6 python scripts/train_residual_flow.py --config-name commands/rlbench/slide_block_to_target/train_taxpose_all.yaml dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512 wandb.group=rlbench_slide_block_to_target resources.num_workers=20"  \
    "./launch.sh autobot 6 ./configs/commands/rlbench/slide_block_to_target/taxpose_all/precision_eval/precision_eval.sh dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512" \
    echo

<!-- stack_wine -->
mkdir -p configs/checkpoints/rlbench/stack_wine/pretraining && touch configs/checkpoints/rlbench/stack_wine/pretraining/all.yaml
./scripts/train_eval.sh \
    "./launch.sh autobot 7 python scripts/train_residual_flow.py --config-name commands/rlbench/stack_wine/train_taxpose_all.yaml dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512 wandb.group=rlbench_stack_wine resources.num_workers=20"  \
    "./launch.sh autobot 7 ./configs/commands/rlbench/stack_wine/taxpose_all/precision_eval/precision_eval.sh dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512" \
    echo

<!-- take_money_out_safe -->
./scripts/train_eval.sh \
    "./launch.sh local 0 python scripts/train_residual_flow.py --config-name commands/rlbench/take_money_out_safe/train_taxpose_all.yaml dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512 wandb.group=rlbench_take_money_out_safe resources.num_workers=20 data_root=/home/beisner/datasets"  \
    "./launch.sh local 0 ./configs/commands/rlbench/take_money_out_safe/taxpose_all/precision_eval/precision_eval.sh dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512 data_root=/home/beisner/datasets" \
    echo

<!-- take_umbrella_out_of_umbrella_stand -->
./scripts/train_eval.sh \
    "./launch.sh local 1 python scripts/train_residual_flow.py --config-name commands/rlbench/take_umbrella_out_of_umbrella_stand/train_taxpose_all.yaml dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512 wandb.group=rlbench_take_umbrella_out_of_umbrella_stand resources.num_workers=20 data_root=/home/beisner/datasets"  \
    "./launch.sh local 1 ./configs/commands/rlbench/take_umbrella_out_of_umbrella_stand/taxpose_all/precision_eval/precision_eval.sh dm.train_dset.demo_dset.anchor_mode=background_robot_removed dm.train_dset.demo_dset.action_mode=gripper_and_object dm.train_dset.demo_dset.num_points=512 data_root=/home/beisner/datasets" \
    echo
