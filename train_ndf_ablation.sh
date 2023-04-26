 
 
sleep 5 &
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/train_residual_flow_ablation.py ablation=0_no_disp_loss &

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/train_residual_flow_ablation.py ablation=8_mlp &
# sleep 5 &
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python scripts/train_residual_flow.py max_epochs=250 ball_radius=0.1 plane_standoff=0.02 seed=0 synthetic_occlusion=True flow_supervision=action2anchor & 

