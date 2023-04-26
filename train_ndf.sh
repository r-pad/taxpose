 
 
sleep 5 &
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/train_residual_flow.py max_epochs=250 ball_radius=0.1 plane_standoff=0.02 seed=0 synthetic_occlusion=True flow_supervision=anchor2action &

# sleep 5 &
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python scripts/train_residual_flow.py max_epochs=250 ball_radius=0.1 plane_standoff=0.02 seed=0 synthetic_occlusion=True flow_supervision=action2anchor & 

