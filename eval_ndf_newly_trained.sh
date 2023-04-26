declare -i m=2
# list_model_name=("21wtib15" "3en3fbxe" "2rt6k4ra")
 
# # for model_name in ${list_model_name[@]}
# for i in {2..5}
#     do
#         sleep 5 &
#         CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_ndf_mug.py grasp_model=kwl8hwnx_127 seed=$i
#     done
#     wait  
# pids=(100775 100777 100779)

# is_process_running() {
#     local pid=$1
#     ps -p $pid > /dev/null 2>&1
#     return $?
# }

# for pid in "${pids[@]}"; do
#     while is_process_running $pid; do   
#         sleep 1
#     done 
# done 

seed=(4 5 6)

for i in "${seed[@]}"
    do
        sleep 5 &
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_ndf_mug.py grasp_model=qr657ckg_250 pose_dist=upright seed=$i &
    done 
 

for i in "${seed[@]}"
    do
        sleep 5 &
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_ndf_mug.py grasp_model=qr657ckg_250 pose_dist=arbitrary seed=$i &
    done 
    wait
# sleep 5 &
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_ndf_mug.py grasp_model=qr657ckg_250 pose_dist=upright seed=3 &

# sleep 5 &
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_ndf_mug.py grasp_model=qr657ckg_250 pose_dist=upright seed=0 &

# sleep 5 &
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_ndf_mug.py grasp_model=qr657ckg_250 pose_dist=upright seed=1 &

# sleep 5 &
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_ndf_mug.py grasp_model=qr657ckg_127 pose_dist=upright seed=0 &

# sleep 5 &
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_ndf_mug.py grasp_model=7fiiadrg_175 pose_dist=upright seed=0 &
    # done
    # wait 


 