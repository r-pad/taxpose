
list_model_name=("3en3fbxe")
declare -i m=2
# list_model_name=("21wtib15" "3en3fbxe" "2rt6k4ra")
 
# for model_name in ${list_model_name[@]}
for i in {0..3}
    do
        sleep 5 &
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_ndf_mug.py seed=$i &
    done
    wait  

touch finished
while [ ! -f finished ] ; do     
    sleep 5
done

for i in {4..7}
    do
        sleep 5 &
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_ndf_mug.py seed=$i &
    done
    wait  


# sleep 5 &
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/train_residual_flow.py max_epochs=250 ball_radius=0.1 plane_standoff=0.02 seed=0 synthetic_occlusion=True flow_supervision=both

