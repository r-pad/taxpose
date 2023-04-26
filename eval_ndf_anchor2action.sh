declare -i m=2
# list_model_name=("21wtib15" "3en3fbxe" "2rt6k4ra")
 
# for model_name in ${list_model_name[@]}
for i in {6..10}
    do
        sleep 5 &
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_ndf_mug.py grasp_model=su58zs1b post_dist=upright seed=$i log_txt_file='/home/exx/Documents/taxpose/search_new_anchor2action_models.txt' &
    done
    wait  

for i in {6..10}
    do
        sleep 5 &
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_ndf_mug.py grasp_model=gah4koz1_127 seed=$i log_txt_file='/home/exx/Documents/taxpose/search_new_anchor2action_models.txt' &
    done
    wait 


 