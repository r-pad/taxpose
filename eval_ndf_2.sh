declare -i m=2
# list_model_name=("21wtib15" "3en3fbxe" "2rt6k4ra")
 
# for model_name in ${list_model_name[@]}
for i in {0..10}
    do
        sleep 5 &
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_ndf_mug.py grasp_model=2rt6k4ra seed=$i log_txt_file='/home/exx/Documents/taxpose/search_existing_models_2rt6k4ra.txt'
    done
    wait  


 