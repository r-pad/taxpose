
BOTTLE_DIR="bottle_table_all_pose_4_cam_half_occ_full_rand_scale"
MUG_DIR="mug_table_all_pose_4_cam_half_occ_full_rand_scale"
BOWL_DIR="bowl_table_all_pose_4_cam_half_occ_full_rand_scale"

SRC_FOLDER="/project_data/held/baeisner/ndf_original/data/training_data/"
DST_FOLDER="/scratch/baeisner/ndf_original/data/training_data/"

# Make sure destination folder exists
mkdir -p $DST_FOLDER

# If Bottle dir doesn't exist, use rsync to copy the tar file and unzip.
if [ ! -d "$DST_FOLDER/$BOTTLE_DIR" ]; then
    echo "Copying $BOTTLE_DIR"
    rsync --progress -avz $SRC_FOLDER/ndf_bottle_data.tar.gz $DST_FOLDER
    cd $DST_FOLDER
    tar -xzf ndf_bottle_data.tar.gz
    rm ndf_bottle_data.tar.gz
else
    echo "$DST_FOLDER/$BOTTLE_DIR already exists"
fi

# If Mug dir doesn't exist, use rsync to copy the tar file and unzip.
if [ ! -d "$DST_FOLDER/$MUG_DIR" ]; then
    echo "Copying $MUG_DIR"
    rsync --progress -avz $SRC_FOLDER/ndf_mug_data.tar.gz $DST_FOLDER
    cd $DST_FOLDER
    tar -xzf ndf_mug_data.tar.gz
    rm ndf_mug_data.tar.gz
else
    echo "$DST_FOLDER/$MUG_DIR already exists"
fi

# If Bowl dir doesn't exist, use rsync to copy the tar file and unzip.
if [ ! -d "$DST_FOLDER/$BOWL_DIR" ]; then
    echo "Copying $BOWL_DIR"
    rsync --progress -avz $SRC_FOLDER/ndf_bowl_data.tar.gz $DST_FOLDER
    cd $DST_FOLDER
    tar -xzf ndf_bowl_data.tar.gz
    rm ndf_bowl_data.tar.gz
else
    echo "$DST_FOLDER/$BOWL_DIR already exists"
fi
