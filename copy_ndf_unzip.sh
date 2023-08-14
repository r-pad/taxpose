SOURCE_DIR=/project_data/held/baeisner/ndf
TARGET_DIR=/scratch/baeisner/ndf

# Copy files from source to target
rsync -av --progress $SOURCE_DIR $TARGET_DIR

output_dir=$TARGET_DIR

# Unzip files
if [ -d "$output_dir"/mug_grasp ]; then
  echo "mug_grasp data already downloaded."
else
    unzip "$output_dir"/mug_grasp/train_data.zip -d "$output_dir"/mug_grasp && rm -rf "$output_dir"/mug_grasp/train_data.zip
    unzip "$output_dir"/mug_grasp/test_data.zip -d "$output_dir"/mug_grasp && rm -rf "$output_dir"/mug_grasp/test_data.zip
fi

if [ -d "$output_dir"/mug_place ]; then
  echo "mug_place data already downloaded."
else
    unzip "$output_dir"/mug_place/train_data.zip -d "$output_dir"/mug_place && rm -rf "$output_dir"/mug_place/train_data.zip
    unzip "$output_dir"/mug_place/test_data.zip -d "$output_dir"/mug_place && rm -rf "$output_dir"/mug_place/test_data.zip
fi

if [ -d "$output_dir"/bowl_grasp ]; then
  echo "bowl_grasp data already downloaded."
else
    unzip "$output_dir"/bowl_grasp/train_data.zip -d "$output_dir"/bowl_grasp && rm -rf "$output_dir"/bowl_grasp/train_data.zip
    unzip "$output_dir"/bowl_grasp/test_data.zip -d "$output_dir"/bowl_grasp && rm -rf "$output_dir"/bowl_grasp/test_data.zip
fi

if [ -d "$output_dir"/bowl_place ]; then
  echo "bowl_place data already downloaded."
else
    unzip "$output_dir"/bowl_place/train_data.zip -d "$output_dir"/bowl_place && rm -rf "$output_dir"/bowl_place/train_data.zip
    unzip "$output_dir"/bowl_place/test_data.zip -d "$output_dir"/bowl_place && rm -rf "$output_dir"/bowl_place/test_data.zip
fi

if [ -d "$output_dir"/bottle_grasp ]; then
  echo "bottle_grasp data already downloaded."
else
    unzip "$output_dir"/bottle_grasp/train_data.zip -d "$output_dir"/bottle_grasp && rm -rf "$output_dir"/bottle_grasp/train_data.zip
    unzip "$output_dir"/bottle_grasp/test_data.zip -d "$output_dir"/bottle_grasp && rm -rf "$output_dir"/bottle_grasp/test_data.zip
fi

if [ -d "$output_dir"/bottle_place ]; then
  echo "bottle_place data already downloaded."
else
    unzip "$output_dir"/bottle_place/train_data.zip -d "$output_dir"/bottle_place && rm -rf "$output_dir"/bottle_place/train_data.zip
    unzip "$output_dir"/bottle_place/test_data.zip -d "$output_dir"/bottle_place && rm -rf "$output_dir"/bottle_place/test_data.zip
fi
