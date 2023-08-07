# !/bin/bash

# Check to see if gdown is installed.
if ! [ -x "$(command -v gdown)" ]; then
  echo 'Error: gdown is not installed.' >&2
  exit 1
fi

# Add an option to specify the output directory.
if [ -z "$1" ]; then
  echo "No output directory specified. Downloading to current directory."
  output_dir="."
else
  output_dir="$1"
fi

# Create the output directory if it doesn't exist.
mkdir -p "$output_dir"


MUG_GRASP_TEST_URL="https://docs.google.com/uc?export=download&id=1GI33ZHKMB5yuZKDMkNhuUxOlEyrR68mR"
MUG_GRASP_TRAIN_URL="https://docs.google.com/uc?export=download&id=1yhRegjfRfzyNMEmOFxU2p2ZRdO57pAdy"

if [ -d "$output_dir"/mug_grasp ]; then
  echo "mug_grasp data already downloaded."
else
    mkdir "$output_dir"/mug_grasp
    gdown "$MUG_GRASP_TEST_URL" -O "$output_dir"/mug_grasp/test_data.zip
    gdown "$MUG_GRASP_TRAIN_URL" -O "$output_dir"/mug_grasp/train_data.zip
    unzip "$output_dir"/mug_grasp/train_data.zip -d "$output_dir"/mug_grasp && rm -rf "$output_dir"/mug_grasp/train_data.zip
    unzip "$output_dir"/mug_grasp/test_data.zip -d "$output_dir"/mug_grasp && rm -rf "$output_dir"/mug_grasp/test_data.zip
fi

MUG_PLACE_TEST_URL="https://docs.google.com/uc?export=download&id=1GcA5Owj6djlsVarOdOfKfkeFBiQIOEdS"
MUG_PLACE_TRAIN_URL="https://docs.google.com/uc?export=download&id=1xtxZygEzTpfmjlJKAA6O4l2uAA9s6dnQ"

if [ -d "$output_dir"/mug_place ]; then
  echo "mug_place data already downloaded."
else
    mkdir "$output_dir"/mug_place
    gdown "$MUG_PLACE_TEST_URL" -O "$output_dir"/mug_place/test_data.zip
    gdown "$MUG_PLACE_TRAIN_URL" -O "$output_dir"/mug_place/train_data.zip
    unzip "$output_dir"/mug_place/train_data.zip -d "$output_dir"/mug_place && rm -rf "$output_dir"/mug_place/train_data.zip
    unzip "$output_dir"/mug_place/test_data.zip -d "$output_dir"/mug_place && rm -rf "$output_dir"/mug_place/test_data.zip
fi

BOWL_GRASP_TEST_URL="https://docs.google.com/uc?export=download&id=15E5boAx3eefkuNQVkOv-X-yzoQgzH-ov"
BOWL_GRASP_TRAIN_URL="https://docs.google.com/uc?export=download&id=1capE6VBNZu5yKjL525mMYuhVUqnLmT9S"

if [ -d "$output_dir"/bowl_grasp ]; then
  echo "bowl_grasp data already downloaded."
else
    mkdir "$output_dir"/bowl_grasp
    gdown "$BOWL_GRASP_TEST_URL" -O "$output_dir"/bowl_grasp/test_data.zip
    gdown "$BOWL_GRASP_TRAIN_URL" -O "$output_dir"/bowl_grasp/train_data.zip
    unzip "$output_dir"/bowl_grasp/train_data.zip -d "$output_dir"/bowl_grasp && rm -rf "$output_dir"/bowl_grasp/train_data.zip
    unzip "$output_dir"/bowl_grasp/test_data.zip -d "$output_dir"/bowl_grasp && rm -rf "$output_dir"/bowl_grasp/test_data.zip
fi

BOWL_PLACE_TEST_URL="https://docs.google.com/uc?export=download&id=1Djx3DZKccF6oBBcKl1Jezs2LNDEzegMG"
BOWL_PLACE_TRAIN_URL="https://docs.google.com/uc?export=download&id=1KxAsV33uOMsXgFCCTxcvPq0DkpU5S_z7"

if [ -d "$output_dir"/bowl_place ]; then
  echo "bowl_place data already downloaded."
else
    mkdir "$output_dir"/bowl_place
    gdown "$BOWL_PLACE_TEST_URL" -O "$output_dir"/bowl_place/test_data.zip
    gdown "$BOWL_PLACE_TRAIN_URL" -O "$output_dir"/bowl_place/train_data.zip
    unzip "$output_dir"/bowl_place/train_data.zip -d "$output_dir"/bowl_place && rm -rf "$output_dir"/bowl_place/train_data.zip
    unzip "$output_dir"/bowl_place/test_data.zip -d "$output_dir"/bowl_place && rm -rf "$output_dir"/bowl_place/test_data.zip
fi

BOTTLE_GRASP_TEST_URL="https://docs.google.com/uc?export=download&id=1bUC07TynJAT1BmFU11k9yo8NfauOBDm1"
BOTTLE_GRASP_TRAIN_URL="https://docs.google.com/uc?export=download&id=1kGQ_vEyl42OGVOe38JoNmTL0E5VevCL5"

if [ -d "$output_dir"/bottle_grasp ]; then
  echo "bottle_grasp data already downloaded."
else
    mkdir "$output_dir"/bottle_grasp
    gdown "$BOTTLE_GRASP_TEST_URL" -O "$output_dir"/bottle_grasp/test_data.zip
    gdown "$BOTTLE_GRASP_TRAIN_URL" -O "$output_dir"/bottle_grasp/train_data.zip
    unzip "$output_dir"/bottle_grasp/train_data.zip -d "$output_dir"/bottle_grasp && rm -rf "$output_dir"/bottle_grasp/train_data.zip
    unzip "$output_dir"/bottle_grasp/test_data.zip -d "$output_dir"/bottle_grasp && rm -rf "$output_dir"/bottle_grasp/test_data.zip
fi

BOTTLE_PLACE_TEST_URL="https://docs.google.com/uc?export=download&id=1V8VmZ4gzCX2ub22wYfOJTlqSEESBW1LA"
BOTTLE_PLACE_TRAIN_URL="https://docs.google.com/uc?export=download&id=1WGs6znQRRT8rAzdkYC--pGFxgv257cwG"
