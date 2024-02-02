#!/bin/bash

# Check if correct number of arguments provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
fi

input_dir=$1
output_dir=$2

# Use rsync to copy the input directory to the output directory
rsync -av --progress "$input_dir/" "$output_dir"

# Function to process folders
process_folder() {
    for folder in "$1"/*; do
        if [ -d "$folder" ]; then
            # Check if the folder contains PNG files
            png_files=$(find "$folder" -maxdepth 1 -name '*.png')
            if [ -n "$png_files" ]; then
                # Folder name for the output file
                folder_name=$(basename "$folder")
                # Create MP4 file at the same directory level
                ffmpeg -framerate 24 -i "$folder/%d.png" -c:v libx264 -qp 0 -preset veryslow "$1/$folder_name.mp4"
                # Remove the folder after MP4 creation
                rm -rf "$folder"
            else
                # Recursively process subfolders
                process_folder "$folder"
            fi
        fi
    done
}

# Process each subfolder in the output directory
process_folder "$output_dir"
