#!/bin/bash

# Check if correct number of arguments provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_directory_with_mp4> <new_output_directory>"
    exit 1
fi

input_dir=$1
new_output_dir=$2

# Use rsync to copy the input directory to the new output directory
rsync -av --progress "$input_dir/" "$new_output_dir"

# Function to recursively process MP4 files and convert them to PNG sequences
process_mp4_files() {
    for file in "$1"/*; do
        if [ -d "$file" ]; then
            # Recursively process subdirectories
            process_mp4_files "$file"
        elif [ -f "$file" ] && [[ "$file" == *.mp4 ]]; then
            # Get the base name without extension
            base_name=$(basename "$file" .mp4)
            # Create a directory for PNG files
            mkdir "$1/$base_name"
            # Extract frames to PNG
            ffmpeg -i "$file" -vsync 0 "$1/$base_name/%d.png"
            # Remove the MP4 file
            rm "$file"
        fi
    done
}

# Process each MP4 file in the new output directory
process_mp4_files "$new_output_dir"
