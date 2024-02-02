#!/bin/bash

# Check if correct number of arguments provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <operation: mp4/png> <input_directory> <output_directory>"
    exit 1
fi

operation=$1
input_dir=$2
output_dir=$3
frame_rate=24 # Adjust this if a different frame rate is required

# Ensure ffmpeg and rsync are installed
if ! command -v ffmpeg &> /dev/null || ! command -v rsync &> /dev/null; then
    echo "Error: ffmpeg and rsync must be installed to run this script."
    exit 1
fi

# Function to recursively convert PNG sequences to MP4
convert_to_mp4() {
    cp -r "$input_dir" "$output_dir"
    process_folder() {
        for folder in "$1"/*; do
            if [ -d "$folder" ]; then
                png_files=$(find "$folder" -maxdepth 1 -name '*.png')
                if [ -n "$png_files" ]; then
                    folder_name=$(basename "$folder")
                    ffmpeg -framerate $frame_rate -i "$folder/%d.png" -c:v libx264 -qp 0 -preset veryslow "$1/$folder_name.mp4"
                    rm -rf "$folder"
                else
                    process_folder "$folder"
                fi
            fi
        done
    }
    process_folder "$output_dir"
}

# Function to recursively convert MP4 to PNG sequences
convert_to_png() {
    cp -r "$input_dir" "$output_dir"
    process_mp4_files() {
        for file in "$1"/*; do
            if [ -d "$file" ]; then
                process_mp4_files "$file"
            elif [ -f "$file" ] && [[ "$file" == *.mp4 ]]; then
                base_name=$(basename "$file" .mp4)
                mkdir "$1/$base_name"
                ffmpeg -i "$file" -r $frame_rate -vsync 0 -start_number 0 "$1/$base_name/%d.png"
                rm "$file"
            fi
        done
    }
    process_mp4_files "$output_dir"
}

# Check operation type and call the respective function
case $operation in
    mp4)
        convert_to_mp4
        ;;
    png)
        convert_to_png
        ;;
    *)
        echo "Invalid operation: $operation. Use 'mp4' or 'png'."
        exit 1
        ;;
esac
