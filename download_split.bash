#!/bin/bash

modality="3d_raw"
extension="bin"

# Define base paths
remote_base="robodata:/robodata/ecocar_logs/processed/CACCDataset/${modality}"
local_base="/media/warthog/Art_SSD/ecocar_processed/CACCDataset/${modality}"

# Define directories to iterate over
if [ "$modality" == "2d_raw" ]; then
    directories=("cam0" "cam1")
elif [ "$modality" == "2d_undistorted" ]; then
    directories=("cam2" "cam3" "cam4")
else
    directories=("os1")
fi

# Number of frames to copy
num_frames=10
seq=0

# Iterate over each directory
for dir in "${directories[@]}"; do
    echo "Processing $dir..."
    # Iterate over each frame
    for ((frame=0; frame<num_frames; frame++)); do
        # Construct the filename pattern for rsync
        filename="${modality}_${dir}_${seq}_${frame}.${extension}"
        
        # Construct source and destination paths
        src_path="${remote_base}/${dir}/${seq}/${filename}"
        dest_dir="${local_base}/${dir}/${seq}"
        dest_path="${dest_dir}/${filename}"

        # Ensure the destination directory exists
        mkdir -p "${dest_dir}"
        
        # Execute rsync command to copy the file
        rsync -azP "${src_path}" "${dest_path}"
    done
done

echo "All specified frames have been copied."