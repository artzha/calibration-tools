import os
import shutil
from pathlib import Path
import argparse 
import numpy as np

# Example usage: python sample_calibrations.py --seq 0 --input_dir /media/arthur/ExtremePro/lsmap_bags_processed --output_dir /media/arthur/ExtremePro/lsmap_bags_processed_subsampled --subsample_factor 10
def parse_args():
    parser = argparse.ArgumentParser(description="Subsample images and lidar data.")
    parser.add_argument("--seq", type=str, help="Path to the directory containing images from camera 1.", required=True)
    parser.add_argument("--input_dir", type=str, help="Path to the input directory.", required=True)
    parser.add_argument("--output_dir", type=str, help="Path to the output directory.", required=True)
    parser.add_argument("--subsample_factor", type=int, default=10, help="Factor by which to subsample frames.", required=False)
    return parser.parse_args()

def numpy_to_pcd(array, filename):
    """
    Convert a Nx3 numpy array to a PCD file.

    Parameters:
    array (numpy.ndarray): Nx3 numpy array containing point cloud data.
    filename (str): The output filename for the PCD file.
    """
    import numpy as np
    assert array.shape[1] == 3, "Input array must be Nx3."

    header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH {len(array)}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {len(array)}
DATA ascii
"""
    with open(filename, 'w') as f:
        f.write(header)
        np.savetxt(f, array, fmt='%f %f %f')

def validate_and_subsample_images(dir_cam1, dir_cam2, dir_lidar, output_dir, subsample_factor=1):
    """
    Args:
        dir_cam1 (str): Path to the directory containing images from camera 1.
        dir_cam2 (str): Path to the directory containing images from camera 2.
        dir_lidar (str): Path to the directory containing lidar .bin files.
        output_dir (str): Path to the output directory.
        subsample_factor (int): Factor by which to subsample frames.
    """
    # Extract frame number from file name
    def extract_frame_number(file):
        return int(file.stem.split("_")[-1])

    def extract_seq_number(file):
        return int(file.stem.split("_")[-2])
    
    # List all files in each directory and sort them
    unsorted_cam1_files = sorted(Path(dir_cam1).glob("2d_raw_cam0_*_*.jpg"))
    unsorted_cam2_files = sorted(Path(dir_cam2).glob("2d_raw_cam1_*_*.jpg"))
    unsorted_lidar_files = sorted(Path(dir_lidar).glob("3d_raw_os1_*_*.bin"))
    unsorted_seq_numbers = [extract_seq_number(file) for file in unsorted_cam1_files]
    unsorted_frame_numbers = [ extract_frame_number(file) for file in unsorted_cam1_files ]

    # Get sort index for frame numbers
    sort_index = sorted(range(len(unsorted_frame_numbers)), key=lambda k: unsorted_frame_numbers[k])
    frame_numbers = [ unsorted_frame_numbers[i] for i in sort_index ]
    seq_numbers = [ unsorted_seq_numbers[i] for i in sort_index ]
    cam1_files = [ unsorted_cam1_files[i] for i in sort_index ]
    cam2_files = [ unsorted_cam2_files[i] for i in sort_index ]
    lidar_files = [ unsorted_lidar_files[i] for i in sort_index ]

    assert len(cam1_files) == len(cam2_files) == len(lidar_files), "Number of files in each directory must be the same."
    # Subsample frames based on the specified factor
    cam1_files = cam1_files[::subsample_factor]
    cam2_files = cam2_files[::subsample_factor]
    lidar_files = lidar_files[::subsample_factor]
    
    # Define output paths
    cam1_output_dir = Path(output_dir) / f"2d_raw/cam0/{seq_numbers[0]}"
    cam2_output_dir = Path(output_dir) / f"2d_raw/cam1/{seq_numbers[0]}"
    lidar_output_dir = Path(output_dir) / f"3d_raw/os1/{seq_numbers[0]}"
    cam1_output_dir.mkdir(parents=True, exist_ok=True)
    cam2_output_dir.mkdir(parents=True, exist_ok=True)
    lidar_output_dir.mkdir(parents=True, exist_ok=True)

    # Copy the subsampled files to the new directory structure
    for cam1_file, cam2_file, lidar_file, frame in zip(cam1_files, cam2_files, lidar_files, frame_numbers):
        # Copy cam1 image
        cam1_dest = cam1_output_dir / f"{frame}.jpg"
        shutil.copy(cam1_file, cam1_dest)

        # Copy cam2 image
        cam2_dest = cam2_output_dir / f"{frame}.jpg"
        shutil.copy(cam2_file, cam2_dest)

        # Convert lidar file from .bin to .pcd
        lidar_np = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 5)
        lidar_np = lidar_np[:, :3]
        assert lidar_np.shape[0] == 131072, "Lidar file must contain 131072 points."
        lidar_dest = lidar_output_dir / f"{frame}.pcd"
        numpy_to_pcd(lidar_np, lidar_dest)

        lidar_dest = lidar_output_dir / f"{frame}.bin"
        shutil.copy(lidar_file, lidar_dest)

    print(f"Finished processing sequence {seq_numbers[0]} with {len(cam1_files)} frames.")
    print(f"Subsampled frames have been saved to {output_dir}")

if __name__ == "__main__":
    args = parse_args()
    cam0_dir = os.path.join(args.input_dir,"2d_raw/cam0", args.seq)
    cam1_dir = os.path.join(args.input_dir, "2d_raw/cam1", args.seq)
    lidar_dir = os.path.join(args.input_dir, "3d_raw/os1", args.seq)

    validate_and_subsample_images(
        dir_cam1=cam0_dir,
        dir_cam2=cam1_dir,
        dir_lidar=lidar_dir,
        output_dir=args.output_dir,
        subsample_factor=args.subsample_factor
    )