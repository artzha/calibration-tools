"""
Copyright (c) 2023 Arthur King Zhang arthurz@cs.utexas.edu

This script copies images from a folder to another folder. It also converts point clouds from .bin to .pcd format.
"""

import sys
import os
import pdb
import shutil
import numpy as np
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Copy images from a folder to another folder')
parser.add_argument('-i', '--indir', type=str, default="/robodata/ecocar_logs/processed/CACCDataset",  help='Input folder that contains camera directories')
parser.add_argument('-s', '--sequence', type=str, default=44, help='Sequence to use for the images')
parser.add_argument('-o', '--outdir', type=str, default='./caccdataset', help='Output folder')
parser.add_argument('-r', '--skiprate', type=int, default=10, help='Number of images to skip per frame. Set to -1 to disable')
parser.add_argument('-sf', '--start_frame', type=int, default=0, help='Frame to begin sampling at')
parser.add_argument('-ef', '--end_frame', type=int, default=0, help='Frame to end sampling at')

SENSOR_LIST = ['cam0', 'cam1', 'cam2', 'cam3', 'cam4', 'os1']

def get_frame_from_name(filename):
    # Extract last frame number from filenames with format 2d_raw_cam0_sequence_frame.png
    return int(filename.split('_')[-1].split('.')[0])

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


def main(args):
    indir = args.indir
    seq = args.sequence
    outdir = args.outdir
    rate = args.skiprate
    start_frame = args.start_frame
    end_frame = args.end_frame
    assert os.path.exists(indir), f'Input folder {indir} does not exist'

    if not os.path.exists(outdir):
        print(f'Creating output folder {outdir}')
        os.makedirs(outdir)

    camdir = os.path.join(indir, '2d_raw')
    cam_subdirs = [subdir for subdir in os.listdir(camdir) if os.path.isdir(os.path.join(camdir, subdir))]
    cam_fulldirs = [os.path.join(camdir, subdir, f'{seq}') for subdir in cam_subdirs]

    #1 Extract frame numbers 
    files = [f for f in os.listdir(cam_fulldirs[0]) if f.endswith('.png')]
    frames = sorted([get_frame_from_name(f) for f in files])

    if args.end_frame > 0:
        frames = frames[args.start_frame:args.end_frame]
    else:
        frames = frames[args.start_frame:]
    
    #2 Copy frames for each camera
    for cam_fulldir in cam_fulldirs:
        subdir = cam_fulldir.split('/')[-2]
        if subdir not in SENSOR_LIST:
            continue
        dstdir = os.path.join(outdir, f'{seq}', subdir)
        os.makedirs(dstdir, exist_ok=True)
        for frame in frames[::rate]:
            src = os.path.join(cam_fulldir, f'2d_raw_{subdir}_{seq}_{frame}.png')
            dst = os.path.join(dstdir, f'{frame}.png')
            shutil.copy(src, dst)
        print(f'Copied camera {subdir} images to {outdir}')
    print("Copied all images to output folder")

    #3 Open each lidar folder, convert to pcd file, and save to output folder
    pcdir = os.path.join(indir, '3d_raw')
    pc_subdirs = [subdir for subdir in os.listdir(pcdir) if os.path.isdir(os.path.join(pcdir, subdir))]
    pc_fulldirs = [os.path.join(pcdir, subdir, f'{seq}') for subdir in pc_subdirs]

    for pc_fulldir in pc_fulldirs:
        subdir = pc_fulldir.split('/')[-2]
        dstdir = os.path.join(outdir, f'{seq}', subdir)
        os.makedirs(dstdir, exist_ok=True)
        for frame in frames[::rate]:
            src = os.path.join(pc_fulldir, f'3d_raw_{subdir}_{seq}_{frame}.bin')
            pc_np = np.fromfile(src, dtype=np.float32).reshape(-1, 5)
            xyz = pc_np[:, :3]
            dst = os.path.join(dstdir, f'{frame}.pcd')
            numpy_to_pcd(xyz, dst)

        print(f'Converted lidar {subdir} point clouds to {outdir}')
    
    print("Converted all point clouds to output folder")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)




