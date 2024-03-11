"""
This script calibrates a fisheye camera from a set of chessboard images.

Misc Notes for CACCDataset: cam2 [python opencv/calibrateFisheye.py -sf 190 -ef 850]
"""

import os
from os.path import join
import numpy as np
import cv2
import glob
import argparse
import ruamel.yaml

from multiprocessing import Pool

# python opencv/undistortFisheye.py -i /robodata/ecocar_logs/processed/CACCDataset/2d_raw -o /robodata/ecocar_logs/processed/CACCDataset/2d_undistort -c cam4

def my_represent_float(self, data):
    if 0 < abs(data) < 1e-5:
        return self.represent_scalar(u'tag:yaml.org,2002:float', '{:.15f}'.format(data).rstrip('0').rstrip('.'))
    else:
        # Default representation for other cases (including 0.0, 1.0, etc.)
        return self.represent_scalar(u'tag:yaml.org,2002:float', repr(data))

yaml = ruamel.yaml.YAML(typ='safe', pure=True)
yaml.width = 2
yaml.representer.add_representer(float, my_represent_float)

parser = argparse.ArgumentParser(description='Visualizes camera lidar calibrations')
parser.add_argument('-i', '--indir', type=str, default="./caccdataset")
parser.add_argument('-ic', '--indir_calib', type=str, default="./calibration_outputs/calibrations")
parser.add_argument('-o', '--outdir', type=str, default='./caccdataset_undistort', help='Output folder')
parser.add_argument('-c', '--camid', type=str, default="cam2", help='Frame to use for the images')
parser.add_argument('-s', '--sequence', type=int, default=44, help='Sequence to use for the images')

# Define the chess board rows and columns
CHECKERBOARD = (7, 10)
IMG_SIZE = (960, 600)
SQUARE_SIZE = 100.0

def undistortFisheyeSingle(inputs):
    fisheye_img_path, calib_dict, undistort_img_path = inputs
    K = np.array(calib_dict['camera_matrix']['data']).reshape(
        calib_dict['camera_matrix']['rows'], calib_dict['camera_matrix']['cols']
    )
    D = np.array(calib_dict['distortion_coefficients']['data']).reshape(
        calib_dict['distortion_coefficients']['rows'], calib_dict['distortion_coefficients']['cols']
    )
    img =cv2.imread(fisheye_img_path)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, IMG_SIZE, cv2.CV_16SC2)
    # map1, map2 = cv2.initUndistortRectifyMap(K, D, np.eye(3), K, IMG_SIZE, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.
    BORDER_CONSTANT)
    if undistort_img_path is not None:
        cv2.imwrite(undistort_img_path, undistorted_img)
        print("Saved undistorted fisheye image to ", undistort_img_path)
    else:
        return undistorted_img

def undistortFisheye(fisheye_img_dir, undistort_img_dir, calib_dict):
    if not os.path.exists(undistort_img_dir):
        os.makedirs(undistort_img_dir)

    fisheye_images = glob.glob(fisheye_img_dir)
    fisheye_images = sorted(fisheye_images, key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))
    fisheye_undistort_img_dirs = [join(undistort_img_dir, os.path.basename(frame).replace("2d_raw", "2d_undistort")) for frame in fisheye_images]
    calib_dict_list = [calib_dict for _ in fisheye_images]

    p = Pool(processes=20)
    p.map(undistortFisheyeSingle, 
        [input for input in zip(fisheye_images, calib_dict_list, fisheye_undistort_img_dirs)]
    )
    p.close()


# Main function
def main(args):
    indir   = args.indir
    indir_calib = args.indir_calib
    outdir  = args.outdir
    seq     = args.sequence
    camid   = args.camid

    print("| Undistorting fisheye camera images |")
    # Define your paths
    fisheye_img_dir = join(indir, f'{seq}/{camid}/*')
    img_outdir = join(outdir, f'{seq}/{camid}')

    if not os.path.exists(fisheye_img_dir):
        fisheye_img_dir = join(indir, f'{camid}/{seq}/*')
        img_outdir = join(outdir, f'{camid}/{seq}')

    # Load calibrations from file
    calib_dir = join(indir_calib, f'{seq}', f'calib_{camid}_intrinsics.yaml')
    with open(calib_dir, 'r') as f:
        calib_dict = yaml.load(f)

    undistortFisheye(fisheye_img_dir, img_outdir, calib_dict)

    print("| Finished undistorting fisheye camera images |")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)