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

def my_represent_float(self, data):
    if 0 < abs(data) < 1e-5:
        return self.represent_scalar(u'tag:yaml.org,2002:float', '{:.15f}'.format(data).rstrip('0').rstrip('.'))
    else:
        # Default representation for other cases (including 0.0, 1.0, etc.)
        return self.represent_scalar(u'tag:yaml.org,2002:float', repr(data))

yaml = ruamel.yaml.YAML(typ='full', pure=True)
yaml.width = 2
yaml.representer.add_representer(float, my_represent_float)

parser = argparse.ArgumentParser(description='Visualizes camera lidar calibrations')
parser.add_argument('-i', '--indir', type=str, default="./caccdataset")
parser.add_argument('-o', '--outdir', type=str, default='calibration_outputs', help='Output folder')
parser.add_argument('-c', '--camid', type=str, default="cam2", help='Frame to use for the images')
parser.add_argument('-s', '--sequences', type=str, default="44,45", help='Sequence to use for the images')

# Define the chess board rows and columns
CHECKERBOARD = (7, 10)
IMG_SIZE = (960, 600)
SQUARE_SIZE = 100.0

def findFisheyeChessboardCorners(inputs):
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

    fisheye_img_path, marked_img_dir = inputs
    img = cv2.imread(fisheye_img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    # Make sure the chess board pattern was found in the image
    if ret:
        # objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (3,3), (-1,-1), subpix_criteria)
        # imgpoints.append(corners)
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        
        marked_img_path = os.path.join(marked_img_dir, os.path.basename(fisheye_img_path))
        cv2.imwrite(marked_img_path, img)
        print(f"Saved marked fisheye image to {marked_img_path}")

        return objp, corners
    # If the chess board pattern was not found in the image
    return None

def calibrateFisheye(fisheye_img_dirs, marked_img_dirs, camid):
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
    assert type(marked_img_dirs) == list, "marked_img_dirs should be a list of directories" 

    for marked_img_dir in marked_img_dirs:
        if not os.path.exists(marked_img_dir):
            os.makedirs(marked_img_dir)

    fisheye_images = []
    fisheye_marked_img_dirs = []
    for fisheye_img_dir_idx, fisheye_img_dir in enumerate(fisheye_img_dirs):        
        fisheye_image_single = glob.glob(fisheye_img_dir)
        fisheye_images.extend(
            sorted(glob.glob(fisheye_img_dir), key=lambda x: int(os.path.basename(x).split('.')[0]))
        )

        marked_img_dir = marked_img_dirs[fisheye_img_dir_idx]
        fisheye_marked_img_dirs.extend(
            [marked_img_dir] * len(fisheye_image_single)
        )

    p = Pool(processes=20)
    data = p.map(findFisheyeChessboardCorners, 
        [input for input in zip(fisheye_images, fisheye_marked_img_dirs)]
    )
    p.close()
    objp_corners = [dp for dp in data if dp is not None]

    N_imm = len(objp_corners)
    objpoints = [dp[0] for dp in objp_corners]
    imgpoints = [dp[1] for dp in objp_corners]

    # Calibrate fisheye camera
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_imm)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_imm)]

    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        IMG_SIZE,
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    )

    print(f"RMS: {rms}")
    print(f"K: {K}")
    print(f"D: {D}")

    # Sample one image to undistort
    # Note we should really be computing the new camera matrix after undistorting
    img =cv2.imread(fisheye_images[0])
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, IMG_SIZE, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
   
    undistort_img_path = os.path.join(marked_img_dir, os.path.basename(fisheye_images[0]))
    cv2.imwrite(undistort_img_path, np.hstack((img, undistorted_img)))
    print("Saved undistorted fisheye image to ", undistort_img_path)

    # Save calibration data
    calib_dict = {
        'camera_name': f'fisheye/{camid}',
        'distortion_model': 'fisheye',
        'image_height': IMG_SIZE[1],
        'image_width': IMG_SIZE[0],
        'camera_matrix': {
            'cols': 3,
            'rows': 3,
            'data': K.flatten().tolist()
        },
        'distortion_coefficients': {
            'cols': 1,
            'rows': 4,
            'data': D.flatten().tolist()
        }
    }

    return calib_dict

# Main function
def main(args):
    indir   = args.indir
    outdir  = args.outdir
    sequences   = args.sequences
    camid       = args.camid
    sequences = [int(seq) for seq in sequences.split(",")]

    print("| Calibrating fisheye camera |")
    # Define your paths
    fisheye_img_dirs = []
    fisheye_img_outdirs = []
    for seq in sequences:
        fisheye_img_dir = join(indir, f'{seq}/{camid}/*')
        fisheye_img_dirs.append(fisheye_img_dir)

        fisheye_img_outdir = join(outdir, f'corners/{seq}/{camid}')
        fisheye_img_outdirs.append(fisheye_img_outdir)
    
    opencv_calib_dict = calibrateFisheye(fisheye_img_dirs, fisheye_img_outdirs, camid)

    # Save calibration data
    for seq in sequences:
        calib_path = join(outdir, f'calibrations', f'{seq}', f'calib_{camid}_intrinsics.yaml')
        os.makedirs(os.path.dirname(calib_path), exist_ok=True)
        with open(calib_path, 'w') as f:
                yaml.dump(opencv_calib_dict, f)
        print(f'Saved calibration data for {camid} to {calib_path}')

    print("| Finished calibrating fisheye camera |")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)