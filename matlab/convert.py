"""
Copyright (c) 2023 Arthur King Zhang arthurz@cs.utexas.edu

This script reads .mat calibration files and converts them to .yaml files in ROS compatible format.

The .mat files are expected to follow the naming conventions below:
- calib_cam0_to_cam1.mat [Extrinsics from camera 0 to camera 1, Intrinsics for camera 0, Intrinsics for camera 1]
- calib_os1_cam0.mat [Extrinsics from OS1 to camera 0, Extrinsic Errors]
"""

import sys
import os
from os.path import join
import pdb
import ruamel.yaml
import argparse
import cv2

import scipy.io as sio
import numpy as np

EXPECTED_CALIBS = ['K_cam0.mat', 'K_cam1.mat', 'T_cam0_cam1.mat', 'T_os1_cam0.mat',
                   'K_cam2.mat', 'K_cam3.mat', 'K_cam4.mat', 
                   'T_os1_cam2.mat', 'T_os1_cam3.mat', 'T_os1_cam4.mat']
#EXPECTED_CALIBS = ['T_cam0_cam1.mat', 'T_os1_cam0.mat',
#                   'T_os1_cam2.mat', 'T_os1_cam3.mat', 'T_os1_cam4.mat']

def my_represent_float(self, data):
    if 0 < abs(data) < 1e-5:
        return self.represent_scalar(u'tag:yaml.org,2002:float', '{:.15f}'.format(data).rstrip('0').rstrip('.'))
    else:
        # Default representation for other cases (including 0.0, 1.0, etc.)
        return self.represent_scalar(u'tag:yaml.org,2002:float', repr(data))

yaml = ruamel.yaml.YAML(typ='full', pure=True)
yaml.width = 2
yaml.representer.add_representer(float, my_represent_float)

parser = argparse.ArgumentParser(description='Copy images from a folder to another folder')
parser.add_argument('-i', '--indir', type=str, default="./matlab_data",  help='Input folder that contains matlab calibration .mat files')
parser.add_argument('-ih', '--image_height', type=int, default=600, help='Image height for the camera')
parser.add_argument('-iw', '--image_width', type=int, default=960, help='Image width for the camera')
parser.add_argument('-s', '--sequence', type=int, default=44, help='Sequence to use for the images')
parser.add_argument('-o', '--outdir', type=str, default="calibration_outputs/calibrations", help='Output folder to save yaml calibration files')
parser.add_argument('-l', '--use_lidar', action='store_false', help='Whether to use lidar data or not')

def read_mat_as_dict(filename, image_size):
    """
    Read a .mat file and return its contents in standardized calibration format.

    Parameters:
    filename (str): The filename of the .mat file.

    Returns:
    dict: A dictionary containing the contents of the .mat file.
    """
    mat = sio.loadmat(filename)

    #1 Decode name to understand what type of calibration it is
    inputs = os.path.splitext(os.path.basename(filename))[0].split('_')
    caltype, sensorname = inputs[:2]

    calib = {}
    if caltype=="K": # Camera Intrinsic
        calib['camera_matrix'] = {
            "rows": 3,
            "cols": 3,
            "data": mat['K'].flatten().tolist()
        }
        calib['camera_name'] = f'{sensorname}'

        dist = np.array([0,0,0,0,0], dtype=np.float64)
        dist[0], dist[1] = mat['raddist'][0][0], mat['raddist'][0][1]
        dist[2], dist[3] = mat['tandist'][0][0], mat['tandist'][0][1]
        if len(mat['raddist'][0]) > 2:
            dist[4] = mat['raddist'][0][2]
        calib['distortion_coefficients'] = {
            "rows": 1,
            "cols": 5,
            "data": dist.flatten().tolist()
        }
        calib['distortion_model'] = "plumb_bob"
        calib['image_width'] = image_size[0]
        calib['image_height'] = image_size[1]
    elif caltype=="T":
        A = mat['A'].reshape(4, 4)

        if "cam0" in sensorname:
            A[:3, 3] = A[:3, 3] * 0.001 # Convert to meters

        calib['extrinsic_matrix'] = {
            "rows": 4,
            "cols": 4,
            "data": mat['A'].flatten().tolist()
        }
        targetname = inputs[2]
        sensorname = f'{sensorname}_to_{targetname}'

    return calib, sensorname

def get_pts2pixel_transform(calib_dict, camid):
    """
    Returns a transformation matrix that converts 3D points in LiDAR frame to image pixel coordinates
    Boilerplate function to get the projection matrix from the calibration dictionary

    P =  Pcam @ Eye(Re | 0) @ T_lidar_cam

    Inputs:
        calib_dict: [dict] calibration dictionary
    Outputs:
        pts2pix: [4 x 4] transformation matrix
    """
    T_lidar_cam = np.array(calib_dict[f'os1_to_{camid}']['extrinsic_matrix']['data']).reshape(4, 4)

    T_canon = np.eye(4)
    T_canon[:3, :3] = np.array(calib_dict[f'{camid}']['rectification_matrix']['data']).reshape(3, 3)

    M = np.array(calib_dict[f'{camid}']['projection_matrix']['data']).reshape(3, 4)[:3, :3]
    P_pix_cam = np.eye(4)
    P_pix_cam[:3, :3] = M

    T_lidar_to_rect = P_pix_cam @ T_canon @ T_lidar_cam

    return T_lidar_to_rect[:3, :]

def compute_lidar_to_cameras(calib_dict, use_lidar):
    """
    Computes the transformation from the lidar to each camera using the extrinsics from lidar to cam0 and then
    cam0 to all other cameras

    TODO: Extend this in the future to arbitrary number of cameras

    Inputs:
        calib_dict (dict): A dictionary containing the calibration data for all sensors
    Outputs:
        calib_dict (dict): The input dictionary with the new transformations added
    """
    T_cam0_cam1 = calib_dict['cam0_to_cam1']['extrinsic_matrix']['data']
    T_cam0_cam1 = np.array(T_cam0_cam1).reshape(4, 4)

    if use_lidar:
        T_os1_cam0 = calib_dict['os1_to_cam0']['extrinsic_matrix']['data']
        T_os1_cam0 = np.array(T_os1_cam0).reshape(4, 4)
        T_os1_to_cam1 = T_cam0_cam1 @ T_os1_cam0

        calib_dict['os1_to_cam1'] = {
            "extrinsic_matrix": {
                "rows": 4,
                "cols": 4,
                "data": T_os1_to_cam1.flatten().tolist()
            }
        }

        # Add projection matrices for the lidar to rectified pixel coordinates for cam0 and cam1
        for camid in ['cam0', 'cam1']:
            T_os1_camrect = get_pts2pixel_transform(calib_dict, camid)
            calib_dict[f'os1_to_{camid}'].update({
                "projection_matrix": {
                    "rows": 3,
                    "cols": 4,
                    "data": T_os1_camrect.flatten().tolist()
                }
            })
    return calib_dict

def compute_intercamera_transformations(calib_dict, transform_filename):
    cam_list = list(calib_dict.keys())
    if "cam0_to_cam1" not in cam_list:
        return

    essentialfilename = join(os.path.dirname(transform_filename), f'E_cam0_cam1.mat')
    intercameraposefilename = join(os.path.dirname(transform_filename), f'T_cam0_cam1.mat')
    assert os.path.exists(essentialfilename), f'Inter-camera calibration {essentialfilename} does not exist'
    assert os.path.exists(intercameraposefilename), f'Inter-camera pose calibration {intercameraposefilename} does not exist'
    
    K1 = np.array(calib_dict['cam0']['camera_matrix']['data'], dtype=np.float64).reshape(3, 3)
    K2 = np.array(calib_dict['cam1']['camera_matrix']['data'], dtype=np.float64).reshape(3, 3)
    dist1 = np.array(calib_dict['cam0']['distortion_coefficients']['data'], dtype=np.float64)
    dist2 = np.array(calib_dict['cam1']['distortion_coefficients']['data'], dtype=np.float64)

    image_size = (calib_dict['cam0']['image_width'], calib_dict['cam0']['image_height'])
    RT = np.array(calib_dict['cam0_to_cam1']['extrinsic_matrix']['data'], dtype=np.float64).reshape(4, 4)
    R = RT[:3, :3].reshape(3, 3)
    T = RT[:3, 3].reshape(3, 1)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, dist1, K2, dist2, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    calib_dict['cam0']['rectification_matrix'] = {
        "rows": 3,
        "cols": 3,
        "data": R1.flatten().tolist()
    }
    calib_dict['cam1']['rectification_matrix'] = {
        "rows": 3,
        "cols": 3,
        "data": R2.flatten().tolist()
    }
    calib_dict['cam0']['projection_matrix'] = {
        "rows": 3,
        "cols": 4,
        "data": P1.flatten().tolist()
    }
    calib_dict['cam1']['projection_matrix'] = {
        "rows": 3,
        "cols": 4,
        "data": P2.flatten().tolist()
    }
    calib_dict['cam0']['disparity_matrix'] = {
        "rows": 4,
        "cols": 4,
        "data": Q.flatten().tolist()
    }
    calib_dict['cam1']['disparity_matrix'] = {
        "rows": 4,
        "cols": 4,
        "data": Q.flatten().tolist()
    }


def main(args):
    indir = args.indir
    seq = args.sequence
    outdir = args.outdir
    use_lidar = args.use_lidar
    assert os.path.exists(indir), f'Input folder {indir} does not exist'

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    calib_dir = join(indir, f'{seq}')

    #1 Load all matlab calibrations to ROS compatible dictionaries
    calib_dict = {}
    intercamera_filename = None
    for filename in os.listdir(calib_dir):
        if filename in EXPECTED_CALIBS:
            print(f'Processing {filename}')
            calib, sensorname = read_mat_as_dict(join(calib_dir, filename), image_size=(args.image_width, args.image_height))
            if sensorname == "cam0_to_cam1":
                intercamera_filename = join(calib_dir, filename)
            calib_dict[sensorname] = calib

    #1a Compute inter camera transformations
    print("|- Computing inter-camera transformations -|")
    if intercamera_filename is not None:
        compute_intercamera_transformations(calib_dict, intercamera_filename)
     
    #2 Compute remaining transformations from the extrinsics
    print("|- Computing remaining lidar to camera transformations -|")
    compute_lidar_to_cameras(calib_dict, use_lidar)
    
    #3 Save all calibrations to yaml files
    for sensorname, calib in calib_dict.items():
        if "to" in sensorname:
            filename = f'calib_{sensorname}.yaml'
        else:
            filename = f'calib_{sensorname}_intrinsics.yaml'
        calib_path = join(outdir, str(seq), filename)
        os.makedirs(os.path.dirname(calib_path), exist_ok=True)
        with open(calib_path, 'w') as f:
            yaml.dump(calib, f)
        print(f'Saved calibration to {calib_path}')

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
