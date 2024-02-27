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

import scipy.io as sio
import numpy as np

EXPECTED_CALIBS = ['K_cam0.mat', 'K_cam1.mat', 'T_cam0_cam1.mat', 'T_os1_cam0.mat']

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
parser.add_argument('-i', '--indir', type=str, help='Input folder that contains matlab calibration .mat files')
parser.add_argument('-s', '--sequence', type=int, default=44, help='Sequence to use for the images')
parser.add_argument('-o', '--outdir', type=str, help='Output folder to save yaml calibration files')

def read_mat_as_dict(filename):
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
        dist[4] = mat['raddist'][0][2]
        calib['distortion_coefficients'] = {
            "rows": 1,
            "cols": 5,
            "data": dist.flatten().tolist()
        }
        calib['distortion_model'] = "plumb_bob"
        calib['image_height'] = 600
        calib['image_width'] = 960
    elif caltype=="T":
        calib['extrinsic_matrix'] = {
            "rows": 4,
            "cols": 4,
            "data": mat['A'].flatten().tolist()
        }
        targetname = inputs[2]
        sensorname = f'{sensorname}_to_{targetname}'

    return calib, sensorname

def compute_lidar_to_cameras(calib_dict):
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

    return calib_dict


def main(args):
    indir = args.indir
    seq = args.sequence
    outdir = args.outdir
    assert os.path.exists(indir), f'Input folder {indir} does not exist'

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    calib_dir = join(indir, f'{seq}')

    #1 Load all matlab calibrations to ROS compatible dictionaries
    calib_dict = {}
    for filename in os.listdir(calib_dir):
        if filename in EXPECTED_CALIBS:
            print(f'Processing {filename}')
            calib, sensorname = read_mat_as_dict(join(calib_dir, filename))
            calib_dict[sensorname] = calib

    #2 Compute remaining transformations from the extrinsics
    print("|- Computing remaining lidar to camera transformations -|")
    compute_lidar_to_cameras(calib_dict)
        
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
