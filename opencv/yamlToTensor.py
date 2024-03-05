# import pdb; pdb.set_trace()
import os
from os.path import join
import sys
import argparse
import numpy as np
import time
import json

from scipy.spatial.transform import Rotation as R

import yaml
with open('/home/arnavbagad/calibration-tools/opencv/calib_cam0_intrinsics.yaml', 'r') as f:
    data = yaml.load(f, Loader=yaml.SafeLoader)
     
intrinsic_yaml_mat = {}

intrinsic_yaml_mat['camera0'] = np.array(data['camera_matrix']['data']).reshape(3, 3)
print (intrinsic_yaml_mat['camera0'])

def cam0_yaml_to_bev(cam0_int_mat):
    print ("temp")


# def yaw_to_homo(pose_np, yaw):
#     trans = pose_np[:, 1:4]
#     rot_mat = R.from_euler('z', yaw, degrees=True).as_matrix()
#     tmp = np.expand_dims(np.eye(4, dtype=np.float64), axis=0)
#     homo_mat = np.repeat(tmp, len(trans), axis=0)
#     homo_mat[:, :3, :3] = rot_mat
#     # homo_mat[:, :3, 3 ] = trans
#     return homo_mat
