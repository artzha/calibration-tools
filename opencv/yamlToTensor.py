# import pdb; pdb.set_trace()
import os
from os.path import join
import sys
import argparse
import numpy as np
import time
import json
import argparse
import ruamel.yaml
import opencv.tensor as tensor

from PIL import Image # Following https://github.com/mit-han-lab/bevfusion/blob/db75150717a9462cb60241e36ba28d65f6908607/mmdet3d/datasets/pipelines/loading.py#L58

from scipy.spatial.transform import Rotation as R

def my_represent_float(self, data):
    if 0 < abs(data) < 1e-5:
        return self.represent_scalar(u'tag:yaml.org,2002:float', '{:.15f}'.format(data).rstrip('0').rstrip('.'))
    else:
        # Default representation for other cases (including 0.0, 1.0, etc.)
        return self.represent_scalar(u'tag:yaml.org,2002:float', repr(data))

yaml = ruamel.yaml.YAML(typ='safe', pure=True)
yaml.width = 2
yaml.representer.add_representer(float, my_represent_float)

parser = argparse.ArgumentParser(description='Convert CODa format calibrations to nuScenes for BEVFusion')
parser.add_argument('-i', '--indir', type=str, default="/robodata/ecocar_logs/processed/CACCDataset", help='Root directory for dataset')
parser.add_argument('-o', '--outdir', type=str, default="./leva_calibrations", help='Output directory for the calibrations')
parser.add_argument('-s', '--sequence', type=int, default=0, help='Sequence to use for the calibrations')

CAMID_LIST = ["cam0", "cam1", "cam2", "cam3", "cam4"]
CAMSUBDIR_LIST = ["2d_raw/cam0", "2d_raw/cam1", "2d_undistort/cam2", "2d_undistort/cam3", "2d_undistort/cam4"]

def read_calibrations_from_dir(calib_dir, camid_list):
    calib_dict = {}
    #1 Camera calibrations
    for camid in camid_list:
        calibration_path = join(calib_dir, f'calib_{camid}_intrinsics.yaml')
        with open(calibration_path) as f:
            data = yaml.load(f)
            calib_dict[camid] = {}
            calib_dict[camid]['K'] = np.array(data['camera_matrix']['data']).reshape(
                data['camera_matrix']['rows'], data['camera_matrix']['cols']
            )
            calib_dict[camid]['d'] = np.array(data['distortion_coefficients']['data']).reshape(
                data['distortion_coefficients']['rows'], data['distortion_coefficients']['cols']
            )

    #2 LiDAR Camera calibrations
    for camid in camid_list:
        calibration_path = join(calib_dir, f'calib_os1_to_{camid}.yaml')
        with open(calibration_path) as f:
            data = yaml.load(f)
            ext_key = f'os1_{camid}'
            calib_dict[ext_key] = {}
            calib_dict[ext_key]['A'] = np.array(data['extrinsic_matrix']['data']).reshape(
                data['extrinsic_matrix']['rows'], data['extrinsic_matrix']['cols']
            )

    return calib_dict


def ecocar2nusc_calibration(calib_dir):
    #1 Open the calibration files
    calib_dict = read_calibrations_from_dir(calib_dir, CAMID_LIST)

    data = {}
    data["lidar2camera"] = []
    data["lidar2image"] = []
    data["camera2ego"] = []
    data["camera_intrinsics"] = []
    data["camera2lidar"] = []
    data["lidar2ego"] = []
    data["img_aug_matrix"] = []
    data["lidar_aug_matrix"] = []
    for camid in CAMID_LIST:
        #2 Dump camera intrinsics 4x4
        intrinsics = np.eye(4, dtype=np.float32)
        intrinsics[:3, :3] = calib_dict[camid]['K']
        data["camera_intrinsics"].append(intrinsics)

        #4 Dump lidar2ego
        lidar2ego = np.eye(4, dtype=np.float32)
        data['lidar2ego'].append(lidar2ego)

        #5 Dump lidar2camera
        lidar2camera = calib_dict[f'os1_{camid}']['A'].astype(np.float32)
        data['lidar2camera'].append(lidar2camera)

        #3 Dump camera2ego (Assume ego is at the lidar frame)
        camera2ego = np.eye(4, dtype=np.float32)
        camera2ego = np.linalg.inv(lidar2camera)
        data['camera2ego'].append(camera2ego)

        #6 Dump camera2lidar
        camera2lidar = np.linalg.inv(lidar2camera)
        data['camera2lidar'].append(camera2lidar)

        #7 Dump lidar2image
        lidar2image = np.eye(4, dtype=np.float32)
        canon = np.zeros((3, 4))
        canon[:3, :3] = np.eye(3)
        lidar2image[:3, :] = calib_dict[camid]['K'] @ canon @ calib_dict[f'os1_{camid}']['A']
        data['lidar2image'].append(lidar2image)

        #8 Dump imageaugmatrix
        img_aug = np.eye(4, dtype=np.float32)
        data['img_aug_matrix'].append(img_aug)

        #9 Dump lidaraugmatrix
        lidar_aug = np.eye(4, dtype=np.float32)
        data['lidar_aug_matrix'].append(lidar_aug)

    # Copy the last fisheye twice to give same number of camera inputs as expected
    data['camera_intrinsics'].append(data['camera_intrinsics'][-1])
    data['lidar2ego'].append(data['lidar2ego'][-1])
    data['lidar2camera'].append(data['lidar2camera'][-1])
    data['camera2ego'].append(data['camera2ego'][-1])
    data['camera2lidar'].append(data['camera2lidar'][-1])
    data['lidar2image'].append(data['lidar2image'][-1])
    data['img_aug_matrix'].append(data['img_aug_matrix'][-1])
    data['lidar_aug_matrix'].append(data['lidar_aug_matrix'][-1])

    return data

def ecocar2nusc_sensor(data_dict, start, end):
    """
    data_dict: Dictionary containing the paths to the sensor data
        cam0: ""
        cam1: ""
        ...
        camn: ""
        os1: ""
    """
    data = {"img": [], "points": []}
    for frame in range(start, end):
        images = []
        for camid in CAMID_LIST:
            cam_dir = data_dict[camid]
            subdir, sensorid, seq = cam_dir.split('/')[-3:]
            img_path = join(cam_dir, f'{subdir}_{sensorid}_{seq}_{frame}.jpg')
            images.append(Image.open(img_path))
        data["img"].append(images)

        points_dir = data_dict["os1"]
        subdir, sensorid, seq = points_dir.split('/')[-3:]
        points_path = join(points_dir, f'{subdir}_{sensorid}_{seq}_{frame}.bin')
        points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 5)
        data["points"].append(points)

    return data

def main(args):
    indir = args.indir
    outdir = args.outdir
    sequence = args.sequence

    assert os.path.exists(indir), "Input directory does not exist"
    os.makedirs(outdir, exist_ok=True)

    calibration_dir = join(indir, "calibrations", f'{sequence}')
    data = ecocar2nusc_calibration(calibration_dir)

    # Load paths to the images and point clouds
    sensordata_dict = {}
    for camid, camsubdir in zip(CAMID_LIST, CAMSUBDIR_LIST):
        cam_dir = join(indir, camsubdir, f'{sequence}')
        sensordata_dict[camid] = cam_dir

    sensordata_dict["os1"] = join(indir, "3d_raw", "os1", f'{sequence}')

    # Load first 10 frames (Not Needed, load directly from ROS)
    # data.update(ecocar2nusc_sensor(sensordata_dict, 0, 10))

    # Save the calibration data to .tensor files
    for key, value in data.items():
        tensor.save(np.array(value, dtype=np.float32), join(outdir, f'{key}.tensor'), True)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

# import yaml
# with open('/home/arnavbagad/calibration-tools/opencv/calib_cam0_intrinsics.yaml', 'r') as f:
#     data = yaml.load(f, Loader=yaml.SafeLoader)
     
# intrinsic_yaml_mat = {}

# intrinsic_yaml_mat['camera0'] = np.array(data['camera_matrix']['data']).reshape(3, 3)
# print (intrinsic_yaml_mat['camera0'])

# def cam0_yaml_to_bev(cam0_int_mat):
#     print ("temp")


# def yaw_to_homo(pose_np, yaw):
#     trans = pose_np[:, 1:4]
#     rot_mat = R.from_euler('z', yaw, degrees=True).as_matrix()
#     tmp = np.expand_dims(np.eye(4, dtype=np.float64), axis=0)
#     homo_mat = np.repeat(tmp, len(trans), axis=0)
#     homo_mat[:, :3, :3] = rot_mat
#     # homo_mat[:, :3, 3 ] = trans
#     return homo_mat
