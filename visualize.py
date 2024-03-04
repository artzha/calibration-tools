"""
Copyright (c) 2023 Arthur King Zhang arthurz@cs.utexas.edu

This script loads point clouds, images and calibrations. It then projects them to the image plane, and saves the results as images.
"""

import os
from os.path import join
import cv2
import numpy as np
import ruamel.yaml
import matplotlib.cm as cm

from scipy.spatial.transform import Rotation as R
from opencv.undistortFisheye import undistortFisheyeSingle

def my_represent_float(self, data):
    if 0 < abs(data) < 1e-5:
        return self.represent_scalar(u'tag:yaml.org,2002:float', '{:.15f}'.format(data).rstrip('0').rstrip('.'))
    else:
        # Default representation for other cases (including 0.0, 1.0, etc.)
        return self.represent_scalar(u'tag:yaml.org,2002:float', repr(data))

yaml = ruamel.yaml.YAML(typ='safe', pure=True)
yaml.width = 2
yaml.representer.add_representer(float, my_represent_float)

import argparse

parser = argparse.ArgumentParser(description='Visualizes camera lidar calibrations')
parser.add_argument('-i', '--indir', type=str, default="/robodata/ecocar_logs/processed/CACCDataset", help='Root directory for dataset')
parser.add_argument('-s', '--sequence', type=int, default=0, help='Sequence to use for the images')
parser.add_argument('-f', '--frame', type=int, default=0, help='Frame to use for the images')
parser.add_argument('-c', '--camid', type=str, default="cam0", help='Frame to use for the images')

def pixels_to_depth(pc_np, calib, IMG_H, IMG_W, return_depth=True, IMG_DEBUG_FLAG=False):
    """
    pc_np:      [N x >=3] point cloud in LiDAR frame
    image_pts   [N x uv]
    calib:      [dict] calibration dictionary
    IMG_W:      [int] image width
    IMG_H:      [int] image height

    Returns depth values in meters
    """
    lidar2camrect = calib['T_lidar_to_cam']

    # Remove points behind camera after coordinate system change
    pc_np = pc_np[:, :3].astype(np.float64) # Remove intensity and scale for opencv
    pc_homo = np.hstack((pc_np, np.ones((pc_np.shape[0], 1))))
    pc_rect_cam = (lidar2camrect @ pc_homo.T).T
    
    lidar_pts= pc_rect_cam / pc_rect_cam[:, -1].reshape(-1, 1)
    MAX_INT32 = np.iinfo(np.int32).max
    MIN_INT32 = np.iinfo(np.int32).min
    lidar_pts = np.clip(lidar_pts, MIN_INT32, MAX_INT32)
    lidar_pts = lidar_pts.astype(np.int32)[:, :2]

    pts_mask = pc_rect_cam[:, 2] > 1

    in_bounds = np.logical_and(
        np.logical_and(lidar_pts[:, 0]>=0, lidar_pts[:, 0]<IMG_W), 
        np.logical_and(lidar_pts[:, 1]>=0, lidar_pts[:, 1]<IMG_H)
    )

    valid_point_mask = in_bounds & pts_mask
    valid_lidar_points  = lidar_pts[valid_point_mask, :]
    valid_lidar_depth   = pc_rect_cam[valid_point_mask, 2] # Use z in cam frame
    
    # Sort lidar depths by farthest to closest
    sortidx = np.argsort(-valid_lidar_depth)
    valid_lidar_depth = valid_lidar_depth[sortidx]
    valid_lidar_points = valid_lidar_points[sortidx, :]

    if IMG_DEBUG_FLAG:
        test_img = np.zeros((IMG_H, IMG_W), dtype=int)
        test_img[valid_lidar_points[:, 1], valid_lidar_points[:, 0]] = 255
        cv2.imwrite("test.png", test_img)

    #1 Create LiDAR depth image
    depth_image_np = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    depth_image_np[valid_lidar_points[:, 1], valid_lidar_points[:, 0]] = valid_lidar_depth
    
    #2 Use cv2 di

    if IMG_DEBUG_FLAG:
        depth_mm = (depth_image_np * 1000).astype(np.uint16)
        cv2.imwrite("pp_depth_max.png", depth_mm)

    if return_depth:
        return depth_image_np
    return valid_lidar_points, valid_lidar_depth

def undistortPinhole(img_path, calib_dict, save_path):
    img_np = cv2.imread(img_path)
    K = np.array(calib_dict['camera_matrix']['data']).reshape(
        calib_dict['camera_matrix']['rows'], calib_dict['camera_matrix']['cols']
    )
    D = np.array(calib_dict['distortion_coefficients']['data']).reshape(
        calib_dict['distortion_coefficients']['rows'], calib_dict['distortion_coefficients']['cols']
    )
    W, H = calib_dict['image_width'], calib_dict['image_height']
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (W,H), 1, (W,H))
    img_undist = cv2.undistort(img_np, K, D, None, new_K)

    x, y, w, h = roi
    img_np = img_undist[y:y+h, x:x+w]
    if save_path is not None:
        cv2.imwrite('visualize.png', img_np)
        return None

    return img_np


def main(args):
    indir = args.indir
    seq = args.sequence
    frame = args.frame
    camid = args.camid

    assert os.path.exists(indir), f'Input folder {indir} does not exist'

    # Load point cloud
    pc_dir = join(indir, "3d_raw", "os1", f'{seq}')
    calib_dir = join(indir, "calibrations", f'{seq}')
    # calib_dir = join("calibration_outputs/calibrations", f'{seq}')

    img_dir = join(indir, "2d_raw", camid, f'{seq}')

    pc_path = join(pc_dir, f'3d_raw_os1_{seq}_{frame}.bin')
    img_path = join(img_dir, f'2d_raw_{camid}_{seq}_{frame}.jpg')
    cam_intrinsics_path = join(calib_dir, f'calib_{camid}_intrinsics.yaml')
    lidarcam_extrinsics_path = join(calib_dir, f'calib_os1_to_{camid}.yaml')

    assert os.path.exists(pc_path), f'Point cloud {pc_path} does not exist'
    if not os.path.exists(img_path): # Robustness to jpg and png
        img_path = img_path.replace(".jpg", ".png")
    assert os.path.exists(img_path), f'Image {img_path} does not exist'
    assert os.path.exists(cam_intrinsics_path), f'Camera intrinsics {cam_intrinsics_path} does not exist'
    assert os.path.exists(lidarcam_extrinsics_path), f'Lidar-camera extrinsics {lidarcam_extrinsics_path} does not exist'
    
    # Load point cloud
    pc_np = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 5)
    with open(cam_intrinsics_path, 'r') as f:
        cam_intrinsics = yaml.load(f)
    with open(lidarcam_extrinsics_path, 'r') as f:
        lidarcam_extrinsics = yaml.load(f)

    
    if cam_intrinsics['distortion_model'] == 'plumb_bob':
        img_np = cv2.imread(img_path)
    elif cam_intrinsics['distortion_model'] == 'fisheye':
        img_np = undistortFisheyeSingle((img_path, cam_intrinsics, None))
    else:
        raise f'Unknown distortion model {cam_intrinsics["distortion_model"]} for camera {camid}'

    # Recompute Camera to LiDAR extrinsics using new camera matrix

    # Project point cloud to image
    IMG_W = cam_intrinsics['image_width']
    IMG_H = cam_intrinsics['image_height']
    T_lidar_to_cam = np.array(lidarcam_extrinsics['extrinsic_matrix']['data']).reshape(
        lidarcam_extrinsics['extrinsic_matrix']['rows'], lidarcam_extrinsics['extrinsic_matrix']['cols']
    )

    # TODO: Delete this
    # import pdb; pdb.set_trace()
    # euler_rot = np.array([102.3646547, 1.9324568, -1.7921589])
    # rot = R.from_euler('xyz', euler_rot, degrees=True)
    # T_lidar_to_cam[:3, :3] = rot.as_matrix()
    # T_lidar_to_cam[:3, :3] = np.array([
    #     [1.0, 0.0, 0.0],
    #     [0.0, 0.0, -1.0],
    #     [0.0, 1.0, 0.0]
    # ])
    # T_lidar_to_cam[:3, 3] = np.array([0.0, 0.0, 0.0])

    K = np.array(cam_intrinsics['camera_matrix']['data']).reshape(
        cam_intrinsics['camera_matrix']['rows'], cam_intrinsics['camera_matrix']['cols']
    )
    T_canon = np.zeros((3, 4))
    T_canon[:3, :3] = np.eye(3)
    T_lidar_to_pixels = K @ T_canon @ T_lidar_to_cam

    calib_dict = {
        'T_lidar_to_cam': T_lidar_to_pixels
    }
    valid_points, valid_depths = pixels_to_depth(pc_np, calib_dict, IMG_H, IMG_W, return_depth=False, IMG_DEBUG_FLAG=True)

    # Color image with depth
    valid_z_map = np.clip(valid_depths, 1, 80)
    norm_valid_z_map = valid_z_map / max(valid_z_map)
    color_map = cm.get_cmap("turbo")(norm_valid_z_map) * 255 # [0,1] to [0, 255]]

    for pt_idx, pt in enumerate(valid_points):
        if color_map[pt_idx].tolist()==[0,0,0]:
            continue # Only circle non background
        img_np = cv2.circle(img_np, (pt[0], pt[1]), radius=2, color=color_map[pt_idx].tolist(), thickness=-1)

    cv2.imwrite("color_depth.png", img_np)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)