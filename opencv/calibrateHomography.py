import numpy as np
import cv2
import matplotlib.pyplot as plt

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata  # for interpolation
import ruamel.yaml

def my_represent_float(self, data):
    if 0 < abs(data) < 1e-5:
        return self.represent_scalar(u'tag:yaml.org,2002:float', '{:.15f}'.format(data).rstrip('0').rstrip('.'))
    else:
        # Default representation for other cases (including 0.0, 1.0, etc.)
        return self.represent_scalar(u'tag:yaml.org,2002:float', repr(data))

yaml = ruamel.yaml.YAML(typ='safe', pure=True)
yaml.width = 2
yaml.representer.add_representer(float, my_represent_float)

def save_calibs(calib_dict, calib_dict_path):
    calib_dict_keys = calib_dict.keys()

    calib_yaml_dict = {}
    for key in calib_dict_keys:
        entry_name = key
        entry_data = calib_dict[key]
        entry_rows = entry_data.shape[0]
        entry_cols = entry_data.shape[1]
        calib_yaml_dict[entry_name] = {
            'rows': entry_rows,
            'cols': entry_cols,
            'data': entry_data.reshape(-1).tolist()
        }

    with open(calib_dict_path, 'w') as yaml_file:
        yaml.dump(calib_yaml_dict, yaml_file)

def create_bev_image(rgbd, calib_dict, x_range, y_range, map_res, keypoints=None):
    """
    This function projects the rgbd image into a bird's eye view image and returns
    the transformation from 3D world coordinates to bev
    """
    depth = rgbd[..., 3]
    H, W = depth.shape

    # --- 1. Project depth image into 3D
    K = calib_dict["K"]
    T_lidar_cam = calib_dict["T_lidar_cam0"]
    Ri = np.zeros((4, 3), dtype=np.float32)
    Ri[:3,:3] = np.eye(3)

    x_range = np.array(x_range, dtype=np.float32)
    y_range = np.array(y_range, dtype=np.float32)
    map_res = np.array(map_res, dtype=np.float32)

    # We'll use the depth image to project into 3D space
    T_pixel_lidar = np.linalg.inv(T_lidar_cam) @ Ri @ np.linalg.inv(K)

    # We'll project each pixel into 3D space
    u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))
    u_grid = u_grid.flatten()
    v_grid = v_grid.flatten()
    z_grid = depth.flatten()

    uvd = np.stack([u_grid, v_grid, np.ones_like(u_grid)], axis=0) * z_grid
    xyz_hom = T_pixel_lidar @ uvd
    xyz_hom[3, :] = 1  # homogeneous

    if keypoints is not None:
        keypoints_depth = depth[keypoints[:, 1], keypoints[:, 0]]
        keypoints_uvd = np.hstack([keypoints, np.ones((keypoints.shape[0], 1))]) * keypoints_depth[:, None]
        keypoints_xyz_hom = T_pixel_lidar @ keypoints_uvd.T # shape (4, N) 
        keypoints_xyz_hom[3, :] = 1  # homogeneous

    # Visualize point cloud in 3D

    # --- 2. Project 3D points into bird's eye view
    x_min, x_max = x_range
    y_min, y_max = y_range
    res = map_res[0]

    H_bev = int((x_max - x_min) / res)
    W_bev = int((y_max - y_min) / res)
    print("BEV image size:", H_bev, W_bev)
    # LIDAR coordinate frame (x forward, y left, z up) origin at 0,0
    # BEV image frame (x down, y right, z up) origin at bottom middle
    T_lidar_bev = np.array([
        [-1, 0, 0, H_bev],
        [0, -1, 0, W_bev/2],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    T_lidar_bev[:3, :3] /= res
    xyz_bev = T_lidar_bev @ xyz_hom
    xy_bev = xyz_bev[:2].T.astype(int) # shape (N, 2)
    xy_bev[:, 0] = np.clip(xy_bev[:, 0], 0, H_bev-1)
    xy_bev[:, 1] = np.clip(xy_bev[:, 1], 0, W_bev-1)

    if keypoints is not None:
        keypoints_xyz_bev = T_lidar_bev @ keypoints_xyz_hom
        keypoints_bev = keypoints_xyz_bev[:2].T.astype(int)
        keypoints_bev[:, 0] = np.clip(keypoints_bev[:, 0], 0, H_bev-1)
        keypoints_bev[:, 1] = np.clip(keypoints_bev[:, 1], 0, W_bev-1)

    # --- 3. Create a dense BEV image using xy_bev correspondences and rgb values
    # We'll use griddata to interpolate the points into a dense grid
    # We'll use the RGB image for the interpolation
    rgb = rgbd[..., :3]
    rgb_bev = np.zeros((H_bev, W_bev, 3), dtype=np.uint8)
    rgb_bev[xy_bev[:, 0], xy_bev[:, 1]] = rgb[v_grid, u_grid]

    if keypoints is not None:
        # Draw keypoints on the BEV image
        for i in range(keypoints_bev.shape[0]):
            cv2.circle(rgb_bev, tuple([keypoints_bev[i, 1], keypoints_bev[i, 0]]), 2, (0, 0, 255), -1)

        # Flip keypoints to (u,v) for visualization
        keypoints_bev = keypoints_bev[:, ::-1]
    else:
        keypoints_bev = None

    cv2.imwrite("bev_rgb.png", rgb_bev)

    return rgb_bev, keypoints_bev

def compute_birdseye_one_cam_lidar(
    img_path,
    depth_path,         # Nx3 in LiDAR frame
    calib_dict,         # K, dist, T_lidar_cam
    pattern_size=(7,7),
    square_size=25.0,
    output_size=(800,600)
):
    """
    1) Detect checkerboard corners in the single camera image -> (u,v).
    2) Build a dense depth map from LiDAR => camera image.
    3) For each corner, read depth => backproject corner to 3D in LiDAR coords.
    4) Create a LiDAR bird’s-eye image (top-down) => (uBEV, vBEV).
    5) Convert each 3D corner => bird’s-eye pixel coordinate => (uBEV, vBEV).
    6) Compute homography from camera’s 2D corners => the bird’s-eye 2D corners.
    """
    # --- 1. Load image & LiDAR
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print("Could not load image:", img_path)
        return None, None, None
    H_img, W_img = img_bgr.shape[:2]

    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) # depth in mm
    depth_map_dense = depth.astype(np.float32) / 1000.0  # convert to meters

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # --- 2. Detect checkerboard corners
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_ACCURACY
    found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags)
    if not found:
        print("Checkerboard not found.")
        return None, None, None
    corners = cv2.cornerSubPix(gray, corners, (5,5), (-1,-1),
                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
    corners_2d = corners.reshape(-1,2)  # Nx2
    N = corners_2d.shape[0]

    # optional debug
    tmp_img = img_bgr.copy()
    cv2.drawChessboardCorners(tmp_img, pattern_size, corners, found)
    for i in range(N):
        cv2.putText(tmp_img, str(i), tuple(corners_2d[i].astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.imwrite("debug_corners.png", tmp_img)

    # --- 5. Create a LiDAR bird’s-eye image (top-down) => (uBEV, vBEV)
    x_range = (0, 6.4)
    y_range = (-6.4, 6.4)
    map_res = (0.01, 0.01)
    rgbd = np.concatenate([img_bgr, depth_map_dense[...,None]], axis=2)
    bev_image, corners_2d_bev = create_bev_image(
        rgbd, calib_dict, 
        x_range=x_range,
        y_range=y_range,
        map_res=map_res,
        keypoints=corners_2d.astype(int)
    )
    W_bev, H_bev = bev_image.shape[:2]

    # --- 7. Now we have correspondences: (corners_2d[i]) in camera image => (corners_2d_bev[i]) in LiDAR-BEV image
    valid_idx = []
    valid_cam2d = []
    valid_bev2d = []
    for i in range(N):
        if not np.any(np.isnan(corners_2d_bev[i])):
            # also must be in range
            uu, vv = corners_2d_bev[i]
            if 0<=uu<W_bev and 0<=vv<H_bev:
                valid_idx.append(i)
                valid_cam2d.append(corners_2d[i])
                valid_bev2d.append(corners_2d_bev[i])

    if len(valid_idx)<4:
        print("Not enough valid corners to compute homography.")
        return None, None, None

    valid_cam2d = np.array(valid_cam2d, dtype=np.float32)
    valid_bev2d = np.array(valid_bev2d, dtype=np.float32)

    # compute homography from camera image => bird’s-eye image
    H_cam_to_bev, mask = cv2.findHomography(valid_cam2d, valid_bev2d, cv2.RANSAC, 5.0)
    if H_cam_to_bev is None:
        print("Homography could not be computed.")
        return None, None, None

    # --- 8. Warp the camera image to the bird’s-eye domain
    warp_bev = cv2.warpPerspective(img_bgr, H_cam_to_bev, (H_bev, W_bev))
    cv2.imwrite("testbev.jpg", warp_bev)

    return warp_bev, H_cam_to_bev, corners_2d_bev, (H_bev, W_bev)  

def computeHomography(img_path, homography, bev_size):
    img = cv2.imread(img_path)
    return cv2.warpPerspective(img, homography, bev_size)

if __name__ == "__main__":
    """
    Example usage. You must supply:
      - left_img_path / right_img_path
      - The known stereo intrinsics/extrinsics (M1, d1, M2, d2, R, T).
    """

    # For demonstration, fill these with your actual calibration values:
    M1 = np.array([
        361.8965393293877, -0.10580708462795267, 302.9124819525714, 0.0, 361.64958921744864,
    245.62807244798387, 0.0, 0.0, 1.0], dtype=np.float64).reshape(3,3)
    d1 = np.array([-0.028406035516288036, 0.05071355291011522, 0.0, 0.0, 0.0], dtype=np.float64)
    M2 = np.array([
        362.48105297700414, 0.16800056890221254, 306.8841720577922, 0.0, 362.2021240247964,
    253.79715038007214, 0.0, 0.0, 1.0], dtype=np.float64).reshape(3,3)
    d2 = np.array([-0.014089871827219729, 0.03159598237700081, 0.0, 0.0, 0.0], dtype=np.float64)
    # A typical small translation in X, no translation in Y or Z
    T_cam0_cam1 = np.array([
        0.9982314792427147, 0.004069631357841654, -0.05930726725716707, -0.24597115933945998,
    -0.005386891869888018, 0.9997419624335147, -0.022067848684617093, -0.0008526035684353491,
    0.05920215574524212, 0.022348303071962623, 0.9979958206850968, -0.008848521537961885,
    0.0, 0.0, 0.0, 1.0], dtype=np.float64).reshape(4,4)
    R  = T_cam0_cam1[:3,:3]  # First 3x3 is the rotation
    T  = T_cam0_cam1[:3, 3:4]  # Last column is the translation
    P1 = np.array([
        301.03012018366815, -393.4954656480896, -45.40045466472516, 93.3445435606706,
        195.53182363924407, 7.214001525676499, -425.6448304003085, 72.43464123927534,
        0.9911603035011668, 0.008290221627530311, -0.13241044138904373, 0.13310304886247887
    ], dtype=np.float64).reshape(3,4)
    P2 = np.array([
        301.03012018366803, -393.4954656480897, -45.400454664725224, -4.133167877013471,
        195.53182363924398, 7.214001525676377, -425.6448304003085, 72.43464123927517,
        0.9911603035011669, 0.008290221627529775, -0.13241044138904393, 0.13310304886247867
    ], dtype=np.float64).reshape(3,4)
    T_lidar_cam0 = np.array([
        0.030801394636303485, -0.9994890671410241, -0.00853690541262335, 0.13671433341901681,
        -0.1214300319852262, 0.004735841171152394, -0.9925886958556744, 0.10139171248790284,
        0.9921219791036411, 0.031609752829894144, -0.12122211887897094, 0.1288240665432358,
        0.0, 0.0, 0.0, 1.0
    ], dtype=np.float64).reshape(4,4)

    # left_image_path  = "/home/arthur/AMRL/Tools/amrl_calibration_tools/calibration-tools/data/2d_raw/cam0/42/2d_raw_cam0_42_23.jpg"
    # right_image_path = "/home/arthur/AMRL/Tools/amrl_calibration_tools/calibration-tools/data/2d_raw/cam1/42/2d_raw_cam1_42_23.jpg"

    left_image_path = "/home/arthur/AMRL/Tools/amrl_calibration_tools/calibration-tools/data/2d_raw/cam0/42/2d_raw_cam0_42_XX.jpg"
    depth_image_path = "/home/arthur/AMRL/Tools/amrl_calibration_tools/calibration-tools/data/2d_raw/depth/42/2d_raw_cam0_42_XX.png"

    calib_dict = {
        "K": M1,
        "dist": d1,
        "T_lidar_cam0": T_lidar_cam0,
    }

    homography_dict = None
    bev_size = None
    for i in range(0, 100):
        left_path = left_image_path.replace("XX", str(i))
        depth_path = depth_image_path.replace("XX", str(i))

        if homography_dict is None:
            birdseye, H_img_to_plane, points_3D, bev_size = compute_birdseye_one_cam_lidar(
                left_path,
                depth_path,
                calib_dict,
                pattern_size=(5, 8), 
                square_size=84.5,      # e.g., each square is 25 mm
            )
            homography_dict = {
                "H_img_to_bev":  H_img_to_plane
            }
            save_calibs(homography_dict, "calib_cam0_to_bev.yaml")
        else:
            H = homography_dict["H_img_to_bev"]
            birdseye = computeHomography(left_path, H, bev_size)

        if birdseye is not None:
            cv2.imwrite("testbev.jpg", birdseye)