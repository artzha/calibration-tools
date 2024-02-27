# calibration-tools


# Sampling Images

To sample images from a dataset, use the `sample.py` script. The script will sample images from the dataset and save them to a new directory. After doing this, you can directly load these images and point clouds to matlab for calibration.

```bash
python matlab/sample.py -i /robodata/ecocar_logs/processed/CACCDataset -s 44 -o ./sample_outputs -r 10
```

# LiDAR Camera Calibration

Use the MATLAB stereo camera calibrator app to calibrate cameras, save the stereo calibration data to a .mat file

```bash
cam0_params = stereoParams.CameraParameters1;
save("calib_cam0.mat", "cam0_params")
cam1_params = stereoParams.CameraParameters2;
save("calib_cam1.mat", "cam1_params")
save("calib_cam0_cam1.mat", "stereoParams", "estimationErrors")
```

```bash
raddist = cam0_params.Intrinsics.RadialDistortion;
tandist = cam0_params.Intrinsics.TangentialDistortion;
K = cam0_params.Intrinsics.K;
save("K_cam0.mat", "K", "raddist", "tandist")

raddist = cam1_params.Intrinsics.RadialDistortion;
tandist = cam1_params.Intrinsics.TangentialDistortion;
K = cam1_params.Intrinsics.K;
save("K_cam1.mat", "K", "raddist", "tandist")

A = stereoParams.PoseCamera2.A;
save("T_cam0_cam1.mat", "A")
```

Then, use the MATLAB LiDAR camera calibrator app to calibrate the LiDAR to the camera. Use fixed intrinsics from
the stereoParams calibration session for the camera. Save the LiDAR camera calibration data to a .mat file.

```bash
A = tform.A
save("T_os1_cam0.mat", "A", "errors")
```

# Postprocess to ROS Compatible Format

Use the `postprocess.py` script to convert the calibration data to a ROS compatible format. The script will save the calibration data to a .yaml file.

```bash
python matlab/convert.py -i matlab_data -s 44 -o /robodata/ecocar_logs/processed/CACCDataset/calibrations
```