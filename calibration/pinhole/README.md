## StereoCalibration with C++ OpenCV

Package for calibrating separate camera intrinsics and extrinsics for each pair. Code given for 2, can be easily expanded to multiple cameras.

### Run Instructions

```bash
mkdir build 
cd build
cmake ..
make
```

### Intrinsic Calibration

Camera matrix and the distortion coefficients saved in YAML file.

```bash
./calibrate -n [num_imgs] -s [square_size] -w [board_width] -h [board_height] -d [imgs_directory] -i [imgs_filename] -o [file_extension] -e [output_filename]
```

### Extrinisic Calibration

```bash
./calibrate_stereo -n [num_imgs] -u [left_cam_calib] -v [right_cam_calib] -L [left_img_dir] -R [right_img_dir] -l [left_img_prefix] -r [right_img_prefix] -o [output_calib_file] -e [file_extension]
```

### Undistort & Rectify

```bash
./undistort_rectify -c [stereo_calib_file] -l [left_img_path] -r [right_img_path] -L [output_left_img] -R [output_right_img]
```

### Webcam Grab Tool

Helper tool to grab frames from two webcams operating as a stereo pair.

```bash
./read -d [imgs_directory] -e [file_extension] -w [img_width] -h [img_height]
```
Hit any key to capture.