## Stereo Fisheye Calibration with C++ OpenCV

Same as the pinhole package, except for fisheye lens cameras.

### Run Instructions

```bash
mkdir build
cd build
cmake ..
make
```

### Calibration

```bash
./calibrate -n [num_imgs] -s [square_size] -w [board_width] -h [board_height] -d [img_dir] -l [left_img_prefix] -r [right_img_prefix] -o [calib_file]
```