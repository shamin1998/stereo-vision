#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char const *argv[])
{
  cv::Mat R1, R2, P1, P2, Q;
  cv::Mat K1, K2, R;
  cv::Vec3d T;
  cv::Vec4d D1, D2;
  cv::Mat img1 = imread("/home/shamin/stereo-vision-master/disparity/scene_imgs/13/left41.jpg", CV_LOAD_IMAGE_COLOR);
  cv::Mat img2 = imread("/home/shamin/stereo-vision-master/disparity/scene_imgs/13/right41.jpg", CV_LOAD_IMAGE_COLOR);

  cv::FileStorage fs1("/home/shamin/stereo-vision-master/disparity/build/mystereocalib.yml", cv::FileStorage::READ);
  fs1["K1"] >> K1;
  fs1["K2"] >> K2;
  fs1["D1"] >> D1;
  fs1["D2"] >> D2;
  fs1["R"] >> R;
  fs1["T"] >> T;
  

  fs1["R1"] >> R1;
  fs1["R2"] >> R2;
  fs1["P1"] >> P1;
  fs1["P2"] >> P2;
  fs1["Q"] >> Q;

  cv::Mat lmapx, lmapy, rmapx, rmapy;
  cv::Mat imgU1, imgU2;
  
  Size img_size = img1.size();

  cv::fisheye::initUndistortRectifyMap(K1, D1, R1, P1, img_size, CV_32F, lmapx, lmapy);
  cv::fisheye::initUndistortRectifyMap(K2, D2, R2, P2, img_size, CV_32F, rmapx, rmapy);
  cv::remap(img1, imgU1, lmapx, lmapy, cv::INTER_LINEAR);
  cv::remap(img2, imgU2, rmapx, rmapy, cv::INTER_LINEAR);
  
  imshow("image1", imgU1);
  imshow("image2", imgU2);
  int k = waitKey(0);
  if (k > 0) {
    cout << "writing to file..." << endl;
    imwrite("left.jpg", imgU1);
    imwrite("right.jpg", imgU2);
  }
  return 0;
}
