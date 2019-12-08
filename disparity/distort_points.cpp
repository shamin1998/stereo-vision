#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>

using namespace std;
using namespace cv;

RNG rng(12345);
vector< Point2d > undis_pts, dis_pts;
Mat img1, img1_undistorted;
cv::Mat K;
cv::Vec4d D;

Point2d getDistortedPoint(Point2d p) {
  double x = (p.x - K.at<double>(0,2)) / K.at<double>(0,0);
  double y = (p.y - K.at<double>(1,2)) / K.at<double>(1,1);

  double r2 = x*x + y*y;

  double theta = atan(sqrt(r2));
  double theta_d = theta*(1. + D[0]*theta*theta + D[1]*theta*theta*theta*theta);
  
  double xDistort = theta_d * x / sqrt(r2);
  double yDistort = theta_d * y / sqrt(r2);

  // Back to absolute coordinates.
  xDistort = xDistort * K.at<double>(0,0) + K.at<double>(0,2);
  yDistort = yDistort * K.at<double>(1,1) + K.at<double>(1,2);
  return Point2d(xDistort, yDistort);
}
void mouseClick(int event, int x, int y, int flags, void* userdata) {
  if (event == EVENT_LBUTTONDOWN) {
    cout << "Clicked: (" << x << ", " << y << ")" << endl;
    Point2d p((double)x, (double)y);
    undis_pts.push_back(p);
    dis_pts.push_back(getDistortedPoint(p));
    for (int i = 0; i < undis_pts.size(); i++) {
      Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255));
      circle(img1_undistorted, Point(undis_pts[i].x, undis_pts[i].y), 3, color, 2, 8, 0);
      circle(img1, Point(dis_pts[i].x, dis_pts[i].y), 3, color, 2, 8, 0);
    }
  }
}

int main(int argc, char const *argv[])
{
  cv::FileStorage fs1("/home/shamin/stereo-vision-master/disparity/build/cam_right.yml", cv::FileStorage::READ);
  img1 = imread("/home/shamin/stereo-vision-master/disparity/scene_imgs/9/right1.jpg", CV_LOAD_IMAGE_COLOR);
  img1_undistorted = imread("right.jpg", CV_LOAD_IMAGE_COLOR);
  K = Mat(3, 3, CV_32F);
  fs1["K1"] >> K;
  fs1["D1"] >> D;
  namedWindow("UNDIS", 1);
  namedWindow("DIS", 1);
  setMouseCallback("UNDIS", mouseClick, NULL);
  while (1) {
    imshow("UNDIS", img1_undistorted);
    imshow("DIS", img1);
    waitKey(10);
  }
  return 0;
}
