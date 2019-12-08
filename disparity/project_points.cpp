#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <stdio.h>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char const *argv[]) {
  vector< Point3f > objectPoints;
  vector< Point2f > imagePoints;
  Mat R, K;
  Vec3d T;
  Vec4d D;
  cv::FileStorage fs1("/home/shamin/stereo-vision-master/disparity/build/mystereocalib.yml", cv::FileStorage::READ);
  fs1["R"] >> R;
  fs1["T"] >> T;
  fs1["K1"] >> K;
  fs1["D1"] >> D;
  Mat img1 = imread("/home/shamin/stereo-vision-master/disparity/build/left.jpg", CV_LOAD_IMAGE_COLOR);
  objectPoints.push_back(Point3f(-1,0.38,0.1));
  objectPoints.push_back(Point3f(1,0.38,0.1));
  objectPoints.push_back(Point3f(-1,0.38,2));
  objectPoints.push_back(Point3f(1,0.38,2));
  objectPoints.push_back(Point3f(1,0,2));
  objectPoints.push_back(Point3f(-1,0,2));
  fisheye::projectPoints(objectPoints, imagePoints, Vec3d::all(0), Vec3d::all(0), Matx33d(K), D);
  for (int i = 0; i < imagePoints.size(); i++) {
    cout << imagePoints[i] << endl;
    circle(img1, imagePoints[i], 3, Scalar(255,0,0), 2, 8, 0);
  }
  imshow("POINTS", img1);
  waitKey(0);
  return 0;
}