#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <cmath>

using namespace cv;
using namespace std;

vector<Point2f> points1;
vector<Point2f> points2;
vector<Vec3f> lines1;
vector<Vec3f> lines2;

clock_t startT, endT;
double cpu_time_used;
RNG rng(12345);

Mat img1, img2, img3, img4, F, disp_left, disp_right, disp_left_corrected;
Mat img1_distorted, img2_distorted, img5, img6;
Mat winImg, grad_left, grad_right, grad_dir_left, grad_dir_right;
int w = 9;
int channels = 3;
float sd_d = 7;
float sd_s = 20;
Point2d cur_left_pt;

Mat K1, K2;
Vec4d D1, D2;

Point2d getDistortedPoint(Point2d p, Mat& C, Vec4d& D) {
  double x = (p.x - C.at<double>(0,2)) / C.at<double>(0,0);
  double y = (p.y - C.at<double>(1,2)) / C.at<double>(1,1);

  double r2 = x*x + y*y;

  double theta = atan(sqrt(r2));
  double theta_d = theta*(1. + D[0]*theta*theta + D[1]*theta*theta*theta*theta);
  
  double xDistort = theta_d * x / sqrt(r2);
  double yDistort = theta_d * y / sqrt(r2);

  xDistort = xDistort * C.at<double>(0,0) + C.at<double>(0,2);
  yDistort = yDistort * C.at<double>(1,1) + C.at<double>(1,2);
  return Point2d(xDistort, yDistort);
}

bool inImg(int x, int y) {
  if (x >= 0 && x < img1.cols && y >= 0 && y < img1.rows)
    return true;
}

pair< Point, bool > findCorresPointRight(Mat img1, Mat img2, Point p, LineIterator it2) {
  Point match_pos = p;
  float min_error = 10000000000;
  vector< long > errors;
  Point2d p_dis = getDistortedPoint(p, K1, D1);
  for(int i = 0; i < it2.count; i+=2)
  {
    Point cur_pt = it2.pos();
		int disp = cur_pt.x - p.x;
    if (abs(disp) > 70) {
      for (int z = 0; z < 2; z++) it2++;
      continue;
    }
    Point2d cur_pt_dis = getDistortedPoint(Point2d(cur_pt.x, cur_pt.y), K2, D2);
    float error = 0;
    //float error = costFunction(img1_distorted, img2_distorted, Point(p_dis.x, p_dis.y), Point(cur_pt_dis.x, cur_pt_dis.y), "SAD");
    for (int k = -w; k <= w; k++) {
      for (int l = -w; l <= w; l++) {
        for (int ch = 0; ch < channels; ch++) {
          error += abs(img1_distorted.at<Vec3b>(k+p_dis.y,l+p_dis.x)[ch] - img2_distorted.at<Vec3b>(k+cur_pt_dis.y,l+cur_pt_dis.x)[ch]);
        }
      }
    }
    if (error < min_error) {
			min_error = error;
      match_pos = cur_pt;
    }
    errors.push_back(error);
    for (int z = 0; z < 2; z++) ++it2;
  }
  bool good_match = true;
  return make_pair(match_pos, good_match);
}

pair< Point, bool > findCorresPointLeft(Mat img1, Mat img2, Point p, LineIterator it2) {
  Point match_pos = p;
  long min_error = 10000000000;
  int error_threshold = 10000;
  vector< long > errors;
  for(int i = 0; i < it2.count; i+=4)
  {
    Point cur_pt = it2.pos();
    int disp = cur_pt.x - p.x;
    if (abs(disp) > 60) {
      for (int z = 0; z < 4; z++) it2++;
      continue;
    }
    long error = 0;
    int w = 7;
    for (int k = -w; k <= w; k++) {
      for (int j = -w; j <= w; j++) {
        if (inImg(p.x + k, p.y + j) && inImg(cur_pt.x + k, cur_pt.y + j)) {
          
          
          error += (long)(abs((img1.at<Vec3b>(p.y + j, p.x + k)[0] - img2.at<Vec3b>(cur_pt.y + j, cur_pt.x + k)[0]))
                      +abs((img1.at<Vec3b>(p.y + j, p.x + k)[1] - img2.at<Vec3b>(cur_pt.y + j, cur_pt.x + k)[1]))
                      +abs((img1.at<Vec3b>(p.y + j, p.x + k)[2] - img2.at<Vec3b>(cur_pt.y + j, cur_pt.x + k)[2])));
        
          
        }
      }
    }
    if (error < min_error) {
      min_error = error;
      match_pos = cur_pt;
    }
    errors.push_back(error);
    for (int z = 0; z < 4; z++) ++it2;
  }
  bool good_match = true;
  return make_pair(match_pos, good_match);
}

void generateLeftDisparityMap() {
  int width = img1.cols;
  int height = img1.rows;
  disp_left = Mat(height, width, CV_8UC1, Scalar(0));
  for (int i = 260; i < 610; i+=1) {
    for (int j = 154; j < 422; j+=1) {
      points1.push_back(Point2f(i, j));
    }
  }
  vector< Point2f > right_pts;
  computeCorrespondEpilines(Mat(points1), 1, F, lines1);
  vector<cv::Vec3f>::const_iterator it = lines1.begin();
  int x = 0;
  for (int i = 260; i < 610; i+=1) {
    for (int j = 154; j < 422; j+=1) {
      LineIterator it2(img2, Point(0,-(*it)[2]/(*it)[1]), Point(img2.cols,-((*it)[2]+(*it)[0]*img2.cols)/(*it)[1]), 8);
      pair< Point, bool > match = findCorresPointRight(img1, img2, points1[x], it2);
      right_pts.push_back(match.first);
      int disparity = abs(i-match.first.x);
      if (match.second) {
        disp_left.at<uchar>(j,i) = ((float)disparity / (float)(70. - 0.)) * 255.0;
      }
      it++;
      x++;
      cout << i << " " << j << endl;
    }
  }
}

void generateRightDisparityMap() {
  int width = img2.cols;
  int height = img2.rows;
  disp_right = Mat(height, width, CV_8UC3, Scalar(0,0,0));
  for (int i = 0; i < width; i+=2) {
    for (int j = 0; j < height; j+=2) {
      points2.push_back(Point2f(i, j));
    }
  }
  computeCorrespondEpilines(Mat(points2), 2, F, lines2);
  vector<cv::Vec3f>::const_iterator it = lines2.begin();
  int x = 0;
  for (int i = 0; i < width; i+=2) {
    for (int j = 0; j < height; j+=2) {
      LineIterator it2(img3, Point(0,-(*it)[2]/(*it)[1]), Point(img1.cols,-((*it)[2]+(*it)[0]*img1.cols)/(*it)[1]), 8);
      pair< Point, bool > match = findCorresPointLeft(img3, img4, points2[x], it2);
      int disparity = ((float)abs(i-match.first.x) / (float)(60. - 0.)) * 255.0;
      int disparity_y = abs(j-match.first.y);
      if (match.second) {
        int left_disparity_x = disp_left.at<Vec3b>(match.first.y,match.first.x)[2];
        int left_disparity_y = disp_left.at<Vec3b>(match.first.y,match.first.x)[0];
        int check_left_right = abs(left_disparity_x - disparity) + abs(left_disparity_y-disparity_y);
        cout << "L-R: " << check_left_right << endl;
        if (check_left_right < 100)
          disp_right.at<Vec3b>(j,i)[2] = disparity;
        else
          disp_right.at<Vec3b>(j,i)[1] = 200;
      } else {
        disp_right.at<Vec3b>(j,i)[1] = 200;
      }
      it++;
      x++;
      cout << i << " " << j << endl;
    }
  }
}

int main(int argc, char const *argv[])
{
  FileStorage fs2("/home/shamin/stereo-vision-master/disparity/map/build/mystereocalib.yml", FileStorage::READ);
  fs2["F"] >> F;
  fs2["K1"] >> K1;
  fs2["K2"] >> K2;
  fs2["D1"] >> D1;
  fs2["D2"] >> D2;
  img1 = imread("/home/shamin/stereo-vision-master/disparity/map/build/left.jpg");
  img2 = imread("/home/shamin/stereo-vision-master/disparity/map/build/right.jpg");
  img1_distorted = imread("/home/shamin/stereo-vision-master/disparity/map/scene_imgs/12/left3.jpg", CV_LOAD_IMAGE_COLOR);
  img2_distorted = imread("/home/shamin/stereo-vision-master/disparity/map/scene_imgs/12/right3.jpg", CV_LOAD_IMAGE_COLOR);
  namedWindow("LEFT", 1);
  namedWindow("RIGHT", 1);
  generateLeftDisparityMap();
  while (1) {
    imshow("LEFT", img1);
    imshow("RIGHT", img2);
    imshow("LEFT_MAP", disp_left);
    if (waitKey(0) > 0){
      imwrite("/home/shamin/stereo-vision-master/disparity/map/build/disp_left.jpg", disp_left);
      break;
    }
  }
  return 0;
}