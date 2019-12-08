#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <time.h>

using namespace cv;
using namespace std;

vector<Point2f> points1;
vector<Point2f> points2;
vector<Vec3f> lines1;

clock_t startT, endT;
double cpu_time_used;
RNG rng(12345);

Mat img1, img2, img3, img4, F;

bool inImg(int x, int y) {
  if (x >= 0 && x < img1.cols && y >= 0 && y < img1.rows)
    return true;
}

Point findCorresPoint(Mat img1, Mat img2, Point2f p, LineIterator it2) {
  ofstream myfile;
  myfile.open ("error.txt");
  Point2f match_pos = p;
  float min_error = 100000000;
  vector< float > errors;
  int max_disp = 100;
  startT = clock();
  //vector< pair<Point2f, int> > error_pos;
  Point prev_pt;
  for(int i = 0; i < it2.count; i+=1)
  {
    Point cur_pt = it2.pos();
    int disp = cur_pt.x - p.x;
    if (abs(disp) > 70) {
      for (int z = 0; z < 1; z++) it2++;
      continue;
    }
    float error = 0.;
    int w = 9;
    for (int k = -w; k <= w; k++) {
      for (int j = -w; j <= w; j++) {
        if (inImg(p.x + k, p.y + j) && inImg(cur_pt.x + k, cur_pt.y + j)) {
          
          error += (float)(abs((int)img2.at<Vec3b>(p.y + j, p.x + k)[0] - (int)img1.at<Vec3b>(cur_pt.y + j, cur_pt.x + k)[0])
                      +abs((int)img2.at<Vec3b>(p.y + j, p.x + k)[1] - (int)img1.at<Vec3b>(cur_pt.y + j, cur_pt.x + k)[1])
                      +abs((int)img2.at<Vec3b>(p.y + j, p.x + k)[2] - (int)img1.at<Vec3b>(cur_pt.y + j, cur_pt.x + k)[2]));
        }
      }
    }
    if (error < min_error) {
      min_error = error;
      match_pos = cur_pt;
      //error_pos.push_back(make_pair(cur_pt, error));
    }
    errors.push_back(error);
    myfile << error << endl;
    for (int z = 0; z < 1; z++) it2++;
    prev_pt = cur_pt;
  }
  for (int i = 0; i < errors.size(); i++) {
    cout << errors[i] << endl;
  }
  //imshow("TMEP", img2);
  //waitKey(0);
  endT = clock();
  cpu_time_used = ((double) (endT - startT)) / CLOCKS_PER_SEC;
  cout << "CPU TIME: " << cpu_time_used << endl;
  myfile.close();
  cout << "Error: " << min_error << endl;
  cout << "Position: " << match_pos.x << " , " << match_pos.y << endl;
  return match_pos;
}

void mouseClick(int event, int x, int y, int flags, void* userdata) {
  if (event == EVENT_LBUTTONDOWN) {
    cout << "Clicked: (" << x << ", " << y << ")" << endl;
    Point2f p((float)x, (float)y);
    points1.push_back(p);
    Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255));
    circle(img2, p, 3, color, 2, 8, 0);
    
    computeCorrespondEpilines(Mat(points1), 2, F, lines1);
    vector<cv::Vec3f>::const_iterator it = lines1.end()-1;
    cv::line(img1,cv::Point(0,-(*it)[2]/(*it)[1]),
                 cv::Point(img2.cols,-((*it)[2]+(*it)[0]*img2.cols)/(*it)[1]),
                 cv::Scalar(255,255,255));
    LineIterator it2(img3, Point(0,-(*it)[2]/(*it)[1]), Point(img1.cols,-((*it)[2]+(*it)[0]*img1.cols)/(*it)[1]), 8, true);
    Point match_pt = findCorresPoint(img3, img4, points1[points1.size()-1], it2);
    
    circle(img1, match_pt, 3, color, 2, 8, 0);
  }
}

int main(int argc, char const *argv[])
{
  FileStorage fs2("/home/shamin/stereo-vision-master/disparity/build/mystereocalib.yml", FileStorage::READ);
  fs2["F"] >> F;
  img1 = imread("left.jpg");
  img2 = imread("right.jpg");
  img3 = imread("left.jpg");
  img4 = imread("right.jpg");
  namedWindow("LEFT", 1);
  namedWindow("RIGHT", 1);
  setMouseCallback("RIGHT", mouseClick, NULL);
  while (1) {
    imshow("LEFT", img1);
    imshow("RIGHT", img2);
    int k = waitKey(10);
    if (k == ' ') break;
  }
  return 0;
}