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

Mat img1, img2, img3, img4, F, img1_distorted, img2_distorted, img5, img6;
Mat winImg, grad_left, grad_right, grad_dir_left, grad_dir_right;
int w = 11;
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

  // Back to absolute coordinates.
  xDistort = xDistort * C.at<double>(0,0) + C.at<double>(0,2);
  yDistort = yDistort * C.at<double>(1,1) + C.at<double>(1,2);
  return Point2d(xDistort, yDistort);
}

bool inImg(int x, int y) {
  if (x >= 0 && x < img1.cols && y >= 0 && y < img1.rows)
    return true;
}

float K(Point p, Mat img, int ch) {
  float denom = 1.;
  for (int i = 0; i < channels; i++) {
    denom *= img.at<Vec3b>(p.y,p.x)[i];
  }
  denom = pow(denom, 1. / (float)channels);
  return log((float)img.at<Vec3b>(p.y,p.x)[ch] / denom);
}

float dist(Point a, Point b) {
  return sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y));
}

float angle_norm(float angle) {
  return (0 < angle && angle < 3.14159) ? angle : (2*3.14159 - angle);
}

float weight_bilateral(Point p, Point t, Mat img, int channel) {
  float e1 = -dist(p,t)/sd_d;
  float b1 = img.at<Vec3b>(p.y,p.x)[0] - img.at<Vec3b>(t.y,t.x)[0];
  float g1 = img.at<Vec3b>(p.y,p.x)[1] - img.at<Vec3b>(t.y,t.x)[1];
  float r1 = img.at<Vec3b>(p.y,p.x)[2] - img.at<Vec3b>(t.y,t.x)[2];
  float e2 = -sqrt(b1*b1 + g1*g1 + r1*r1)/sd_s;
  return exp(e1);
}

float sum_bilateral(Point p, Mat img, int ch) {
  float z = 0.;
  for (int i = -w; i <= w; i++) {
    for (int j = -w; j <= w; j++) {
      z += weight_bilateral(Point(p.x+i,p.y+j), p, img, ch);
    }
  }
  float sum = 0.;
  for (int i = -w; i <= w; i++) {
    for (int j = -w; j <= w; j++) {
      sum += weight_bilateral(Point(p.x+i,p.y+j), p, img, ch) * K(Point(p.x+i,p.y+j), img, ch);
    }
  }
  return sum / z;
}

float costFunction(Mat img1, Mat img2, Point p1, Point p2, string method) {
  float cost = 0.;
  if (method == "NCC") {
    float ch1_mean[channels], ch2_mean[channels], num[channels], denom1[channels], denom2[channels], cost_ch[channels];
    for (int i = 0; i < channels; i++) {
      ch1_mean[i] = ch2_mean[i] = num[channels] = denom1[channels] = denom2[channels] = cost_ch[channels] = 0;
    }
    float N = (2. * w + 1.)*(2. * w + 1.);
    for (int i = -w; i <= w; i++) {
      for (int j = -w; j <= w; j++) {
        for (int ch = 0; ch < channels; ch++) {
          ch1_mean[ch] += img1.at<Vec3b>(j+p1.y,i+p1.x)[ch];
          ch2_mean[ch] += img2.at<Vec3b>(j+p2.y,i+p2.x)[ch];
        }
      }
    }
    for (int i = 0; i < channels; i++) {
      ch1_mean[i] /= N;
      ch2_mean[i] /= N;
    }
    for (int i = -w; i <= w; i++) {
      for (int j = -w; j <= w; j++) {
        for (int ch = 0; ch < channels; ch++) {
          num[ch] += ((float)img1.at<Vec3b>(j+p1.y,i+p1.x)[ch] - ch1_mean[ch]) * ((float)img2.at<Vec3b>(j+p2.y,i+p2.x)[ch] - ch2_mean[ch]);
          denom1[ch] += ((float)img1.at<Vec3b>(j+p1.y,i+p1.x)[ch] - ch1_mean[ch]) * ((float)img1.at<Vec3b>(j+p1.y,i+p1.x)[ch] - ch1_mean[ch]);
          denom2[ch] += (float)(img2.at<Vec3b>(j+p2.y,i+p2.x)[ch] - ch2_mean[ch]) * (float)(img2.at<Vec3b>(j+p2.y,i+p2.x)[ch] - ch2_mean[ch]);
        }
      }
    }
    for (int i = 0; i < channels; i++) {
      if (denom1[i] < 0.000001 || denom2[i] < 0.000001)
        cost_ch[i] = 0.;
      else
        cost_ch[i] = num[i] / (sqrt(denom1[i]) * sqrt(denom2[i]));
      cost += cost_ch[i];
    }
    cost = 1. - cost / (float)channels;
  } else if (method == "SAD") {
    for (int i = -w; i <= w; i++) {
      for (int j = -w; j <= w; j++) {
        for (int ch = 0; ch < channels; ch++) {
          cost += abs(img1.at<Vec3b>(j+p1.y,i+p1.x)[ch] - img2.at<Vec3b>(j+p2.y,i+p2.x)[ch]);
        }
      }
    }
  } else if (method == "ANCC") {
    int M = (2*w+1)*(2*w+1);
    float VL[M][channels], VR[M][channels];
    float WL[M][channels], WR[M][channels];
    float BSUML[channels], BSUMR[channels];
    for (int i = 0; i < channels; i++) {
      BSUML[i] = sum_bilateral(p1, img1, i);
      BSUMR[i] = sum_bilateral(p2, img2, i);
    }
    int idx = 0;
    for (int i = -w; i <= w; i++) {
      for (int j = -w; j <= w; j++) {
        for (int ch = 0; ch < channels; ch++) {
          VL[idx][ch] = K(Point(i+p1.x,j+p1.y), img1, ch) - BSUML[ch];
          WL[idx][ch] = weight_bilateral(Point(i+p1.x,j+p1.y), p1, img1, ch);
          VR[idx][ch] = K(Point(i+p2.x,j+p2.y), img2, ch) - BSUMR[ch];
          WR[idx][ch] = weight_bilateral(Point(i+p2.x,j+p2.y), p2, img2, ch);
        }
        idx++;
      }
    }
    float ANCC[channels];
    float num[channels], denom1[channels], denom2[channels];
    for (int i = 0; i < channels; i++)
      ANCC[i] = num[i] = denom1[i] = denom2[i] = 0.;
    for (int ch = 0; ch < channels; ch++) {
      for (int m = 0; m < M; m++) {
        num[ch] += WL[m][ch]*WR[m][ch]*VL[m][ch]*VR[m][ch];
        denom1[ch] += (WL[m][ch]*VL[m][ch])*(WL[m][ch]*VL[m][ch]);
        denom2[ch] += (WR[m][ch]*VR[m][ch])*(WR[m][ch]*VR[m][ch]);
      }
      ANCC[ch] = num[ch] / (sqrt(denom1[ch])*sqrt(denom2[ch]));
      num[ch] = denom1[ch] = denom2[ch] = 0.;
    }
    for (int i = 0; i < channels; i++) {
      cost += ANCC[i];
      cout << ANCC[i] << " ";
    }
    cout << endl;
    cost = 1 - cost / 3.;
  } else if (method == "ANG") {
    float denom = 0.;
    float num = 0.;
    for (int i = -w; i <= w; i++) {
      for (int j = -w; j <= w; j++) {
        float error = 0.;
        float weight =  weight_bilateral(Point(p1.x+i,p1.y+j), p1, img1, 0) * weight_bilateral(Point(p2.x+i,p2.y+j), p2, img2, 0);
        for (int ch = 0; ch < channels; ch++) {
          error += abs(img1.at<Vec3b>(p1.y+j,p1.x+i)[ch] - img2.at<Vec3b>(p2.y+j,p2.x+i)[ch]);
          //float m = grad_left.at<Vec3f>(p1.y+j,p1.x+i)[ch] - grad_right.at<Vec3b>(p2.y+j,p2.x+i)[ch];
          //float theta = grad_dir_left.at<Vec3f>(p1.y+j,p1.x+i)[ch] - grad_dir_right.at<Vec3f>(p2.y+j,p2.x+i)[ch];
          //error += abs(m) + angle_norm(abs(theta));
        }
        error = min(error, (float)500.);
        num += weight * error;
        denom += weight;
      }
    }
    cost = num / denom;
  }
  return cost;
}

Point findCorresPoint(Mat img1, Mat img2, Point2d p, LineIterator it2) {
  ofstream myfile;
  myfile.open ("error.txt");
  Point2f match_pos = p;
  float min_error = 100000000;
  vector< float > errors;
  int max_disp = 100;
  startT = clock();
  //vector< pair<Point2f, int> > error_pos;
  Point prev_pt;
  Point2d p_dis = getDistortedPoint(p, K1, D1);
  for(int i = 0; i < it2.count; i+=1)
  {
    Point cur_pt = it2.pos();
    int disp = cur_pt.x - p.x;
    if (abs(disp) > 100) {
      for (int z = 0; z < 1; z++) it2++;
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
    Point2d p_dis = getDistortedPoint(p, K1, D1);
    Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255));
    circle(img1, p, 6, color, 2, 8, 0);
    circle(img5, p_dis, 6, color, 2, 8, 0);
    computeCorrespondEpilines(Mat(points1), 1, F, lines1);
    vector<cv::Vec3f>::const_iterator it = lines1.end()-1;
    cv::line(img2,cv::Point(0,-(*it)[2]/(*it)[1]),
                 cv::Point(img2.cols,-((*it)[2]+(*it)[0]*img2.cols)/(*it)[1]),
                 cv::Scalar(0,0,255));
    LineIterator it2(img4, Point(0,-(*it)[2]/(*it)[1]), Point(img2.cols,-((*it)[2]+(*it)[0]*img2.cols)/(*it)[1]), 8, true);
    Point match_pt = findCorresPoint(img3, img4, points1[points1.size()-1], it2);
    Point2d match_pt_dis = getDistortedPoint(Point2d(match_pt.x, match_pt.y), K2, D2);
    circle(img6, match_pt_dis, 6, color, 2, 8, 0);
    circle(img2, match_pt, 6, color, 2, 8, 0);
  }
}

int main(int argc, char const *argv[])
{
  FileStorage fs2("/home/shamin/stereo-vision-master/disparity/mystereocalib.yml", FileStorage::READ);
  fs2["F"] >> F;
  fs2["K1"] >> K1;
  fs2["K2"] >> K2;
  fs2["D1"] >> D1;
  fs2["D2"] >> D2;
  img1 = imread("left.jpg", 1);
  img2 = imread("right.jpg", 1);
  img1_distorted = imread("/home/shamin/stereo-vision-master/disparity/imgs/left5.jpg", CV_LOAD_IMAGE_COLOR);
  img2_distorted = imread("/home/shamin/stereo-vision-master/disparity/imgs/right5.jpg", CV_LOAD_IMAGE_COLOR);
  img3 = img1.clone();
  img4 = img2.clone();
  img5 = img1_distorted.clone();
  img6 = img2_distorted.clone();
  namedWindow("LEFT", 1);
  namedWindow("RIGHT", 1);
  namedWindow("LEFT_DIS", 1);
  namedWindow("RIGHT_DIS", 1);
  setMouseCallback("LEFT", mouseClick, NULL);
  while (1) {
    imshow("LEFT", img1);
    imshow("RIGHT", img2);
    imshow("LEFT_DIS", img5);
    imshow("RIGHT_DIS", img6);
    int k = waitKey(10);
    if (k == ' ') break;
    /*
    if (k == ' ') {
      imwrite("left_epi.jpg", img1);
      imwrite("right_epi.jpg", img2);
      break;
    }
    */
  }
  return 0;
}