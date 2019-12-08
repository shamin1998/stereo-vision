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

vector<Point2d> points1;
vector<Point2d> points2;
vector<Vec3f> lines1;

clock_t startT, endT;
double cpu_time_used;
RNG rng(12345);

Mat img1, img2, img3, img4, F, img1_distorted, img2_distorted, img5, img6;
Mat winImg, grad_left, grad_right, grad_dir_left, grad_dir_right, img_res1, img_res2;
int w = 10;
int channels = 3;
float sd_d = 100;
float sd_s = 16;
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

float weight_bilateral_ANCC(Point p, Point t, Mat img, int channel) {
  float e1 = dist(p,t)/2*sd_d;
  float e2 = (img.at<Vec3b>(p.y,p.x)[channel] - img.at<Vec3b>(p.y,p.x)[channel])*(img.at<Vec3b>(p.y,p.x)[channel] - img.at<Vec3b>(p.y,p.x)[channel]) / 2*sd_s;
  return exp(-e1-e2);
}

float sum_bilateral(Point p, Mat img, int ch) {
  float z = 0.;
  float sum = 0.;
  for (int i = -w; i <= w; i++) {
    for (int j = -w; j <= w; j++) {
      z += weight_bilateral_ANCC(Point(p.x+i,p.y+j), p, img, ch);
      sum += weight_bilateral_ANCC(Point(p.x+i,p.y+j), p, img, ch) * K(Point(p.x+i,p.y+j), img, ch);
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
          if (inImg(p1.x+i,p1.y+j) && inImg(p2.x+i,p2.y+j)) {
            ch1_mean[ch] += img1.at<Vec3b>(j+p1.y,i+p1.x)[ch];
            ch2_mean[ch] += img2.at<Vec3b>(j+p2.y,i+p2.x)[ch];
          }
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
          if (inImg(p1.x+i,p1.y+j) && inImg(p2.x+i,p2.y+j)) {
            num[ch] += ((float)img1.at<Vec3b>(j+p1.y,i+p1.x)[ch] - ch1_mean[ch]) * ((float)img2.at<Vec3b>(j+p2.y,i+p2.x)[ch] - ch2_mean[ch]);
            denom1[ch] += ((float)img1.at<Vec3b>(j+p1.y,i+p1.x)[ch] - ch1_mean[ch]) * ((float)img1.at<Vec3b>(j+p1.y,i+p1.x)[ch] - ch1_mean[ch]);
            denom2[ch] += (float)(img2.at<Vec3b>(j+p2.y,i+p2.x)[ch] - ch2_mean[ch]) * (float)(img2.at<Vec3b>(j+p2.y,i+p2.x)[ch] - ch2_mean[ch]);
          }
        }
      }
    }
    for (int i = 0; i < channels; i++) {
      if (denom1[i] > 0.000001 && denom2[i] > 0.000001)
        cost_ch[i] = num[i] / (sqrt(denom1[i]) * sqrt(denom2[i]));
      else
        cost_ch[i] = 0.;
      cost += cost_ch[i];
    }
    cost /= 3.;
  } else if (method == "SAD") {
    for (int i = -w; i <= w; i++) {
      for (int j = -w; j <= w; j++) {
        for (int ch = 0; ch < channels; ch++) {
          if (inImg(p1.x+i,p1.y+j) && inImg(p2.x+i,p2.y+j))
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
          WL[idx][ch] = weight_bilateral_ANCC(Point(i+p1.x,j+p1.y), p1, img1, ch);
          VR[idx][ch] = K(Point(i+p2.x,j+p2.y), img2, ch) - BSUMR[ch];
          WR[idx][ch] = weight_bilateral_ANCC(Point(i+p2.x,j+p2.y), p2, img2, ch);
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
    cost = cost / 3.;
  } else if (method == "ANG") {
    float denom = 0.;
    float num = 0.;
    for (int i = -w; i <= w; i++) {
      for (int j = -w; j <= w; j++) {
        float error = 0.;
        float weight =  weight_bilateral(Point(p1.x+i,p1.y+j), p1, img1, 0) * weight_bilateral(Point(p2.x+i,p2.y+j), p2, img2, 0);
        for (int ch = 0; ch < channels; ch++) {
          error += abs(img1.at<Vec3b>(p1.y+j,p1.x+i)[ch] - img2.at<Vec3b>(p2.y+j,p2.x+i)[ch]);
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

Point findCorresPoint(Mat img1, Mat img2, Point2d p) {
  ofstream myfile;
  myfile.open ("error.txt");
  Point2f match_pos = p;
  cur_left_pt = p;
  float min_error = 1e9;
  vector< float > errors;
  int max_disp = 100;
  startT = clock();
  int offset = 60;
  Point2d p_dis = getDistortedPoint(p, K1, D1);
  for(int i = p.x - offset; i < p.x + offset; i+=2)
  {
    Point2d cur_pt((double)i, p.y);
    int disp = cur_pt.x - p.x;
    float error = costFunction(img1, img2, p, cur_pt, "SAD");
    if (error < min_error) {
      min_error = error;
      match_pos = cur_pt;
    }
    errors.push_back(error);
    myfile << error << endl;
  }
  float Y = (p.y - 305.) / (4.2 * (float)abs(p.x - match_pos.x));
  endT = clock();
  cpu_time_used = ((double) (endT - startT)) / CLOCKS_PER_SEC;
  cout << "CPU TIME: " << cpu_time_used << endl;
  myfile.close();
  cout << "Error: " << min_error << endl;
  cout << "Position: " << match_pos.x << " , " << match_pos.y << endl;
  return match_pos;
}

void generateWindowVisualization(Mat img1, Mat img2, Point p, Point match_pt) {
  int refx = w, refy = w;
  for (int i = -w; i <= w; i++) {
    for (int j = -w; j <= w; j++) {
      winImg.at<Vec3b>(j+refy,i+refx) = img1.at<Vec3b>(p.y+j,p.x+i);
    }
  }
  refx = 3*w+1;
  for (int i = -w; i <= w; i++) {
    for (int j = -w; j <= w; j++) {
      winImg.at<Vec3b>(j+refy,i+refx) = img2.at<Vec3b>(match_pt.y+j,match_pt.x+i);
    }
  }
}

void mouseClick(int event, int x, int y, int flags, void* userdata) {
  if (event == EVENT_LBUTTONDOWN) {
    cout << "Clicked: (" << x << ", " << y << ")" << endl;
    Point2d p((double)x, (double)y);
    points1.push_back(p);
    Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255));
    circle(img1, p, 3, color, 2, 8, 0);
    Point match_pt = findCorresPoint(img3, img4, points1[points1.size()-1]);
    generateWindowVisualization(img3, img4, p, match_pt);
    circle(img2, match_pt, 3, color, 2, 8, 0);
   
  }
}

void mouseClickRight(int event, int x, int y, int flags, void* userdata) {
  if (event == EVENT_LBUTTONDOWN) {
    Point2d p((double)x, (double)y);
    for (int i = -w; i <= w; i++) {
      for (int j = -w; j <= w; j++) {
        winImg.at<Vec3b>(j+w,i+5*w+2) = img4.at<Vec3b>(y+j,x+i);
      }
    }
    float error = costFunction(img3, img4, cur_left_pt, p, "NCC");
    cout << "Manual window error: " << error << endl;
    Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255));
  }
}

void convertToSingleChannel(Mat& img, int channel, string method) {
  for (int i = 0; i < img1.cols; i++) {
    for (int j = 0; j < img1.rows; j++) {
      if (method == "lightness")
        img.at<Vec3b>(j,i)[channel] = (float)(max(img.at<Vec3b>(j,i)[0], max(img.at<Vec3b>(j,i)[1], img.at<Vec3b>(j,i)[2])) + min(img.at<Vec3b>(j,i)[0], min(img.at<Vec3b>(j,i)[1], img.at<Vec3b>(j,i)[2]))) / 2.;
      else if (method == "average")
        img.at<Vec3b>(j,i)[channel] = ((float)img.at<Vec3b>(j,i)[2] + (float)img.at<Vec3b>(j,i)[1] + (float)img.at<Vec3b>(j,i)[0]) / 3.;
      else if (method == "luminosity")
        img.at<Vec3b>(j,i)[channel] = 0.21 * (float)img.at<Vec3b>(j,i)[2] + 0.72 * (float)img.at<Vec3b>(j,i)[1] + 0.07 * (float)img.at<Vec3b>(j,i)[0];
      for (int ch = 0; ch < 3; ch++) {
        if (ch != channel) img.at<Vec3b>(j,i)[ch] = 0;
      }
    }
  }
}

void normalizeChannels(Mat& img1, Mat& img2, int channels) {
  float img1_mean[channels], img2_mean[channels];
  for (int i = 0; i < img1.cols; i++) {
    for (int j = 0; j < img1.rows; j++) {
      for (int ch = 0; ch < channels; ch++) {
        img1_mean[ch] += img1.at<Vec3b>(j,i)[ch];
        img2_mean[ch] += img2.at<Vec3b>(j,i)[ch];
      }
    }
  }
  for (int ch = 0; ch < channels; ch++) {
    img1_mean[ch] /= (float)(img1.cols * img1.rows);
    img2_mean[ch] /= (float)(img1.cols * img1.rows);
    if (img1_mean[ch] < img2_mean[ch]) {
      float ratio = img1_mean[ch] / img2_mean[ch];
      for (int i = 0; i < img1.cols; i++) {
        for (int j = 0; j < img1.rows; j++) {
          float a = ratio * (float)img2.at<Vec3b>(j,i)[ch];
          if (a > 255) a = 255;
          img2.at<Vec3b>(j,i)[ch] = a;
        }
      }
    } else {
      float ratio = img2_mean[ch] / img1_mean[ch];
      for (int i = 0; i < img1.cols; i++) {
        for (int j = 0; j < img1.rows; j++) {
          float a = ratio * (float)img1.at<Vec3b>(j,i)[ch];
          if (a > 255) a = 255;
          img1.at<Vec3b>(j,i)[ch] = a;
        }
      }
    }
  }
}

void computeGradients(Mat& img, Mat& grad, Mat& grad_dir) {
  grad = Mat(img.rows, img.cols, CV_32FC3, Scalar(0,0,0));
  grad_dir = Mat(img.rows, img.cols, CV_32FC3, Scalar(0,0,0));
  for (int i = 1; i < img.cols-1; i++) {
    for (int j = 1; j < img.rows-1; j++) {
      float gx, gy;
      for (int ch = 0; ch < channels; ch++) {
        gx = img.at<Vec3b>(j,i+1)[ch] - img.at<Vec3b>(j,i-1)[ch];
        gy = img.at<Vec3b>(j+1,i)[ch] - img.at<Vec3b>(j-1,i)[ch];
        grad.at<Vec3f>(j,i)[ch] = sqrt(gx*gx + gy*gy);
        grad_dir.at<Vec3f>(j,i)[ch] = atan2(gy, gx);
      }
    }
  }
}

int main(int argc, char const *argv[])
{
  FileStorage fs("/home/shamin/stereo-vision-master/disparity//build/mystereocalib.yml", FileStorage::READ);
  K1 = Mat(3, 3, CV_32F);
  K2 = Mat(3, 3, CV_32F);
  fs["F"] >> F;
  fs["K1"] >> K1;
  fs["D1"] >> D1;
  fs["K2"] >> K2;
  fs["D2"] >> D2;
  winImg = Mat(2*w+1, 3*(2*w + 1), CV_8UC3, Scalar(0,0,0));
  img1 = imread("/home/shamin/stereo-vision-master/disparity/000088_11.png");
  img2 = imread("/home/shamin/stereo-vision-master/disparity/000088_11.png");
  resize(img1, img_res1, Size(662, 200));
  resize(img2, img_res2, Size(662, 200));
  img1 = img_res1.clone();
  img2 = img_res2.clone();
  img3 = img1.clone();
  img4 = img2.clone();
  namedWindow("LEFT", 1);
  namedWindow("RIGHT", 1);
  namedWindow("WINDOWVISUAL", 1);
  setMouseCallback("LEFT", mouseClick, NULL);
  setMouseCallback("RIGHT", mouseClickRight, NULL);
  while (1) {
    imshow("LEFT", img1);
    imshow("RIGHT", img2);
    imshow("WINDOWVISUAL", winImg);
    int k = waitKey(10);
    if (k == ' ') break;
  }
  destroyWindow("LEFT");
  destroyWindow("RIGHT");
  destroyWindow("WINDOWVISUAL");
  return 0;
}