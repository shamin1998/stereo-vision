#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <cmath>
#include <opencv2/features2d/features2d.hpp>

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
Mat msimg1, msimg2, img1_gray, img2_gray;
Mat grad_left, grad_right, grad_dir_left, grad_dir_right;
Mat int_img1, int_img2;
Mat img_res1, img_res2;
int w = 9;
int channels = 3;
float sd_d = 100.;
float sd_s = 16;

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
  return exp(e1 + e2);
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

float costFunction(Mat& img1, Mat& img2, Point p1, Point p2, string method) {
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
        for (int ch = 0; ch < 3; ch++) {
          if (inImg(i+p1.x,j+p1.y) && inImg(i+p2.x,j+p2.y))
            cost += abs(img1.at<Vec3b>(j+p1.y,i+p1.x)[ch] - img2.at<Vec3b>(j+p2.y,i+p2.x)[ch]);
        }
      }
    }
    /*
    Point a1(p1.x-w, p1.y-w);
    Point b1(p1.x+w, p1.y+w);
    for (int ch = 0; ch < channels; ch++) {
      cost += int_img1.at<Vec3f>(a1.y,a1.x)[ch] - int_img1.at<Vec3f>(b1.y,b1.x-1)[ch] - int_img1.at<Vec3f>(b1.y-1,b1.x)[ch] + int_img1.at<Vec3f>(b1.y-1,b1.x-1)[ch];
    }
    */
  } else if (method == "SSD") {
    float term = 0.;
    for (int i = -w; i <= w; i++) {
      for (int j = -w; j <= w; j++) {
        for (int ch = 0; ch < 3; ch++) {
          if (inImg(i+p1.x,j+p1.y) && inImg(i+p2.x,j+p2.y)) {
            term = abs(img1.at<Vec3b>(j+p1.y,i+p1.x)[ch] - img2.at<Vec3b>(j+p2.y,i+p2.x)[ch]);
            cost += term * term;
          }
        }
      }
    }
  } else if (method == "SADGRAY") {
    for (int i = -w; i <= w; i++) {
      for (int j = -w; j <= w; j++) {
        cost += abs(img1.at<uchar>(j+p1.y,i+p1.x) - img2.at<uchar>(j+p2.y,i+p2.x));
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
        if (inImg(p1.x+i,p1.y+j) && inImg(p2.x+i,p2.y+j)) {
          float error = 0.;
          float weight =  weight_bilateral(Point(p1.x+i,p1.y+j), p1, img1, 0) * weight_bilateral(Point(p2.x+i,p2.y+j), p2, img2, 0);
          for (int ch = 0; ch < channels; ch++) {
            error += abs(img1.at<Vec3b>(p1.y+j,p1.x+i)[ch] - img2.at<Vec3b>(p2.y+j,p2.x+i)[ch]);
            //float m = grad_left.at<Vec3f>(p1.y+j,p1.x+i)[ch] - grad_right.at<Vec3b>(p2.y+j,p2.x+i)[ch];
            //float theta = grad_dir_left.at<Vec3f>(p1.y+j,p1.x+i)[ch] - grad_dir_right.at<Vec3f>(p2.y+j,p2.x+i)[ch];
            //error += abs(m) + angle_norm(abs(theta));
          }
          error = min(error, (float)500.);
          num += weight * error * error;
          denom += weight;
        }
      }
    }
    cost = num / denom;
  }
  return cost;
}

pair< Point, bool > findCorresPointLeft(Mat& img1, Mat& img2, Point p) {
  Point match_pos = p;
  float min_error = 1e9;
  int offset = 50;
  for(int i = max(0, p.x - offset); i < min(img1.cols, p.x + offset); i+=2)
  {
    Point cur_pt(i, p.y);
    int disp = cur_pt.x - p.x;
    long error = costFunction(img1, img2, p, cur_pt, "SAD");
    if (error < min_error) {
      min_error = error;
      match_pos = cur_pt;
    }
  }
  bool good_match = true;
  return make_pair(match_pos, good_match);
}

pair< Point, bool > findCorresPointRight(Mat& img1, Mat& img2, Point p) {
  Point match_pos = p;
  float min_error = -1;
  float min_disp_error = 1e9;
  int offset = 70;
  vector< float > errors;
  for(int i = max(0, p.x - offset); i < min(img2.cols, p.x + offset); i+=2)
  {
    Point cur_pt(i, p.y);
    /*
    float left_cost = costFunction(img1, img2, p, cur_pt, "SAD");
    for (int j = max(0, cur_pt.x - offset); j < min(img1.cols, cur_pt.x + offset); j+=2) {
      float right_cost = costFunction(img1, img2, Point(j, p.y), cur_pt, "SAD");
      float error = left_cost*left_cost + right_cost*right_cost;
      if (error < min_error) {
        min_error = error;
        match_pos = cur_pt;
      }
    } 
    */
    /*
    float left_cost = costFunction(img1, img2, p, cur_pt, "SAD");
    pair< Point, bool > potential_match = findCorresPointLeft(img2, img1, cur_pt);
    float right_cost = costFunction(img1, img2, potential_match.first, cur_pt, "SAD");
    float disp_error = left_cost*left_cost + right_cost*right_cost;
    if (disp_error < min_disp_error) {
      min_disp_error = disp_error;
      match_pos = cur_pt;
    }
    */
    float error = costFunction(img1, img2, p, cur_pt, "NCC");
    errors.push_back(error);
    if (error > min_error) {
      min_error = error;
      match_pos = cur_pt;
    }
  }
  bool good_match = true;
  /*
  float error_thresh = 1.2 * min_error;
  int count = 0;
  for (int i = 0; i < errors.size(); i++) {
    if (errors[i] < error_thresh) count++;
  }
  if (count > 30) good_match = false;
  */
  return make_pair(match_pos, good_match);
}

void generateLeftDisparityMap() {
  int width = img1.cols;
  int height = img1.rows;
  disp_left = Mat(img1.rows, img1.cols, CV_8UC1, Scalar(0));
  /*
  msimg1 = Mat(height, width, CV_8UC3, Scalar(0,0,0));
  msimg2 = Mat(height, width, CV_8UC3, Scalar(0,0,0));
  pyrMeanShiftFiltering(img1, msimg1, 5, 5, 3);
  pyrMeanShiftFiltering(img2, msimg2, 5, 5, 3);
  cout << "Performed mean shift." << endl;
  */
  startT = clock();
  for (int i = 0; i < width; i+=1) {
    for (int j = 0; j < height; j+=1) {
      pair< Point, bool > match = findCorresPointRight(img1, img2, Point(i,j));
      int disparity = abs(i-match.first.x);
      if (match.second) {
        disp_left.at<uchar>(j,i) = ((float)disparity / 70.) * 255.0;
      } else {
        cout << "Bad!" << endl;
        disp_left.at<uchar>(j,i) = 0;
      }
      cout << i << " " << j << endl;
    }
  }
  endT = clock();
  cpu_time_used = ((double) (endT - startT)) / CLOCKS_PER_SEC;
  cout << "CPU TIME: " << cpu_time_used << endl;
  /*
  vector< Point2f > out_left_pts, out_right_pts;
  correctMatches(F, points1, right_pts, out_left_pts, out_right_pts);
  disp_left_corrected = Mat(height, width, CV_8UC3, Scalar(0,0,0));
  for (int i = 0; i < out_left_pts.size(); i++) {
    /*
    if (inImg(out_left_pts[i].x, out_left_pts[i].y) && inImg(out_right_pts[i].x, out_right_pts[i].y)) {
      
    }
    int disparity = abs(out_left_pts[i].x - out_right_pts[i].x);
    disp_left_corrected.at<Vec3b>(out_left_pts[i].y, out_left_pts[i].x)[2] = ((float)disparity / (float)(60. - 0.)) * 255.0;
  }
  */
}

void generateRightDisparityMap() {
  int width = img2.cols;
  int height = img2.rows;
  disp_right = Mat(height, width, CV_8UC3, Scalar(0,0,0));
  msimg1 = Mat(height, width, CV_8UC3, Scalar(0,0,0));
  msimg2 = Mat(height, width, CV_8UC3, Scalar(0,0,0));
  disp_left = imread("/home/shamin/stereo-vision-master/disparity/map/build/disp_left.jpg", CV_LOAD_IMAGE_COLOR);
  pyrMeanShiftFiltering(img1, msimg1, 15, 5, 3);
  pyrMeanShiftFiltering(img2, msimg2, 15, 5, 3);
  cout << "Performed mean shift." << endl;
  for (int i = 0; i < width; i+=1) {
    for (int j = 0; j < height; j+=1) {
      pair< Point, bool > match = findCorresPointLeft(msimg1, msimg2, Point(i,j));
      int disparity = abs(i-match.first.x);
      if (match.second) {
        int left_disparity_x = (float)disp_left.at<Vec3b>(match.first.y,match.first.x)[2] * 250. / 255.;
        int check_left_right = abs(left_disparity_x - disparity);
        cout << "L-R: " << check_left_right << endl;
        if (check_left_right < 20)
          disp_right.at<Vec3b>(j,i)[2] = ((float)abs(i-match.first.x) / (float)(250 - 0)) * 255.0;
        else
          disp_right.at<Vec3b>(j,i)[1] = 200;
      } else {
        disp_right.at<Vec3b>(j,i)[1] = 200;
      }
      cout << i << " " << j << endl;
    }
  }
}
//Mat Kimg1, Kimg2;
void generateDepthMap() {
  OrbFeatureDetector detector(500, 1.2f, 8, 31, 0);
  detector.descriptorSize();
  /*
  int width = img1.cols;
  int height = img1.rows;
  vector< KeyPoint > img1_keypts, img2_keypts;
  Mat img1_descriptors, img2_descriptors;
  
  int nfeatures=96000;
  float scaleFactor=1.2f;
  int nlevels=8;
  int edgeThreshold=15; // Changed default (31);
  int firstLevel=0;
  int WTA_K=2;
  int scoreType=ORB::HARRIS_SCORE;
  int patchSize=31;
  
  Ptr<ORB> detector = ORB::create(
    nfeatures,
    scaleFactor,
    nlevels,
    edgeThreshold,
    firstLevel,
    WTA_K,
    scoreType,
    patchSize
  );
  detector->detectAndCompute(img1, Mat(), img1_keypts, img1_descriptors);
  detector->detectAndCompute(img2, Mat(), img2_keypts, img2_descriptors);
  cv::Mat results;
  cv::Mat dists;
  int k=2; // find the 2 nearest neighbors
  if(img1_descriptors.type()==CV_8U)
  {
      // Binary descriptors detected (from ORB or Brief)

      // Create Flann LSH index
      cv::flann::Index flannIndex(img2_descriptors, cv::flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);

      // search (nearest neighbor)
      flannIndex.knnSearch(img1_descriptors, results, dists, k, cv::flann::SearchParams() );
  }
  else
  {
      // assume it is CV_32F
      // Create Flann KDTree index
      cv::flann::Index flannIndex(img2_descriptors, cv::flann::KDTreeIndexParams(), cvflann::FLANN_DIST_EUCLIDEAN);

      // search (nearest neighbor)
      flannIndex.knnSearch(img1_descriptors, results, dists, k, cv::flann::SearchParams() );
  }

  // Conversion to CV_32F if needed
  if(dists.type() == CV_32S)
  {
      cv::Mat temp;
      dists.convertTo(temp, CV_32F);
      dists = temp;
  }
  //drawKeypoints(img1, img1_keypts, Kimg1, Scalar(255,0,0));
  //drawKeypoints(img2, img2_keypts, Kimg2, Scalar(255,0,0));
  
  float nndrRatio = 0.7;
  vector<Point2f> mpts_1, mpts_2; // Used for homography
  for(unsigned int i=0; i<results.rows; ++i)
  {
      // Check if this descriptor matches with those of the objects
      // Apply NNDR
      if(results.at<int>(i,0) >= 0 && results.at<int>(i,1) >= 0 && dists.at<float>(i,0) <= nndrRatio * dists.at<float>(i,1))
      {
          mpts_1.push_back(img1_keypts.at(i).pt);

          mpts_2.push_back(img2_keypts.at(results.at<int>(i,0)).pt);
      }
  }
  for (int i = 0; i < mpts_1.size(); i++) {
    circle(img1, mpts_1[i], 3, Scalar(255,0,0), 2, LINE_8);
    circle(img2, mpts_2[i], 3, Scalar(255,0,0), 2, LINE_8);
  }
  */
}

void normalizeImgs() {
  float img1_mean = 0., img2_mean = 0.;
  for (int i = 0; i < img1_gray.cols; i++) {
    for (int j = 0; j < img1_gray.rows; j++) {
      img1_mean += img1_gray.at<uchar>(j,i);
      img2_mean += img2_gray.at<uchar>(j,i);
    }
  }
  img1_mean /= (float)(img1_gray.rows * img1_gray.cols);
  img2_mean /= (float)(img1_gray.rows * img1_gray.cols);
  if (img1_mean < img2_mean) {
    float ratio = img1_mean / img2_mean;
    for (int i = 0; i < img2_gray.cols; i++) {
      for (int j = 0; j < img2_gray.rows; j++) {
        int a = (int)(ratio * (float)img2_gray.at<uchar>(j,i));
        if (a > 255) a = 255;
        img2_gray.at<uchar>(j,i) = a;
      }
    }
  } else {
    float ratio = img2_mean / img1_mean;
    for (int i = 0; i < img1_gray.cols; i++) {
      for (int j = 0; j < img1_gray.rows; j++) {
        int a = (int)(ratio * (float)img1_gray.at<uchar>(j,i));
        if (a > 255) a = 255;
        img1_gray.at<uchar>(j,i) = a;
      }
    }
  }
}

void groundDisparity() {
  disp_left = Mat(img1.rows, img1.cols, CV_8UC1, Scalar(0));
  float height_from_cam = 0.26;
  float max_disp = -1., min_disp = 1000.;
  for (int i = 330; i < 813; i++) {
    for (int j = 361; j < 471; j++) {
      float disparity = ((float)j - 312.) / (5.17 * height_from_cam);
      max_disp = max(max_disp, disparity);
      min_disp = min(min_disp, disparity);
    }
  }
  for (int i = 330; i < 813; i++) {
    for (int j = 361; j < 471; j++) {
      float disparity = ((float)j - 312.) / (5.17 * height_from_cam);
      disp_left.at<uchar>(j,i) = disparity / (max_disp) * 255.;
    }
  }
}

void markGroundPlane() {
  disp_left = Mat(img1.rows, img1.cols, CV_8UC3, Scalar(0,0,0));
  for (int i = 0; i < img1.cols; i++) {
    for (int j = 282; j < img1.rows; j++) {
      disp_left.at<Vec3b>(j,i)[0] = img1_gray.at<uchar>(j,i);
    }
  }
  for (int i = 0; i < img1.cols; i++) {
    for (int j = 0; j < img1.rows; j++) {
      pair< Point, bool > match = findCorresPointRight(img1_gray, img2_gray, Point(i,j));
      float disparity = abs(i-match.first.x);
      float Y = ((float)j - 286.) / (5.16 * disparity);
      if (Y <= 0.38 && Y >= 0.36) {
        disp_left.at<Vec3b>(j,i)[1] = ((float)disparity / 200.) * 255.0;
      }
      cout << i << " " << j << endl;
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

void markTexturelessRegions() {
  disp_left = Mat(img1.rows, img1.cols, CV_8UC3, Scalar(0,0,0));
  for (int i = 0; i < img1.cols; i++) {
    for (int j = 0; j < img1.rows; j++) {
      float sum = 0.;
      for (int k = -w; k <= w; k++) {
        for (int l = -w; l <= w; l++) {
          if (inImg(i+k, j+l)) {
            for (int ch = 0; ch < channels; ch++) {
              sum += grad_left.at<Vec3f>(j+l,i+k)[ch];
            }
          }
        }
      }
      if (sum < 1000) {
        disp_left.at<Vec3b>(j,i)[2] = 255;
      } else {
        disp_left.at<Vec3b>(j,i) = img1.at<Vec3b>(j,i);
      }
    }
  }
}

void postProcessDisparityMap() {
  Mat new_disp;
  adaptiveBilateralFilter(disp_left, new_disp, Size(5,5), 10, 20, Point(-1,-1), BORDER_DEFAULT);
  disp_left = new_disp.clone();
}

void computeIntegralImgs() {
  int_img1 = Mat(img1.rows, img1.cols, CV_32FC3, Scalar(0,0,0));
  for (int i = 0; i < img1.cols; i++) {
    for (int j = 0; j < img1.rows; j++) {
      for (int ch = 0; ch < channels; ch++) {
        float a1 = (i == 0) ? 0 : int_img1.at<Vec3f>(j,i-1)[ch];
        float b1 = (j == 0) ? 0 : int_img1.at<Vec3f>(j-1,i)[ch];
        float c1 = (i > 0 && j > 0) ? int_img1.at<Vec3f>(j-1,i-1)[ch] : 0;
        
        int_img1.at<Vec3f>(j,i)[ch] = abs(img1.at<Vec3b>(j,i)[ch] - img2.at<Vec3b>(j,i)[ch]) + a1 + b1 - c1;
      }
    }
  }
}

int main(int argc, char const *argv[])
{
  FileStorage fs2("/home/shamin/stereo-vision-master/disparity/map/build/mystereocalib.yml", FileStorage::READ);
  fs2["F"] >> F;
  img1 = imread("left3.jpg");
  img2 = imread("right3.jpg");
  namedWindow("LEFT", 1);
  namedWindow("RIGHT", 1);
  namedWindow("LEFT_MAP", 1);
  generateLeftDisparityMap();
  while (1) {
    imshow("LEFT", img1);
    imshow("RIGHT", img2);
    imshow("LEFT_MAP", disp_left);
    if (waitKey(0) > 0){
      imwrite("/home/shamin/stereo-vision-master/disparity/map/build/disp_left.png", disp_left);
      break;
    }
  }
  destroyWindow("LEFT");
  destroyWindow("RIGHT");
  destroyWindow("LEFT_MAP");
  return 0;
}