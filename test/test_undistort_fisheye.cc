/**
 * @file test_undistort_fisheye.cc
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2021-02-23
 *
 * @copyright Copyright (c) 2021
 *
 */
#include "undistort.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int width = 2560;
int height = 1440;
double fx = 2052.01136163;
double fy = 2299.88726199;
double cx = 1308.23302753;
double cy = 713.03063759;
double k1 = -0.47040093;
double k2 = 1.0270395;
double k3 = -2.00061705;
double k4 = 1.64023946;

int main() {

  std::string video_dir =
      "/home/xt/Documents/data/DATASETS/ros_bag/BL-EX346HP-15M/";
  std::string video_file = video_dir + "192.168.1.146_01_20210205105528697.mp4";
  VideoCapture capture(video_file);

  UndistorterFisheye undistorter(width, height, fx, fy, cx, cy, k1, k2, k3, k4);

  int i = 0;
  while (1) {
    Mat frame;
    capture >> frame;
    if (frame.empty()) {
      break;
    }
    i++;
    // if (i < 1200) {
    //   continue;
    // }

    cv::Mat undistorted_frame;

    undistorter.undistort(frame, undistorted_frame);

    resize(frame, frame, {0, 0}, 0.5, 0.5);
    resize(undistorted_frame, undistorted_frame, {0, 0}, 0.5, 0.5);
    imshow("ori", frame);
    imshow("undistorted", undistorted_frame);
    waitKey(31);
  }
  return 0;
}