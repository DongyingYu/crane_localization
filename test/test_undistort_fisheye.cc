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
#include "frame.h"
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

Intrinsic intrinsic(fx, fy, cx, cy);

int main() {

  std::string video_dir =
      "/home/xt/Documents/data/DATASETS/ros_bag/BL-EX346HP-15M/";
  std::string video_file = video_dir + "192.168.1.146_01_20210205105528697.mp4";
  VideoCapture capture(video_file);

  UndistorterFisheye undistorter(width, height, fx, fy, cx, cy, k1, k2, k3, k4);

  int i = 0;
  while (1) {
    Mat img;
    capture >> img;
    if (img.empty()) {
      break;
    }
    i++;
    // if (i < 1200) {
    //   continue;
    // }

    cv::Mat undistorted_img, undistorted_img_2;

    undistorter.undistort(img, undistorted_img);

    cv::fisheye::undistortImage(img, undistorted_img_2, undistorter.getK(),
                                undistorter.getD(), undistorter.getNewK(),
                                img.size());

    Frame::Ptr frame = std::make_shared<Frame>(img, intrinsic);

    for (auto &kp : frame->keypoints_) {
      // kp.size *= 10;
    }
    undistorter.undistortPoint(frame->keypoints_, frame->un_keypoints_);

    cv::Mat img_kp, un_img_kp, un_img_kp_2;
    cv::drawKeypoints(frame->img_, frame->keypoints_, img_kp,
                      cv::Scalar::all(-1),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::drawKeypoints(undistorted_img, frame->un_keypoints_, un_img_kp,
                      cv::Scalar::all(-1),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::drawKeypoints(undistorted_img_2, frame->un_keypoints_, un_img_kp_2,
                      cv::Scalar::all(-1),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    double draw_scale = 0.3;
    resize(img_kp, img_kp, {0, 0}, draw_scale, draw_scale);
    resize(un_img_kp, un_img_kp, {0, 0}, draw_scale, draw_scale);
    resize(un_img_kp_2, un_img_kp_2, {0, 0}, draw_scale, draw_scale);
    imshow("ori", img_kp);
    imshow("undistorted", un_img_kp);
    imshow("undistorted2", un_img_kp_2);
    waitKey();
  }
  return 0;
}