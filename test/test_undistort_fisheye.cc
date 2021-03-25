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
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include "camera_model.h"
#include "frame.h"
using namespace std;
using namespace cv;

int main(int argc, char **argv) {
  // 默认参数
  std::string video_file =
      "/home/xt/Documents/data/DATASETS/ros_bag/BL-EX346HP-15M/"
      "192.168.1.146_01_20210205105528697.mp4";
  // std::string video_file =
  // "/home/xt/Documents/data/DATASETS/ros_bag/BL-EX346HP-15M/crane/78.mp4";
  std::string yaml_file =
      "/home/xt/Documents/data/3D-Mapping/3D-Reconstruction/case-base/"
      "crane_localization/conf/BL-EX346HP-15M.yaml";
  int skip_frames = 1200;
  double scale_image = 1.0;
  // 从命令行获取参数
  if (argc == 1) {
  } else if (argc == 3) {
    video_file = argv[1];
    yaml_file = argv[2];
  } else if (argc == 4) {
    video_file = argv[1];
    yaml_file = argv[2];
    skip_frames = atoi(argv[3]);
  } else if (argc == 5) {
    video_file = argv[1];
    yaml_file = argv[2];
    skip_frames = atoi(argv[3]);
    scale_image = atof(argv[4]);
  } else {
    std::cout << "Usage: exec video_file yaml_file" << std::endl;
    std::cout << "       exec video_file yaml_file skip_frame" << std::endl;
    std::cout << "       exec video_file yaml_file skip_frame scale_image"
              << std::endl;
    return 0;
  }
  std::cout << "[INFO]: video_file = " << video_file << std::endl;
  std::cout << "[INFO]: yaml_file = " << yaml_file << std::endl;
  std::cout << "[INFO]: skip_frame = " << skip_frames << std::endl;
  std::cout << "[INFO]: scale_image = " << scale_image << std::endl;

  CameraModel::Ptr camera_model =
      std::make_shared<CameraModelPinholeEqui>(yaml_file);

  if (std::abs(scale_image - 1.0) > std::numeric_limits<float>::epsilon()) {
    camera_model->scale(scale_image);
  }

  VideoCapture capture(video_file);

  int i = 0;
  while (1) {
    Mat img;
    capture >> img;
    if (img.empty()) {
      std::cout << "[ERROR]: No image from video capture" << std::endl;
      break;
    }
    i++;

    std::cout << "image size 1 " << img.size() << std::endl;
    if (std::abs(scale_image - 1.0) > std::numeric_limits<float>::epsilon()) {
      cv::resize(img, img, {0, 0}, scale_image, scale_image);
    }
    std::cout << "image size 2 " << img.size() << std::endl;

    cv::Mat un_img;
    Frame::Ptr frame = std::make_shared<Frame>(img, camera_model);

    double draw_scale = 0.5;
    frame->debugDraw(draw_scale);

    camera_model->undistort(img, un_img);
    // camera_model->undistortImage(img, un_img);

    cv::Mat img_kp, un_img_kp;
    cv::drawKeypoints(un_img, frame->getUnKeyPoints(), un_img_kp,
                      cv::Scalar::all(-1),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    resize(un_img_kp, un_img_kp, {0, 0}, draw_scale, draw_scale);
    imshow("undistorted", un_img_kp);
    if (waitKey(30) == 27) break;
    waitKey();
  }
  return 0;
}