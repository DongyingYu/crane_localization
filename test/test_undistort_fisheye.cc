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
#include "camera_model.h"
#include "frame.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>
#include <yaml-cpp/yaml.h>
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main(int argc, char** argv) {
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
  // 从命令行获取参数
  if (argc == 1) {
  } else if (argc == 3) {
    video_file = argv[1];
    yaml_file = argv[2];
  } else if (argc == 4) {
    video_file = argv[1];
    yaml_file = argv[2];
    skip_frames = atoi(argv[3]);
  } else {
    std::cout << "Usage: exec video_file yaml_file" << std::endl;
    std::cout << "       exec video_file yaml_file skip_frame" << std::endl;
  }
  std::cout << "[INFO]: video_file = " << video_file << std::endl;
  std::cout << "[INFO]: yaml_file = " << yaml_file << std::endl;
  std::cout << "[INFO]: skip_frame = " << skip_frames << std::endl;

  // 读取相机内参，创建相机参数模型
  YAML::Node node = YAML::LoadFile(yaml_file);
  if (!node) {
    std::cout << "[ERROR]: Open yaml failed " << yaml_file << std::endl;
    exit(-1);
  }
  CameraModel::Ptr camera_model;
  // 相机模型和畸变模型
  if (node["cam0"]) {
    YAML::Node cam = node["cam0"];
    std::string camera = cam["camera_model"].as<std::string>();
    std::string distortion = cam["distortion_model"].as<std::string>();
    auto intrinsics = cam["intrinsics"].as<std::vector<double>>();
    auto dist_coef = cam["distortion_coeffs"].as<std::vector<double>>();
    auto resolution = cam["resolution"].as<std::vector<int>>();
    cv::Size img_size = cv::Size(resolution[0], resolution[1]);
    if (camera == "pinhole" && distortion == "equidistant") {
      camera_model = std::make_shared<CameraModelPinholeEqui>(
          intrinsics, img_size, dist_coef);
    } else {
      std::cout << "[ERROR]: unsupported Camera model" << std::endl;
      exit(-1);
    }
  } else {
    std::cout << "[ERROR]: failed to read camera param from " << yaml_file
              << std::endl;
    exit(-1);
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

    cv::Mat un_img, un_img_2;


    // cv::fisheye::undistortImage(img, un_img_2, undistorter.getK(),
    //                             undistorter.getD(), undistorter.getNewK(),
    //                             img.size());

    Frame::Ptr frame = std::make_shared<Frame>(img, camera_model);

    double draw_scale = 0.3;
    frame->debugDraw(draw_scale);

    camera_model->undistort(img, un_img);

    cv::Mat img_kp, un_img_kp, un_img_kp_2;
    cv::drawKeypoints(un_img, frame->getUnKeyPoints(), un_img_kp,
                      cv::Scalar::all(-1),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    // cv::drawKeypoints(un_img_2, frame->getUnKeyPoints(), un_img_kp_2,
    //                   cv::Scalar::all(-1),
    //                   cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // resize(img_kp, img_kp, {0, 0}, draw_scale, draw_scale);
    resize(un_img_kp, un_img_kp, {0, 0}, draw_scale, draw_scale);
    // resize(un_img_kp_2, un_img_kp_2, {0, 0}, draw_scale, draw_scale);
    // imshow("ori", img_kp);
    imshow("undistorted", un_img_kp);
    // // imshow("undistorted2", un_img_kp_2);
    waitKey();
  }
  return 0;
}