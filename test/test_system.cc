/**
 * @file test_system.cc
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2021-02-19
 *
 * @copyright Copyright (c) 2021
 *
 */
#include "system.h"

double fx = 2428.05872198;
double fy = 1439.05448033;
double cx = 1439.05448033;
double cy = 846.58407292;
Intrinsic intrinsic = Intrinsic(fx, fy, cx, cy).scale(0.5);

int main(int argc, char **argv) {
  // config
  std::string video_file =
      "/home/xt/Documents/data/DATASETS/ros_bag/BL-EX346HP-15M/crane/78.mp4";
  int skip_frames = 1200;
  double scale_image = 0.6;
  bool transpose_image = true;

  // skip some frames
  cv::Mat img;
  cv::VideoCapture capture(video_file);
  while (skip_frames > 0) {
    capture >> img;
    skip_frames--;
  }

  if (transpose_image) {
    intrinsic = intrinsic.transpose();
  }
  intrinsic = intrinsic.scale(scale_image);

  auto system = std::make_shared<System>(intrinsic);

  // std::vector
  int cnt = 0;
  while (1) {
    cnt++;
    if (cnt % 15 != 0) {
      continue;
    }
    capture >> img;
    if (transpose_image) {
      cv::transpose(img, img);
    }
    cv::resize(img, img, {0, 0}, scale_image, scale_image);
    Frame::Ptr frame_cur = std::make_shared<Frame>(img, intrinsic);

    std::cout << "[INFO]: insert new frame " << cnt << std::endl;
    system->insertNewFrame(frame_cur);

    cv::waitKey();
  }
}