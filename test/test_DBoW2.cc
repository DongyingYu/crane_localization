/**
 * @file test_DBoW2.cc
 * @author Dongying (yudong2817@sina.com)
 * @brief
 * @version 0.1
 * @date 2021-02-24
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "frame.h"
#include "localization.h"
#include "websocket_endpoint.h"

int main(int argc, char **argv) {
  std::string video_file = "/home/ipsg/dataset_temp/78_test_o.mp4";
  int skip_frames = 0;

  // skip some frames
  cv::Mat img;
  cv::VideoCapture capture(video_file);
  while (skip_frames > 0) {
    capture >> img;
    skip_frames--;
  }

  auto location = std::make_shared<Localization>(
      "./vocabulary/ORBvoc.txt", "./vocabulary/image_save2/rgb.txt", 0.01,
      false, 3);

  int cnt = 0;
  while (1) {
    cnt++;
    capture >> img;
    // 现场部署时需要用
    // if(cnt%3 != 0)
    //   continue;
    // cv::imshow("image_video", img);
    Frame::Ptr frame = std::make_shared<Frame>(img);
    double position;
    bool status = location->localize(frame, position, true);
    std::cout << "The frame cnt : " << cnt << std::endl;
    std::cout << "The crane position is : " << position << std::endl;
    cv::waitKey(50);
  }

  return 0;
}
