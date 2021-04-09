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
  std::string video_file = "/home/ipsg/dataset_temp/74_new_test_cut.mp4";
  int skip_frames = 0;

  // skip some frames
  cv::Mat img;
  cv::VideoCapture capture(video_file);
  while (skip_frames > 0) {
    capture >> img;
    skip_frames--;
  }

  auto location = std::make_shared<Localization>(
      "./vocabulary/ORBvoc.txt", "./vocabulary/image_save3/rgb.txt", 0.01,
      true, 3);

  int cnt = 0;
  for (int cnt = 0;; ++cnt){
    capture >> img;
    if(img.rows == 0 || img.cols == 0) continue;
    // 输入图像的尺寸变换与先验图像尺寸不同不会影响匹配,但是transpose影响非常大，如果transpose不同，会带来问题
    cv::transpose(img,img);
    // cv::resize(img,img,{0,0},0.5,0.5);
    // 现场部署时需要用
    // if(cnt%3 != 0)
    //   continue;
    // cv::imshow("image_video", img);
    Frame::Ptr frame = std::make_shared<Frame>(img);
    double position;
    bool status = location->localize(frame, position, true);
    std::cout << "The frame cnt : " << cnt << std::endl;
    std::cout << "The crane position is : " << position << std::endl;
    cv::waitKey();
  }

  return 0;
}