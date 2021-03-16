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
#include <yaml-cpp/yaml.h>
#include <chrono>

int main(int argc, char **argv) {
  // 默认参数
  // std::string video_source =
  //     "rtsp://admin:wattman2020@192.168.1.146:554/Streaming/Channels/101?transportmode=unicast&profile=Profile_1";

  std::string video_file = "/home/ipsg/dataset_temp/78_cut.mp4";
  std::string yaml_file = "./conf/pipeline.yaml";
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

  auto system = std::make_shared<System>(yaml_file);

  // cv::VideoCapture capture(video_file, CAP_FFMPEG);
  cv::VideoCapture capture(video_file);

  if (!capture.isOpened()) {
  	std::cout << "Could not open the input video: " << video_file 
              << std::endl; 	
    return -1;
  }

  cv::Mat img;
  while (skip_frames-- > 0) {
    capture >> img;
  }

  int cnt = 0;
  for (int cnt = 0;; ++cnt) {
    capture >> img;
    system->insertNewImage(img);
    cv::waitKey(5);
  }
}