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
#include <yaml-cpp/yaml.h>
#include <chrono>
#include <opencv2/videoio.hpp>
#include "system.h"


int main(int argc, char **argv) {
  // 默认参数
  // std::string video_source =
  //     "rtsp://admin:wattman2020@192.168.1.146:554/Streaming/Channels/101?transportmode=unicast&profile=Profile_1";

  std::string video_file = "/home/ipsg/dataset_temp/78_cut.mp4";
  std::string yaml_file = "./conf/pipeline.yaml";
  int skip_frames = 1200;
  // 根据天车的id号，选择加载不同的先验位置信息
  int crane_id = 0;
  // 从命令行获取参数
  if (argc == 1) {
  } else if (argc == 3) {
    crane_id = atoi(argv[1]);
    video_file = argv[2];
  } else if (argc == 4) {
    crane_id = atoi(argv[1]);
    video_file = argv[2];
    yaml_file = argv[3];
  } else if (argc == 5) {
    crane_id = atoi(argv[1]);
    video_file = argv[2];
    yaml_file = argv[3];
    skip_frames = atoi(argv[4]);
  } else {
    std::cout << "Usage: exec video_file yaml_file" << std::endl;
    std::cout << "       exec video_file yaml_file skip_frame" << std::endl;
  }

  std::cout << "[INFO]: crane_id = " << crane_id << std::endl;
  std::cout << "[INFO]: video_file = " << video_file << std::endl;
  std::cout << "[INFO]: yaml_file = " << yaml_file << std::endl;
  std::cout << "[INFO]: skip_frame = " << skip_frames << std::endl;

  auto system = std::make_shared<System>(yaml_file, crane_id);

  // cv::VideoCapture capture(video_file);
  // if (!capture.isOpened()) {
  //   std::cout << "Could not open the input video: " << video_file <<
  //   std::endl;
  //   sleep(1000);
  //   cv::VideoCapture capture(video_file);
  //   std::cout << "Reconnect to the video: " << video_file << std::endl;
  //   return -1;
  // }

  bool capture_status = true;
  
  cv::VideoCapture capture(video_file,cv::CAP_GSTREAMER);
  {
    // capture.open(video_file);
    if (!capture.isOpened()) capture_status = false;
  }

  while (!capture_status) {
    std::cout << "[WARNING]: Could not open the input video: " << video_file
              << std::endl;
    capture.open(video_file);
    std::cout << "[INFO]: Reconnect to the video: " << video_file << std::endl;
    if (capture.isOpened()) capture_status = true;
    usleep(500000);
  }

  cv::Mat img;
  while (skip_frames-- > 0) {
    capture >> img;
  }

  int cnt = 0;
  for (int cnt = 0;; ++cnt) {
    capture.read(img);
    if (img.empty()) {
      std::cerr << "ERROR: blank frame. \n";

      {
        capture.open(video_file);
        if (!capture.isOpened()) capture_status = false;
      }
      while (!capture_status) {
        std::cout << "[WARNING]: Could not open the input video: " << video_file
                  << std::endl;
        capture.open(video_file);
        std::cout << "[INFO]: Reconnect to the video: " << video_file
                  << std::endl;
        if (capture.isOpened()) capture_status = true;
        usleep(500000);
      }
    }
    // 现场部署时需要用
    // if (cnt % 3 != 0) continue;
    system->insertNewImage(img);
    cv::waitKey();
  }
  system->stop();
}