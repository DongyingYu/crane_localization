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

int main(int argc, char **argv) {
  // 默认参数
  // std::string video_source =
  //     "rtsp://admin:wattman2020@192.168.1.146:554/Streaming/Channels/101?transportmode=unicast&profile=Profile_1";

  std::string video_file = "/home/ipsg/dataset_temp/78_cut.mp4";
  std::string yaml_file = "./conf/pipeline.yaml";
  int skip_frames = 1200;
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
  } else if(argc == 5){
    crane_id = atoi(argv[1]);
    video_file = argv[2];
    yaml_file = argv[3];
    skip_frames = atoi(argv[4]);
  }else {
    std::cout << "Usage: exec video_file yaml_file" << std::endl;
    std::cout << "       exec video_file yaml_file skip_frame" << std::endl;
  }

  std::cout << "[INFO]: crane_id = " << crane_id << std::endl;
  std::cout << "[INFO]: video_file = " << video_file << std::endl;
  std::cout << "[INFO]: yaml_file = " << yaml_file << std::endl;
  std::cout << "[INFO]: skip_frame = " << skip_frames << std::endl;

  auto system = std::make_shared<System>(yaml_file);

  // skip some frames
  cv::Mat img;
  cv::VideoCapture capture(video_file);

  // 捕获视频数据
  // cv::VideoCapture video_capture(video_source, CAP_FFMPEG);              //
  // Open input

  // if (!video_capture.isOpened())
  // {
  // 	std::cout << "Could not open the input video: " << video_source <<
  // std::endl; 	return -1;
  // }

  while (skip_frames-- > 0) {
    // video_capture >> img;
    capture >> img;
  }

  int cnt = 0;
  for (int cnt = 0;; ++cnt) {
    capture >> img;
    system->insertNewImage(img);

    cv::waitKey(5);
  }
}