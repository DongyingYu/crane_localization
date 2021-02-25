/**
 * @file system.cc
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2021-02-19
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "system.h"
#include "optimizer.h"
#include <yaml-cpp/yaml.h>

System::System(const std::string &yaml_file, const bool &transpose_image,
               const double &scale_camera_model)
    : transpose_image_(transpose_image) {
  YAML::Node node = YAML::LoadFile(yaml_file);
  if (!node) {
    std::cout << "[ERROR]: Open yaml failed " << yaml_file << std::endl;
    exit(-1);
  }
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
      camera_model_ = std::make_shared<CameraModelPinholeEqui>(
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

  if (scale_camera_model != 1) {
    camera_model_->scale(scale_camera_model);
  }
  if (transpose_image_) {
    camera_model_->transpose();
  }

  thread_ = std::thread(&System::run, this);
}

bool System::isInputQueueEmpty() {
  std::unique_lock<std::mutex> lock(input_mutex_);
  return input_frames_.empty();
}

void System::insertNewImage(const cv::Mat &img) {
  cv::Mat image = img;
  if (transpose_image_) {
    cv::transpose(img, image);
  }
  Frame::Ptr frame = std::make_shared<Frame>(image, camera_model_);
  std::unique_lock<std::mutex> lock(input_mutex_);
  input_frames_.emplace_back(frame);
}

void System::run() {
  while (1) {
    if (isInputQueueEmpty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(30));
      continue;
    }
    {
      std::unique_lock<std::mutex> lock(input_mutex_);
      last_frame_ = cur_frame_;
      cur_frame_ = input_frames_.front();
      input_frames_.pop_front();
    }
    if (!last_frame_) {
      // 只有一帧，啥事也不干
      continue;
    }
    if (!cur_map_) {
      Map::Ptr cur_map = initializer_->initialize(last_frame_, cur_frame_);
      std::cout << "[INFO]: The map before g2o" << std::endl;
      cur_map->printMap();
      G2oOptimizer::mapBundleAdjustment(cur_map);
      std::cout << "[INFO]: The map after g2o" << std::endl;
      cur_map->printMap();
      {
        std::unique_lock<std::mutex> lock(map_mutex_);
        cur_map_ = cur_map;
      }
    } else {
      std::cout << "[INFO]: track new frame with cur_map: "
                << cur_frame_->frame_id_ << std::endl;
      cur_map_->trackNewFrame(cur_frame_);

      if (cur_frame_->frame_id_ - cur_map_->frames_.back()->frame_id_ >= 1) {
        cur_map_->frames_.emplace_back(cur_frame_);
        G2oOptimizer::mapBundleAdjustment(cur_map_);
        cur_map_->printMap();
      }
    }
  }
}