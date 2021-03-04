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

  // 其他初始化
  cur_map_ = std::make_shared<Map>();
  thread_ = std::thread(&System::run, this);
}

bool System::isInputQueueEmpty() {
  std::unique_lock<std::mutex> lock(mutex_input_);
  return input_frames_.empty();
}

void System::insertNewImage(const cv::Mat &img) {
  cv::Mat image = img;
  if (transpose_image_) {
    cv::transpose(img, image);
  }
  Frame::Ptr frame = std::make_shared<Frame>(image, camera_model_);
  std::unique_lock<std::mutex> lock(mutex_input_);
  input_frames_.emplace_back(frame);
}

void System::run() {
  while (1) {
    if (isInputQueueEmpty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(30));
      continue;
    }
    {
      std::unique_lock<std::mutex> lock(mutex_input_);
      last_frame_ = cur_frame_;
      cur_frame_ = input_frames_.front();
      input_frames_.pop_front();
    }
    if (!last_frame_) {
      // 只有一帧，啥事也不干
      continue;
    }
    if (!cur_map_->checkInitialized()) {
      cur_map_->initialize(last_frame_, cur_frame_);
      std::cout << "[INFO]: The map before g2o" << std::endl;
      cur_map_->debugPrintMap();
      G2oOptimizer::mapBundleAdjustment(cur_map_, 10);
      std::cout << "[INFO]: The map after g2o" << std::endl;
      cur_map_->debugPrintMap();
      G2oOptimizerForLinearMotion::mapBundleAdjustment(cur_map_);
      std::cout << "[INFO]: The map after g2o LinearMotion" << std::endl;
      cur_map_->debugPrintMap();
    } else {
      std::cout << "[INFO]: track new frame with cur_map: "
                << cur_frame_->getFrameId() << std::endl;
      cur_map_->trackNewFrameByKeyFrame(cur_frame_);

      std::cout << "[INFO]: The map after g2o LinearMotion" << std::endl;
      cur_map_->debugPrintMap();

      // 判断是否插入关键帧
      if (cur_map_->checkIsNewKeyFrame(cur_frame_)) {
        cur_map_->insertKeyFrame(cur_frame_);
        std::map<size_t, std::pair<Frame::Ptr, bool>> frames_data;
        std::map<size_t, std::pair<MapPoint::Ptr, bool>> mps_data;
        std::map<size_t, std::vector<std::pair<size_t, size_t>>>
            observations_data;
        cur_map_->requestG2oInputKeyFrameBa(frames_data, mps_data,
                                            observations_data);
        G2oOptimizerForLinearMotion::optimize(frames_data, mps_data,
                                              observations_data, 50);
        // 计算优化后的地图点的平均z值，计算尺度
        Eigen::Vector3d ave_kf_mp = Eigen::Vector3d::Zero();
        for(auto &it : mps_data) {
          auto &mp = it.second.first;
          ave_kf_mp += mp->toEigenVector3d();
        }
        ave_kf_mp /= mps_data.size();
        cur_map_->ave_kf_mp_ = ave_kf_mp;
        ave_kf_mp[0] = 0.0;
        double scale = Map::kCraneHeight / ave_kf_mp.norm();
        cur_map_->setScale(scale);
      }
    }
  }
}
