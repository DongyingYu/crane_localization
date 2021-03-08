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
#include "utils.h"

System::System(const std::string &yaml_file, const bool &transpose_image,
               const double &scale_camera_model)
    : transpose_image_(transpose_image) {

  camera_model_ = std::make_shared<CameraModelPinholeEqui>(yaml_file);

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

      G2oOptimizer::Ptr opt = cur_map_->buildG2oOptKeyFrameBa();
      opt->optimize();
      opt->optimizeLinearMotion();

      std::cout << "[INFO]: The initialized map after g2o LinearMotion"
                << std::endl;
      cur_map_->debugPrintMap();
    } else {
      size_t frame_id = cur_frame_->getFrameId();

      // 对每一个新来的图像帧与之前的图像帧执行优化
      bool track_status = cur_map_->trackNewFrameByKeyFrame(cur_frame_);

      if (track_status) {
        Eigen::Quaterniond q(cur_frame_->getEigenRotWc());
        Eigen::Vector3d twc = cur_frame_->getEigenTransWc();
        double scale = cur_map_->getScale();
        double position = twc[0] * scale;
        std::cout << "[INFO]: Frame " << frame_id << ", " << toString(q) << ", "
                  << position << std::endl
                  << std::endl;
      } else {
        // 跟踪丢失，在这里添加重定位部分
        std::cout << "[WARNING]: Frame " << frame_id << ", track frame failed "
                  << std::endl
                  << std::endl;
        continue;
      }

      // 判断是否插入关键帧
      if (cur_map_->checkIsNewKeyFrame(cur_frame_)) {
        std::cout << "[INFO]: Insert New KeyFrame " << frame_id << std::endl; 
        cur_map_->insertKeyFrame(cur_frame_);

        // 删除所有最近的帧，仅仅保留当前帧
        cur_map_->clearRecentFrames();
        cur_map_->insertRecentFrame(cur_frame_);

        // 关键帧优化，对关键帧执行优化
        G2oOptimizer::Ptr opt = cur_map_->buildG2oOptKeyFrameBa();
        opt->optimizeLinearMotion();

        // 计算优化后的地图点的平均z值，计算尺度
        Eigen::Vector3d ave_kf_mp = opt->calAveMapPoint();
        cur_map_->ave_kf_mp_ = ave_kf_mp;
        ave_kf_mp[0] = 0.0;
        double scale = Map::kCraneHeight / ave_kf_mp.norm();
        cur_map_->setScale(scale);


        cur_map_->debugPrintMap();
        std::cout << std::endl;
      }
    }
  }
}
