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
#include "config_parser.h"
#include "utils.h"

System::System(const std::string &config_yaml) {

  auto config_parser = ConfigParser(config_yaml);
  camera_model_ =
      std::make_shared<CameraModelPinholeEqui>(config_parser.camera_yaml_);

  if (config_parser.scale_camera_model_ != 1) {
    camera_model_->scale(config_parser.scale_camera_model_);
  }
  if (config_parser.transpose_image_) {
    camera_model_->transpose();
    transpose_image_ = config_parser.transpose_image_;
  }

  // 其他初始化
  cur_map_ = std::make_shared<Map>();
  locater = std::make_shared<Localization>(config_parser.vocabulary_,
                                           config_parser.pre_saved_images_,
                                           config_parser.threshold_,
                                           config_parser.transpose_image_,3);
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

double System::getPosition() {
  std::unique_lock<std::mutex> lock(mutex_position_);
  return position_;
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
      // 在这里之后进队cur_frame_对绝对为姿进行匹配用以后续计算offset,开始的两个关键阵仅取第2个关键帧用来计算
      {
      double true_position;
      auto status = locater->localize(cur_frame_,true_position,true);
      if( status ){
        cur_frame_->setFlag(true);
        cur_frame_->setAbsPosition(true_position);
        // 初始offset
        cur_map_->calculateOffset();
      }
      }

      std::cout << "[INFO]: The initialized map after g2o LinearMotion"
                << std::endl;
      cur_map_->debugPrintMap();
    } else {
      size_t frame_id = cur_frame_->getFrameId();

      bool track_status = cur_map_->trackNewFrameByKeyFrame(cur_frame_);

      if (track_status) {
        Eigen::Quaterniond q(cur_frame_->getEigenRotWc());
        Eigen::Vector3d twc = cur_frame_->getEigenTransWc();
        double scale = cur_map_->getScale();
        double position = twc[0] * scale;
        // 加上偏移量输出绝对位置信息
        double offset_temp = cur_map_->getOffset();
        std::cout << "[INFO]: Frame " << frame_id << ", " << toString(q) << ", " << std::endl
                  << "[INFO]: Frame relative position: " << -position << std::endl
                  << "[INFO]: Frame absolute position: " << -position + offset_temp << std::endl
                  << std::endl;
      } else {
        // 跟踪丢时候的重新建立地图点，相当于开始新的初始化，只是初始位姿是给定值
        Frame::Ptr last_kf = cur_map_->getLastKeyFrame();
        cur_map_->initialize(last_kf, cur_frame_);
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

        // 关键帧优化
        G2oOptimizer::Ptr opt = cur_map_->buildG2oOptKeyFrameBa();
        opt->optimizeLinearMotion();
        // 计算优化后的地图点的平均z值，计算尺度
        Eigen::Vector3d ave_kf_mp = opt->calAveMapPoint();
        cur_map_->ave_kf_mp_ = ave_kf_mp;
        ave_kf_mp[0] = 0.0;
        double scale = Map::kCraneHeight / ave_kf_mp.norm();
        cur_map_->setScale(scale);
        
        double true_position;

        auto status = locater->localize(cur_frame_,true_position,true);
        if( status ){
          cur_frame_->setFlag(true);
          cur_frame_->setAbsPosition(true_position);
          // 更新offset
          cur_map_->calculateOffset();
        }

        std::cout << "[INFO]: The size of keyframes are : " << cur_map_->getKeyframesSize() << std::endl;

        cur_map_->debugPrintMap();
        std::cout << std::endl;
      }
    }
  }
}
