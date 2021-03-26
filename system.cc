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

System::System(const std::string &config_yaml, const int &crane_id)
    : crane_id_(crane_id) {
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
  debug_draw_ = config_parser.debug_draw_;
  scale_image_ = config_parser.scale_image_;

  std::string pre_load_images;
  if (crane_id == 1) {
    pre_load_images = config_parser.pre_saved_images_one_;
    std::cout << "[INFO]: \033[33m Crane data one loading test ... \033[0m "
              << std::endl;
  } else if (crane_id == 2) {
    std::cout << "[INFO]: \033[33m Crane data two loading test ... \033[0m "
              << std::endl;
    pre_load_images = config_parser.pre_saved_images_two_;
  } else if (crane_id == 3) {
    std::cout << "[INFO]: \033[33m Crane data thr loading test ... \033[0m "
              << std::endl;
    pre_load_images = config_parser.pre_saved_images_thr_;
  } else {
    std::cout << "[INFO]: \033[33m Crane data fou loading test ... \033[0m "
              << std::endl;
    pre_load_images = config_parser.pre_saved_images_fou_;
  }
  // 其他初始化
  cur_map_ = std::make_shared<Map>(config_parser.sliding_window_size_local_,
                                   config_parser.sliding_window_size_global_);
  locater = std::make_shared<Localization>(
      config_parser.vocabulary_, pre_load_images, config_parser.threshold_,
      config_parser.transpose_image_, 3);
  thread_ = std::thread(&System::run, this);

  // 获取服务器地址
  std::string server_address = config_parser.server_address_;
  ws_endpoint_.connect(server_address);

  // bool done = false;
  // while (!done) {
  //   // 返回视频的http地址
  //   bool link_status = ws_endpoint_.parsing();
  //   if (!link_status)
  //     continue;
  //   else {
  //     // 添加文件数据信息写入txt的操作，存储顺序：id rtsp流地址
  //     done = true;
  //   }
  // }
  // ws_endpoint_.connect("ws://192.168.1.106:18001/ws?client_type=edge1&id=1");
}

bool System::isInputQueueEmpty() {
  std::unique_lock<std::mutex> lock(mutex_input_);
  return input_images_.empty();
}

void System::insertNewImage(const cv::Mat &img) {
  cv::Mat image = img.clone();
  if (image.rows <= 0 || image.cols <= 0) return;
  if (transpose_image_) {
    cv::transpose(img, image);
  }

  if (std::abs(scale_image_ - 1.0) > std::numeric_limits<float>::epsilon()) {
    cv::resize(image, image, {0, 0}, scale_image_, scale_image_);
  }
  int rows = image.rows;
  int cols = image.cols;
  if (rows > 0 && cols > 0 && rows == camera_model_->getImageSize().height &&
      cols == camera_model_->getImageSize().width) {
    std::unique_lock<std::mutex> lock(mutex_input_);
    while (input_images_.size() > 10) {
      std::cout << "[WARNING]: Drop an image because slam system's low fps"
                << std::endl;
      input_images_.pop_front();
    }
    std::cout << "\033[33m image save list's size: \033[0m " << input_images_.size()  << std::endl;
    input_images_.emplace_back(image);
  } else {
    std::cout << "[WARNING]: invalid image size (width=" << cols
              << " height=" << rows << std::endl;
  }
}

void System::setPosition(const double &pos) {
  std::unique_lock<std::mutex> lock(mutex_position_);
  position_ = pos;
}

double System::getPosition() {
  std::unique_lock<std::mutex> lock(mutex_position_);
  return position_;
}

void System::stop() { thread_.join(); }

// clang-format off
void System::run() {
    // 图像帧跟踪失败 标记标记信息
  int cnt_failed;
  int id_failed;
  while (1) {
    if (isInputQueueEmpty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(30));
      continue;
    }
    {
      cv::Mat image;
      {
        std::unique_lock<std::mutex> lock(mutex_input_);
        image = input_images_.front();
        input_images_.pop_front();
      }
      Frame::Ptr frame = std::make_shared<Frame>(image, camera_model_);
      // 特征点数等于零会导致程序退出。
      if (frame->getUnKeyPointsSize() > 0) {
        last_frame_ = cur_frame_;
        cur_frame_ = frame;
      }
    }
    if (!last_frame_) {
      // 只有一帧，啥事也不干
      continue;
    }
    if (!cur_map_->checkInitialized()) {
      bool init_status = cur_map_->initialize(last_frame_, cur_frame_, debug_draw_);
      if(!init_status) {
        continue;
      }
      int cnt_failed=1;
      int id_failed=0;

      G2oOptimizer::Ptr opt = cur_map_->buildG2oOptKeyFrameBa();
      opt->optimize();
      opt->optimizeLinearMotion();
      // 在这里之后进队cur_frame_对绝对为姿进行匹配用以后续计算offset,开始的两个关键阵仅取第2个关键帧用来计算
      {
        double true_position;
        auto status = locater->localize(cur_frame_, true_position, false);
        if(status) {
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

      int track_status = cur_map_->trackNewFrameByKeyFrame(cur_frame_, debug_draw_);

      cur_map_->debugPrintMap();

      if(track_status == 3) {
        std::cout << "\033[33m -------------------------Normal tracking-------------------------\033[0m" << std::endl;
        Eigen::Quaterniond q(cur_frame_->getEigenRotWc());
        Eigen::Vector3d twc = cur_frame_->getEigenTransWc();
        double scale = cur_map_->getScale();
        std::cout << "[INFO]: test the value of scale:   " << scale << std::endl;
        double position = twc[0] * scale;
        // 加上偏移量输出绝对位置信息
        double offset_temp = cur_map_->getOffset();
        std::cout << "[INFO]: Frame " << frame_id << ", " << toString(q) << ", " << std::endl
                  << "[INFO]: Frame relative position: " << -position << std::endl
                  << "[INFO]: Frame absolute position: " << -position + offset_temp << std::endl
                  << std::endl;
        position_ = -position + offset_temp;
        // 需修改，天车ID从外部传入
        ws_endpoint_.send(position_, crane_id_);
      } else if(track_status == 2){
        // 跟踪丢时候的重新建立地图点，相当于开始新的初始化，只是初始位姿是给定值(备用)
        std::cout << "[WARNING]: Frame " << frame_id << std::endl
                  << "\033[31m ----------------------------------------------track frame failed with too few 3d point----------------------------------------------- \033[0m" << std::endl
                  << "\033[31m ----------------------------------------------track frame failed with too few 3d point-----------------------------------------------\033[0m" << std::endl
                  << "\033[31m ----------------------------------------------track frame failed with too few 3d point----------------------------------------------- \033[0m" << std::endl;
        Frame::Ptr last_kf = cur_map_->getLastKeyFrame();

        std::cout << "..........................Try to reinitialize in.........................." << std::endl;
        cur_map_->initialize(last_kf, cur_frame_);
        std::cout << "..........................Try to reinitialize out.........................." << std::endl;
        continue;
      } else {
        //若连续10帧与之前关键帧都跟踪不上，则重新初始化，is_initialized_置为false
        std::cout << "\033[33m -------------------------track frame failed with too few matches-------------------------\033[0m" << std::endl;
        if(cur_frame_->getFrameId() - id_failed == 1)
          cnt_failed++;
        else
          cnt_failed = 1;
        id_failed = cur_frame_->getFrameId();

        if(cnt_failed > 10){
          cur_map_->setInitializeStatus(false);
          //history_maps_.emplace_back(cur_map_);
          cur_map_ = std::make_shared<Map>();
          std::cout << "\033[33m -------------------------system fails for 10 consecutive frames, reinitialize-------------------------\033[0m" << std::endl;
        }
        continue;
      }
      

      // 判断是否插入关键帧
      if (cur_map_->checkIsNewKeyFrame(cur_frame_)) {
        std::cout << "\033[33;44m [INFO]: Insert New KeyFrame \033[0m " << frame_id << std::endl;
        cur_map_->releaseLastKeyframeimg();
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

        auto status = locater->localize(cur_frame_,true_position,false);
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
// clang-format on
