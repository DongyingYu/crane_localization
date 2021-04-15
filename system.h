/**
 * @file system.h
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2021-02-09
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once
#include <chrono>
#include <list>
#include <mutex>
#include <thread>
#include "camera_model.h"
#include "frame.h"
#include "localization.h"
#include "map.h"
#include "websocket_endpoint.h"

class System {
 public:
  using Ptr = std::shared_ptr<System>;

  /**
   * @brief Construct a new System object
   *
   * @param config_yaml yaml文件中的相机参数部分，来源与kalibr标定结果
   */
  System(const std::string &config_yaml, const int &crane_id);

  /**
   * @brief 插入新的一帧
   */
  void insertNewImage(const cv::Mat &img);

  void setPosition(const double &pos);

  double getPosition();

  void stop();

  void updatecoef(const std::vector<cv::Point2f> &points);

 private:
  /**
   * @brief 初始化地图，并追踪新的一帧
   */
  void run();

  bool isInputQueueEmpty();

 private:
  // 相机模型（包含畸变模型）
  CameraModel::Ptr camera_model_;
  // 内部处理时，是否将图像转置（相机模型中的参数也会跟着一起调整）
  bool transpose_image_ = false;
  // 内部处理时，是否将图片缩放（缩小图片，为了减少计算量，注意需与scale_camera_model相对应）
  double scale_image_ = 1.0;

  // 数据输入
  std::mutex mutex_input_;
  std::list<cv::Mat> input_images_;

  // 临时变量
  Frame::Ptr cur_frame_ = nullptr;
  Frame::Ptr last_frame_ = nullptr;

  // 当前地图
  std::mutex mutex_map;
  Map::Ptr cur_map_ = nullptr;

  // 所有历史地图，用于地图合并，全局建图
  std::vector<Map::Ptr> history_maps_;

  // 绝对位置定位
  Localization::Ptr locater_;

  // 当前的位置，不一定是实时传入的图片对应的相机位置，可能有延迟。
  std::mutex mutex_position_;
  double position_ = 0;

  // slam线程
  std::thread thread_;

  // websocket

  MySocket::websocket_endpoint ws_endpoint_;

  // Points data for line fitting
  // std::vector<cv::Point2f> points_data_= std::vector<cv::Point2f>(10,cv::Point2f(0.0,0.0));
  std::vector<cv::Point2f> points_data_;

  // debug
  double debug_draw_;

  int crane_id_;

  bool pop_frame_;
  bool save_position_;

  // 对应于4辆天车的初始尺度，运行中不断更新
  double offset_ = 0.0;
  double k1_ = 12.53;
  double k2_ = 12.53;
  double k3_ = 6.75;
  double k4_ = 13.18;
};