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
#include "camera_model.h"
#include "frame.h"
#include "initializer.h"
#include <chrono>
#include <list>
#include <mutex>
#include <thread>

class System {
public:
  using Ptr = std::shared_ptr<System>;

  /**
   * @brief Construct a new System object
   *
   * @param yaml_file yaml文件中的相机参数部分，来源与kalibr标定结果
   * @param transpose_image
   * 内部处理时，是否将图像转置（相机模型中的参数也会跟着一起调整）
   * @param scale_camera_model
   * 将相机模型中的参数，缩放scale_camera_model倍，才与实际输入的图像相符
   */
  System(const std::string &yaml_file, const bool &transpose_image,
         const double &scale_camera_model = 1.0);

  /**
   * @brief 初始化地图，并追踪新的一帧
   */
  void run();

  bool isInputQueueEmpty();

  /**
   * @brief 插入新的一帧
   */
  void insertNewImage(const cv::Mat &img);

public:
  // 数据输入
  std::mutex input_mutex_;
  std::list<Frame::Ptr> input_frames_;

  // 地图相关
  // 地图初始化
  Initializer::Ptr initializer_ = std::make_shared<Initializer>();
  // 当前地图
  std::mutex map_mutex_;
  Map::Ptr cur_map_ = nullptr;
  // 历史地图
  std::vector<Map::Ptr> maps_;

  // todo: keyframe
  Frame::Ptr cur_frame_ = nullptr;
  Frame::Ptr last_frame_ = nullptr;
  Frame::Ptr last_keyframe = nullptr;

private:
  CameraModel::Ptr camera_model_;

  std::thread thread_;

  bool transpose_image_ = false;
};