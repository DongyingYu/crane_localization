/**
 * @file map.h
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2021-02-09
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once
#include "frame.h"
#include "mappoint.h"
#include <mutex>
#include <vector>

class Map {

public:
  using Ptr = std::shared_ptr<Map>;

  Map() {}

  /**
   * @brief 单目初始化
   *
   * @todo 将现有的initializer整合进此函数中
   */
  bool initializeMono();

  bool trackNewFrame(Frame::Ptr frame);

  /**
   * @brief 旋转相机的pose，使得每一帧的旋转都相同，且平移量在X轴
   */
  void rotateFrameToXTranslation();

  // debug
  void printMap() const;

public:
  std::mutex mutex_mappoints_;
  std::vector<MapPoint::Ptr> mappoints_;

  std::mutex mutex_frames_;
  std::vector<Frame::Ptr> frames_;
};
