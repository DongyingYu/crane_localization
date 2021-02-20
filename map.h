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

class Map
{

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
   * @brief 地图初始化后的第一次BA优化
   * @note 理论上天车相机位姿变换，应当无旋转，只有一个线性位移。
   *       为了①方便实现且可拓展性，②可能实际天车运行时，不一定符合理想情况；
   *       我们使用一般形式SE3顶点来表示相机位姿，而限制条件通过边的形式加入图中。
   */
  void initial_ba(const int & n_iterations=10);

public:
  std::mutex mutex_mappoints_;
  std::vector<MapPoint::Ptr> mappoints_;


  std::mutex mutex_frames_;
  std::vector<Frame::Ptr> frames_;
};
