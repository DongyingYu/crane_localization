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
#include "frame.h"
#include "initializer.h"
#include <chrono>
#include <list>
#include <mutex>
#include <thread>

class System {
public:

  using Ptr = std::shared_ptr<System>;

  System(const Intrinsic& intrinsic) : intrinsic_(intrinsic) {
    thread_ = std::thread(&System::run, this);
  }

  void run();

  inline bool checkInput() {
    std::unique_lock<std::mutex> lock(input_mutex_);
    return !input_frames_.empty();
  }

  inline void insertNewFrame(const Frame::Ptr& frame) {
    std::unique_lock<std::mutex> lock(input_mutex_);
    input_frames_.emplace_back(frame);
  }

public:
  // 数据输入
  std::mutex input_mutex_;
  std::list<Frame::Ptr> input_frames_;

  // 地图相关
  // 地图初始化
  Initializer::Ptr initializer_ = std::make_shared<Initializer>();
  // 当前地图
  Map::Ptr cur_map_ = nullptr;
  // 历史地图
  std::vector<Map::Ptr> maps_;
  
  // todo: keyframe
  Frame::Ptr cur_frame_ = nullptr;
  Frame::Ptr last_frame_ = nullptr;
  Frame::Ptr last_keyframe = nullptr;

private:
  Intrinsic intrinsic_;

  std::thread thread_;
};