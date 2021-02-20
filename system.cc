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

void System::run() {
    while (1) {
      if (!checkInput()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        continue;
      }
      {
        std::unique_lock<std::mutex> lock(input_mutex_);
        last_frame_ = cur_frame_;
        cur_frame_ = input_frames_.front();
        input_frames_.pop_front();
      }
      if (!last_frame_){
        // 只有一帧，啥事也不干
        continue;
      }
      if (!cur_map_){
        cv::Mat R, t;
        cur_map_ = initializer_->initialize(last_frame_, cur_frame_, intrinsic_.K());
        G2oOptimizer::mapBundleAdjustment(cur_map_, 10);
      } else {
        cur_map_->trackNewFrame(cur_frame_);

        if (cur_frame_->frame_id_ - last_frame_->frame_id_ >= 15) {
          cur_map_->frames_.emplace_back(cur_frame_);
        }

      }

    }
  }