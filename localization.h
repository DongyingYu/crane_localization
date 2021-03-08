/**
 * @file /**
 * @file localization.h
 * @author Dongying (yudong2817@sina.com)
 * @brief
 * @version 0.1
 * @date 2021-02-28
 *
 * @copyright Copyright (c) 2021
 */
#pragma once

#include "ORBVocabulary.h"
#include "frame.h"
#include <chrono>
#include <list>
#include <mutex>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>
#include <utility>

class Localization {

public:
  using Ptr = std::shared_ptr<Localization>;

  /**
   * @brief Construct a new Localization object
   *
   * @param[in] vocab_file 词表文件
   * @param[in] preload_keyframes 事先保存的“关键帧”
   * @param[in] win_size 对分值进行滑动平均的窗口大小
   */
  Localization(const std::string &vocab_file,
               const std::string &preload_keyframes,
               const bool &transpose_image=false, const int &win_size = 3);
  ~Localization();

  /**
   * @brief 对输入图像进行定位
   *
   * @param[in] image
   * @param[in] verbose
   * @return int 返回相似度最高的图片的索引
   */
  int localize(const cv::Mat &image, const bool &verbose = false);

private:
  std::vector<cv::Mat> loadImages(const std::string &index_filename);

  // todo: using shared_ptr
  ORBVocabulary *pVocabulary_;
  std::vector<Frame::Ptr> frames_;
  std::deque<std::vector<float>> winFrames_;

  int win_size_;
};
