/***
 * @file:  loopclosing.h
 * @author: Dongying (yudong2817@sina.com)
 * @brief:
 * @version:  0.1
 * @date:  2021-04-20
 * @copyright: Copyright (c) 2021
 */

#pragma once

#include<chrono>
#include<list>
#include<mutex>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include<queue>
#include<thread>
#include<utility>
#include<vector>
#include<map
#include<memory>
#include<string>

class Loopclosing {
 public:
  using ptr = std::shared_ptr<Loopclosing>;

  Loopclosing();

  ~Loopclosing();

 private:
};