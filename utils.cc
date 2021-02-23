/**
 * @file utils.cc
 * @author xiaotaw (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2021-02-22
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#include "utils.h"
#include <numeric>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <Eigen/Core>

void statistic(const std::vector<double> &data,
                      const std::string &name) {
  int cnt = 0;
  double ave = 0, min_v = 0, max_v = 0;

  if (!data.empty()) {
    cnt = data.size();
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    auto mm = std::minmax_element(data.begin(), data.end());
    ave = sum / data.size();
    min_v = *mm.first;
    max_v = *mm.second;
  }

  std::cout << "[INFO]: " << name << " cnt: " << cnt << " min: " << min_v
            << " max: " << max_v << " ave: " << ave << std::endl;
}


void calAveStddev(const std::vector<double> &data, double &ave, double &stddev) {
  if (data.empty()) {
    return;
  }
  double sum = std::accumulate(data.begin(), data.end(), 0.0);
  ave = sum / data.size();

  // std::cout << "  data.size()=" << data.size() << " ave=" << ave << std::endl; 

  double accum = 0;
  for(const double& d : data) {
    accum += (d - ave) * (d - ave);
  }
  if (data.size() - 1 > 0) {
    stddev = std::sqrt(accum / (data.size() - 1));
  }
}

void calAveStddev(const std::vector<cv::Point2f> &data, cv::Point2f &ave, cv::Point2f &stddev) {
  if (data.empty()) {
    return;
  }
  cv::Point2f sum = std::accumulate(data.begin(), data.end(), cv::Point2f(0, 0));
  ave = cv::Point2f(sum.x / data.size(), sum.y / data.size());

  cv::Point2f accum = cv::Point2f(0, 0);
  for (const cv::Point2f &p : data) {
    cv::Point2f d = p - ave;
    accum += cv::Point2f(d.x * d.x, d.y * d.y);
  }
  if (data.size() - 1 > 0) {
    double stddev_x = std::sqrt(accum.x / (data.size() - 1));
    double stddev_y = std::sqrt(accum.y / (data.size() - 1));
    stddev = cv::Point2f(stddev_x, stddev_y);
  }
}


std::string toString(const Eigen::Quaterniond &q) {
  std::stringstream ss;
  ss << q.w() << " " << q.x() << " " << q.y() << " " << q.z();
  return ss.str();
}


std::string toString(const Eigen::Vector3d &v) {
  std::stringstream ss;
  ss << v.x() << " " << v.y() << " " << v.z(); 
  return ss.str();
}