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
#include <Eigen/Core>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <sstream>

void statistic(const std::vector<double> &data, const std::string &prefix_str) {
  int cnt = 0;
  double ave = 0, min_v = 0, max_v = 0, sum = 0;

  if (!data.empty()) {
    cnt = data.size();
    sum = std::accumulate(data.begin(), data.end(), 0.0);
    auto mm = std::minmax_element(data.begin(), data.end());
    ave = sum / data.size();
    min_v = *mm.first;
    max_v = *mm.second;
  }

  std::cout << prefix_str << " cnt: " << cnt << " min: " << min_v
            << " max: " << max_v << " ave: " << ave << " sum: " << sum
            << std::endl;
}

void calAveStddev(const std::vector<double> &data, double &ave,
                  double &stddev) {
  if (data.empty()) {
    return;
  }
  double sum = std::accumulate(data.begin(), data.end(), 0.0);
  ave = sum / data.size();

  // std::cout << "  data.size()=" << data.size() << " ave=" << ave <<
  // std::endl;

  double accum = 0;
  for (const double &d : data) {
    accum += (d - ave) * (d - ave);
  }
  if (data.size() - 1 > 0) {
    stddev = std::sqrt(accum / (data.size() - 1));
  }
}

void calAveStddev(const std::vector<cv::Point2f> &data, cv::Point2f &ave,
                  cv::Point2f &stddev, const bool &abs) {
  if (data.empty()) {
    return;
  }
  std::vector<cv::Point2f> data_;
  for (const auto &d : data) {
    cv::Point2f p = d;
    if (abs) {
      p = cv::Point2f(std::abs(d.x), std::abs(d.y));
    }
    data_.emplace_back(p);
  }

  cv::Point2f sum =
      std::accumulate(data_.begin(), data_.end(), cv::Point2f(0, 0));
  ave = cv::Point2f(sum.x / data_.size(), sum.y / data_.size());

  cv::Point2f accum = cv::Point2f(0, 0);
  for (const cv::Point2f &p : data_) {
    cv::Point2f d = p - ave;
    accum += cv::Point2f(d.x * d.x, d.y * d.y);
  }
  if (data_.size() - 1 > 0) {
    double stddev_x = std::sqrt(accum.x / (data_.size() - 1));
    double stddev_y = std::sqrt(accum.y / (data_.size() - 1));
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