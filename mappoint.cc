/*
 * @file:  
 * @author: Dongying (yudong2817@sina.com)
 * @brief:  
 * @version:  
 * @date:  Do not edit 
 * @copyright: Copyright (c) 2021
 */
/**
 * @file mappoint.cc
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2021-02-22
 *
 * @copyright Copyright (c) 2021
 *
 */
#include "mappoint.h"

MapPoint::MapPoint(float x, float y, float z) : x_(x), y_(y), z_(z) {
  mp_id_ = total_mp_cnt_++;
}

Eigen::Vector3d MapPoint::toEigenVector3d() {
  Eigen::Vector3d ret;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    ret << x_, y_, z_;
  }
  return ret;
}

void MapPoint::fromEigenVector3d(const Eigen::Vector3d &vec) {
  std::unique_lock<std::mutex> lock(mutex_);
  x_ = vec[0];
  y_ = vec[1];
  z_ = vec[2];
}

void MapPoint::rotate(const Eigen::Quaterniond &q) {
  Eigen::Vector3d p = toEigenVector3d();
  p = q * p;
  fromEigenVector3d(p);
}

size_t MapPoint::getId() const { return mp_id_; }

std::vector<std::pair<size_t, size_t>> MapPoint::getObservation() {
  std::unique_lock<std::mutex> lock(mutex_observation_);
  return observations_;
}

void MapPoint::eraseObservation(const std::map<size_t, size_t> &e_index) {
  std::unique_lock<std::mutex> lock(mutex_observation_);
  // std::map<size_t, size_t>::iterator find_it;
  // find_it = e_index.find(it.first);
  // auto flag = (find_it == e_index.end()) ? 0 : 1;
  // if (flag) {
  for (auto &it : e_index) {
    std::vector<std::pair<size_t, size_t>>::iterator iter;
    for (iter = observations_.begin(); iter != observations_.end();) {
      if ((*iter).first == it.second) {
        // 返回指向被删元素下一个位置元素的迭代器
        iter = observations_.erase(iter);
      } else {
        ++iter;
      }
    }
  }
}

int MapPoint::getObservationSize() {
  std::unique_lock<std::mutex> lock(mutex_observation_);
  return observations_.size();
}

int MapPoint::getPointValue(const int &num) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (num == 1) {
    return x_;
  } else if (num == 2) {
    return y_;
  } else {
    return z_;
  }
}

size_t MapPoint::total_mp_cnt_ = 0;
