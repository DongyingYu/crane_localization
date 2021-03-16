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

int MapPoint::getObservationSize() {
  std::unique_lock<std::mutex> lock(mutex_observation_);
  return observations_.size();
}

size_t MapPoint::total_mp_cnt_ = 0;
