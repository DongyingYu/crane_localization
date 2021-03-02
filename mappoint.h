/**
 * @file mappoint.h
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2021-02-09
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <memory> // for shared_ptr etc.
#include <mutex>
#include <vector>

class MapPoint {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  using Ptr = std::shared_ptr<MapPoint>;

  MapPoint(float x, float y, float z);

  Eigen::Vector3d toEigenVector3d();

  void fromEigenVector3d(const Eigen::Vector3d &vec);

  void rotate(const Eigen::Quaterniond &q);

  size_t getId() const;

  std::vector<std::pair<size_t, size_t>> getObservation();

  // obs.first: frame_id
  // osb.second: keypoint index in frame with frame_id
  std::mutex mutex_observation_;
  std::vector<std::pair<size_t, size_t>> observations_;

private:
  std::mutex mutex_;
  float x_;
  float y_;
  float z_;

  // id
  size_t mp_id_;
  static size_t total_mp_cnt_;
};
