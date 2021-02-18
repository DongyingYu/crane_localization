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
#include <memory> // for shared_ptr etc.
#include <Eigen/Core>
#include <vector>

class MapPoint
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  
  using Ptr = std::shared_ptr<MapPoint>;
  using WPtr = std::weak_ptr<MapPoint>;

  MapPoint(float x, float y, float z) : x_(x), y_(y), z_(z) {}

  inline Eigen::Vector3d toEigenVector3d() const
  {
    Eigen::Vector3d ret;
    ret << x_, y_, z_;
    return ret;
  }


  // obs.first: frame idx in map.frames_
  // osb.second: keypoint idx in this frame
  std::vector<std::pair<int, int>> observations_;

  float x_;
  float y_;
  float z_;
};
