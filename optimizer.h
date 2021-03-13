/**
 * @file optimizer.h
 * @author xiaotaw (you@domain.com)
 * @brief g2o optimizer
 * @version 0.1
 * @date 2021-02-19
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once
#include "frame.h"
#include "g2o_types_linear_motion.h"
#include "mappoint.h"
#include "third_party/g2o/g2o/core/block_solver.h"
#include "third_party/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "third_party/g2o/g2o/core/robust_kernel_impl.h"
#include "third_party/g2o/g2o/core/solver.h"
#include "third_party/g2o/g2o/core/sparse_optimizer.h"
#include "third_party/g2o/g2o/solvers/dense/linear_solver_dense.h"
#include "third_party/g2o/g2o/solvers/eigen/linear_solver_eigen.h"
#include "third_party/g2o/g2o/solvers/structure_only/structure_only_solver.h"
#include "third_party/g2o/g2o/stuff/sampler.h"
#include "third_party/g2o/g2o/types/sba/types_six_dof_expmap.h"

class G2oOptimizer {
public:
  using Ptr = std::shared_ptr<G2oOptimizer>;

  G2oOptimizer(std::map<size_t, std::pair<Frame::Ptr, bool>> frames_data,
               std::map<size_t, std::pair<MapPoint::Ptr, bool>> mps_data,
               std::map<size_t, std::vector<std::pair<size_t, size_t>>>
                   observations_data)
      : frames_data_(frames_data), mps_data_(mps_data),
        observations_data_(observations_data) {}

  /**
   * @brief BA优化
   * @param[in] n_iteration 迭代次数
   */
  void optimize(const int &n_iteration = 10);

  /**
   * @brief 限制直线运动的BA优化
   * @param[in] n_iteration 迭代次数
   */
  void optimizeLinearMotion(const int &n_iteration = 10);

  /**
   * @brief calculate the minimal rotation, which can convert src to dst.
   * @return Eigen::Quaterniond
   */
  static Eigen::Quaterniond calMinRotation(const Eigen::Vector3d &src,
                                           const Eigen::Vector3d &dst);

  /**
   * @brief 计算优化后的mp_data_中的地图点的坐标平均值
   * @return Eigen::Vector3d
   */
  Eigen::Vector3d calAveMapPoint();

private:
  /**
   * @brief {frame_id, frame, fixed}, fixed表示该帧的pose是否固定
   */
  std::map<size_t, std::pair<Frame::Ptr, bool>> frames_data_;

  /**
   * @brief {mp_id, mappoint, fixed}, fixed表示该地图点的坐标是否固定
   */
  std::map<size_t, std::pair<MapPoint::Ptr, bool>> mps_data_;

  /**
   * @brief {mp_id, [(frame_id, keypoint_id), ]},
   * 地图点mp，被若干帧所观测到，如：第frame_id帧中的第keypoint_id个特征点
   */
  std::map<size_t, std::vector<std::pair<size_t, size_t>>> observations_data_;
};