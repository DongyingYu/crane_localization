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
#include "map.h"
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
  /**
   * @brief 优化frame的位姿
   *
   * @param[in out] frame 待优化的帧
   * @param[in] map 当前地图
   * @param[in] n_iteration 迭代次数
   *
   * @todo 重新设计，避免使用裸指针Map*
   */
  static void optimizeFramePose(Frame::Ptr frame, Map *map,
                                const int &n_iteration = 10);

  /**
   * @brief 优化map中的mappoints和frame pose
   *
   * @param[in out] map 待优化的地图
   * @param[in] n_iteration 迭代次数
   */
  static void mapBundleAdjustment(Map::Ptr map, const int &n_iteration = 25);
};

/**
 * @brief 限定直线运动的图优化
 *
 */
class G2oOptimizerForLinearMotion {
public:
  /**
   * @brief calculate the minimal rotation, which can convert src to dst.
   * @return Eigen::Quaterniond
   */
  static Eigen::Quaterniond calMinRotation(const Eigen::Vector3d &src,
                                           const Eigen::Vector3d &dst);

  static void mapBundleAdjustmentOnlyPose(Map::Ptr map,
                                          const int &n_iteration = 10);

  static void mapBundleAdjustment(Map::Ptr map,
                                          const int &n_iteration = 10);

  
};