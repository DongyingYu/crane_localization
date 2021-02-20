/**
 * @file map.cc
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2021-02-09
 *
 * @copyright Copyright (c) 2021
 *
 */
#include "map.h"
#include "optimizer.h"
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
#include <numeric>

bool Map::trackNewFrame(Frame::Ptr cur_frame) {
  Frame::Ptr last_frame = frames_.back();
  std::vector<cv::DMatch> good_matches;
  last_frame->matchWith(cur_frame, good_matches, true);

  // 特征点匹配
  int cnt_3d = 0, cnt_not_3d = 0;
  for (const cv::DMatch &m : good_matches) {
    int x3D_idx = last_frame->mappoint_idx_[m.queryIdx];
    if (x3D_idx >= 0) {
      cnt_3d++;
      cur_frame->mappoint_idx_[m.trainIdx] = x3D_idx;
    } else {
      cnt_not_3d++;
    }
  }
  std::cout << "[INFO]: cnt_3d=" << cnt_3d << " cnt_not_3d=" << cnt_not_3d
            << std::endl;

  if (cnt_3d < 50) {
    std::cout << "[WARNING]: Matched 3d mappoints is less than 50: " << cnt_3d
              << std::endl;
    std::cout << "           This may lead to wrong pose." << std::endl;
  }

  // todo 将cnt_not_3d个特征点的匹配，三角化，添加进入地图。

  // 优化相机位姿
  cur_frame->setPose(last_frame->Tcw_);
  G2oOptimizer::optimizeFramePose(cur_frame, this, 10);

  // todo
  // 计算重投影误差，排除外点，之后，重新优化；或者采用类似orbslam2的方式，四次迭代，每次迭代中判断内点和外点
}
