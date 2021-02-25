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
#include "utils.h"
#include <numeric>

bool Map::trackNewFrame(Frame::Ptr cur_frame) {
  Frame::Ptr last_frame = frames_.back();

  // 1. 特征点匹配
  std::vector<cv::DMatch> good_matches;
  std::vector<cv::Point2f> points1, points2;
  int n_match =
      last_frame->matchWith(cur_frame, good_matches, points1, points2, true);
  if (n_match < 50) {
    std::cout
        << "[WARNING]: Too less matched keypoint, this may lead to wrong pose: "
        << n_match << std::endl;
  }

  // 2. 使用PnP给出当前帧的相机位姿
  int cnt_3d = 0;
  for (const cv::DMatch &m : good_matches) {
    int x3D_idx = last_frame->mappoint_idx_[m.queryIdx];
    if (x3D_idx >= 0) {
      cnt_3d++;
      cur_frame->mappoint_idx_[m.trainIdx] = x3D_idx;
    }
  }
  std::cout << "[INFO]: cnt_3d=" << cnt_3d << " cnt_not_3d=" << n_match - cnt_3d
            << std::endl;
  cur_frame->setPose(last_frame->Tcw_);
  G2oOptimizer::optimizeFramePose(cur_frame, this, 10);

  // 3. 将剩余配对特征点三角化
  
  

  // todo
  // 计算重投影误差，排除外点，之后，重新优化；或者采用类似orbslam2的方式，四次迭代，每次迭代中判断内点和外点
}

void Map::rotateFrameToXTranslation() {
  Eigen::Vector3d twc = frames_.back()->getEigenTwc();
  if (twc.norm() < std::numeric_limits<float>::epsilon()) {
    std::cout << "[WARNING]: twc.norm() is too little: " << twc.norm()
              << std::endl;
    return;
  }

  // unfinished
}

void Map::printMap() const {
  for (const auto &frame : frames_) {
    std::cout << "Frame: " << frame->frame_id_ << std::endl;
    std::cout << frame->Tcw_ << std::endl;
  }
  Eigen::Vector3d twc1 = frames_.back()->getEigenTwc();
  std::cout << "[INFO]: twc of the last frame " << toString(twc1) << std::endl;

  Eigen::Vector3d sum = Eigen::Vector3d::Zero();
  for (const auto &mp : mappoints_) {
    sum += mp->toEigenVector3d();
  }
  std::cout << "[INFO]: MapPoint mean: " << toString(sum / mappoints_.size())
            << std::endl;

  std::cout << "[INFO]: twc of the last frame "
            << toString(twc1 * 9 * (sum / mappoints_.size())[2]) << std::endl;
}