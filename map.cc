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

bool Map::track(Frame::Ptr frame) {}

void Map::initial_ba() {
  // create optimizer
  g2o::SparseOptimizer optimizer;
  optimizer.setVerbose(true);

  // solver algorithm
  using BlockSolverType = g2o::BlockSolver_6_3;
  // using LinearSolverType =
  //     g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;
  using LinearSolverType =
      g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>;
  auto solver = new g2o::OptimizationAlgorithmLevenberg(
      g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
  optimizer.setAlgorithm(solver);

  // camera vertex
  for (int i = 0; i < frames_.size(); ++i) {
    auto v = new g2o::VertexSE3Expmap();
    v->setId(i);
    v->setFixed(i == 0);
    v->setEstimate(
        g2o::SE3Quat(frames_[i]->getEigenR(), frames_[i]->getEigenT()));
    optimizer.addVertex(v);
  }

  // mappoint vertex and edge
  for (int i = 0; i < mappoints_.size(); ++i) {
    int id = frames_.size() + i;
    auto v = new g2o::VertexSBAPointXYZ();
    v->setId(id);
    v->setEstimate(mappoints_[i]->toEigenVector3d());
    optimizer.addVertex(v);

    for (const std::pair<int, int> &obs : mappoints_[i]->observations_) {
      int frame_id = obs.first;
      int keypoint_id = obs.second;
      auto kp = frames_[frame_id]->keypoints_[keypoint_id];
      Eigen::Vector2d uv;
      uv << kp.pt.x, kp.pt.y;

      auto edge = new g2o::EdgeSE3ProjectXYZ();
      edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                             optimizer.vertex(id)));
      edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                             optimizer.vertex(frame_id)));
      edge->setMeasurement(uv);
      edge->setInformation(Eigen::Matrix2d::Identity());

      edge->fx = frames_[frame_id]->intrinsic_.fx;
      edge->fy = frames_[frame_id]->intrinsic_.fy;
      edge->cx = frames_[frame_id]->intrinsic_.cx;
      edge->cy = frames_[frame_id]->intrinsic_.cy;

      optimizer.addEdge(edge);
    }
  }

  // optimize
  optimizer.initializeOptimization();
  optimizer.optimize(10);
}