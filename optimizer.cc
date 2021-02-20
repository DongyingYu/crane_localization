/**
 * @file optimizer.cc
 * @author xiaotaw (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2021-02-19
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#include "optimizer.h"

void G2oOptimizer::optimizeFramePose(Frame::Ptr frame, Map* map, const int& n_iteration) {

  std::cout << "[INFO]: before optimization frame->Tcw_: " << std::endl;
  std::cout << frame->Tcw_ << std::endl; 

  // create g2o optimizer
  g2o::SparseOptimizer optimizer;
  optimizer.setVerbose(true);

  using BlockSolverType = g2o::BlockSolver_6_3;
  using LinearSolverType = g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;
  auto solver = new g2o::OptimizationAlgorithmLevenberg(
    g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>())
  );
  optimizer.setAlgorithm(solver);

  {
    // camera vertex
    auto v = new g2o::VertexSE3Expmap();
    v->setId(0);
    v->setEstimate(
      g2o::SE3Quat(frame->getEigenR(), frame->getEigenT())
    );
    optimizer.addVertex(v);
  }

  // edge
  int mp_cnt = 0;
  for(int i=0; i<frame->mappoint_idx_.size(); ++i){
      int mp_idx = frame->mappoint_idx_[i];
      if (mp_idx < 0) {
        continue;
      }
      auto kp = frame->keypoints_[i];
      Eigen::Vector2d uv;
      uv << kp.pt.x, kp.pt.y;

      auto e = new g2o::EdgeSE3ProjectXYZOnlyPose();
      e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
      e->setMeasurement(uv);
      e->setInformation(Eigen::Matrix2d::Identity());

      e->fx = frame->intrinsic_.fx;
      e->fy = frame->intrinsic_.fy;
      e->cx = frame->intrinsic_.cx;
      e->cy = frame->intrinsic_.cy;
      
      e->Xw = map->mappoints_[mp_idx]->toEigenVector3d();

      optimizer.addEdge(e);

  }

  // optimize
  optimizer.initializeOptimization();
  optimizer.optimize(n_iteration);

  // optimizatioin result
  auto v = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
  Eigen::Matrix4d mat = v->estimate().to_homogeneous_matrix();
  frame->setPose(mat);

  std::cout << "[INFO]: after optimization frame->Tcw_: " << std::endl;
  std::cout << frame->Tcw_ << std::endl; 

}


void G2oOptimizer::mapBundleAdjustment(Map::Ptr map, const int &n_iteration) {
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
  for (int i = 0; i < map->frames_.size(); ++i) {
    auto v = new g2o::VertexSE3Expmap();
    v->setId(i);
    v->setFixed(i == 0);
    v->setEstimate(
        g2o::SE3Quat(map->frames_[i]->getEigenR(), map->frames_[i]->getEigenT()));
    optimizer.addVertex(v);
  }

  // mappoint vertex and edge
  for (int i = 0; i < map->mappoints_.size(); ++i) {
    int id = map->frames_.size() + i;
    auto v = new g2o::VertexSBAPointXYZ();
    v->setId(id);
    v->setEstimate(map->mappoints_[i]->toEigenVector3d());
    v->setMarginalized(true);
    optimizer.addVertex(v);

    for (const std::pair<int, int> &obs : map->mappoints_[i]->observations_) {
      int frame_id = obs.first;
      int keypoint_id = obs.second;
      auto kp = map->frames_[frame_id]->keypoints_[keypoint_id];
      Eigen::Vector2d uv;
      uv << kp.pt.x, kp.pt.y;

      auto edge = new g2o::EdgeSE3ProjectXYZ();
      edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                             optimizer.vertex(id)));
      edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                             optimizer.vertex(frame_id)));
      edge->setMeasurement(uv);
      edge->setInformation(Eigen::Matrix2d::Identity());

      edge->fx = map->frames_[frame_id]->intrinsic_.fx;
      edge->fy = map->frames_[frame_id]->intrinsic_.fy;
      edge->cx = map->frames_[frame_id]->intrinsic_.cx;
      edge->cy = map->frames_[frame_id]->intrinsic_.cy;

      optimizer.addEdge(edge);
    }
  }

  // optimize
  optimizer.initializeOptimization();
  optimizer.optimize(n_iteration);

  // optimize result

  // camera pose
  for (int i=0; i<map->frames_.size(); ++i) {
    auto v = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(i));
    Eigen::Matrix4d emat = v->estimate().to_homogeneous_matrix();
    map->frames_[i]->setPose(emat);
    std::cout << "frame " << i << std::endl << emat << std::endl;
  }
  // mappoints
  for (int i=0; i<map->mappoints_.size(); ++i){
    auto v = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(i+map->frames_.size()));
    Eigen::Vector3d evec = v->estimate();
    map->mappoints_[i]->setValue(evec);
    // std::cout << "mappoint " << i << " change ";
    // std::cout << evec - map->mappoints_[i]->toEigenVector3d() << std::endl;
  }
}