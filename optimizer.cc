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

void G2oOptimizer::optimizeFramePose(Frame::Ptr frame, Map *map,
                                     const int &n_iteration) {

  std::cout << "[INFO]: before optimization frame->Tcw_: " << std::endl;
  std::cout << frame->Tcw_ << std::endl;

  // create g2o optimizer
  g2o::SparseOptimizer optimizer;
  optimizer.setVerbose(true);

  using BlockSolverType = g2o::BlockSolver_6_3;
  using LinearSolverType =
      g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;
  auto solver = new g2o::OptimizationAlgorithmLevenberg(
      g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
  optimizer.setAlgorithm(solver);

  {
    // camera vertex
    auto v = new g2o::VertexSE3Expmap();
    v->setId(0);
    v->setEstimate(g2o::SE3Quat(frame->getEigenR(), frame->getEigenT()));
    optimizer.addVertex(v);
  }

  // edges
  int mp_cnt = 0;
  for (int i = 0; i < frame->mappoint_idx_.size(); ++i) {
    int mp_idx = frame->mappoint_idx_[i];
    if (mp_idx < 0) {
      continue;
    }
    auto kp = frame->keypoints_[i];
    Eigen::Vector2d uv;
    uv << kp.pt.x, kp.pt.y;

    auto e = new g2o::EdgeSE3ProjectXYZOnlyPose();
    e->setVertex(
        0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
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
  auto v = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
  Eigen::Matrix4d mat = v->estimate().to_homogeneous_matrix();
  frame->setPose(mat);

  std::cout << "[INFO]: after optimization frame->Tcw_: " << std::endl;
  std::cout << frame->Tcw_ << std::endl;

  // // release resource
  // for (int i=0; i<int(optimizer.vertices().size()); ++i) {
  //   auto v = optimizer.vertices()[i];
  //   delete v;
  // }
  // for (int i=0; i<int(optimizer.edges().size()); ++i) {
  //   auto e = optimizer.edges()[i];
  //   delete e;
  // }
  // optimizer.clear();
  // delete solver;
  // solver = nullptr;
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
    v->setEstimate(g2o::SE3Quat(map->frames_[i]->getEigenR(),
                                map->frames_[i]->getEigenT()));
    optimizer.addVertex(v);
  }

  // mappoint vertex and e
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

      auto e = new g2o::EdgeSE3ProjectXYZ();
      e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                          optimizer.vertex(id)));
      e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                          optimizer.vertex(frame_id)));
      e->setMeasurement(uv);
      e->setInformation(Eigen::Matrix2d::Identity());

      e->fx = map->frames_[frame_id]->intrinsic_.fx;
      e->fy = map->frames_[frame_id]->intrinsic_.fy;
      e->cx = map->frames_[frame_id]->intrinsic_.cx;
      e->cy = map->frames_[frame_id]->intrinsic_.cy;

      optimizer.addEdge(e);
    }
  }

  // optimize
  optimizer.initializeOptimization();
  optimizer.optimize(n_iteration);

  // optimize result

  // camera pose
  for (int i = 0; i < map->frames_.size(); ++i) {
    auto v = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(i));
    Eigen::Matrix4d emat = v->estimate().to_homogeneous_matrix();
    map->frames_[i]->setPose(emat);
    std::cout << "frame " << i << std::endl << emat << std::endl;
  }
  // mappoints
  for (int i = 0; i < map->mappoints_.size(); ++i) {
    auto v = static_cast<g2o::VertexSBAPointXYZ *>(
        optimizer.vertex(i + map->frames_.size()));
    Eigen::Vector3d evec = v->estimate();
    map->mappoints_[i]->setValue(evec);
    // std::cout << "mappoint " << i << " change ";
    // std::cout << evec - map->mappoints_[i]->toEigenVector3d() << std::endl;
  }

  // // release resource
  // for (int i=0; i<int(optimizer.vertices().size()); ++i) {
  //   auto v = optimizer.vertices()[i];
  //   delete v;
  // }
  // for (int i=0; i<int(optimizer.edges().size()); ++i) {
  //   auto e = optimizer.edges()[i];
  //   delete e;
  // }
  // optimizer.clear();
  // delete solver;
  // solver = nullptr;
}

namespace g2o {}

/**
 * @brief calculate the minimal rotation, which can convert src to dst.
 * @return g2o::Quaternion
 */
Eigen::Quaterniond
G2oOptimizerForLinearMotion::calMinRotation(const Eigen::Vector3d &src,
                                            const Eigen::Vector3d &dst) {
  // 计算src和dst之间的夹角
  double s = src.norm();
  double d = dst.norm();
  assert(s > std::numeric_limits<double>::epsilon());
  assert(d > std::numeric_limits<double>::epsilon());
  double cos_theta = src.dot(dst) / (s * d);
  if (cos_theta < 0) {
    std::cout << "[WARNING]: cos_theta is expected to be greater than zero: "
              << cos_theta << std::endl;
  }
  double theta = std::acos(cos_theta);

  // 计算src和dst的法向量
  Eigen::Vector3d n = src.cross(dst);
  n.normalize();
  // 旋转向量
  Eigen::AngleAxisd aa(theta, n);
  Eigen::Quaterniond q(aa);

  std::cout << "[DEBUG]: src " << s << std::endl
            << src / s << std::endl
            << src << std::endl;
  std::cout << "         dst " << d << std::endl
            << dst / d << std::endl
            << dst << std::endl;
  std::cout << "         rotated src " << std::endl << q * src / s << std::endl;

  return q;
}

void G2oOptimizerForLinearMotion::mapBundleAdjustment(Map::Ptr map,
                                                      const int &n_iteration) {
  // 初始旋转量
  Eigen::Vector3d trans = map->frames_.back()->getEigenT();
  Eigen::Quaterniond q_init = calMinRotation(trans, Eigen::Vector3d(1, 0, 0));

  // optimizer
  g2o::SparseOptimizer optimizer;
  optimizer.setVerbose(true);
  using BlockSolverType = g2o::BlockSolverPL<3, 1>;
  using LinearSolverType =
      g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;
  auto solver = new g2o::OptimizationAlgorithmLevenberg(
      g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
  optimizer.setAlgorithm(solver);

  // 顶点类型1：公共的旋转量
  {
    auto v = new g2o::VertexSO3Expmap();
    v->setId(0);
    v->setEstimate(q_init);
    optimizer.addVertex(v);
  }

  // 顶点类型2：每一帧的平移量（设定为x方向）
  for (int i = 0; i < map->frames_.size(); ++i) {
    Eigen::Vector3d trans = map->frames_[i]->getEigenT();

    auto v = new g2o::VertexLineTranslation();
    v->setId(1 + i);
    v->setFixed(i == 0);
    v->setEstimate((q_init * trans)[0]);
    v->setMarginalized(true);
    optimizer.addVertex(v);
  }

  // 边
  for (int i = 0; i < map->mappoints_.size(); ++i) {
    MapPoint::Ptr &mp = map->mappoints_[i];
    Eigen::Vector3d Xw = mp->toEigenVector3d();

    for (const std::pair<int, int> &obs : mp->observations_) {
      int frame_id = obs.first;
      int keypoint_id = obs.second;
      Frame::Ptr frame = map->frames_[frame_id];
      auto kp = frame->keypoints_[keypoint_id];
      Eigen::Vector2d uv(kp.pt.x, kp.pt.y);

      auto e = new g2o::EdgeSO3LinearMotionProjectXYZOnlyPose(
          Xw, frame->intrinsic_.getEigenK());

      e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                          optimizer.vertex(0)));
      e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                          optimizer.vertex(1 + frame_id)));
      e->setMeasurement(uv);
      e->setInformation(Eigen::Matrix2d::Identity());

      optimizer.addEdge(e);
    }
  }

  // optimize
  optimizer.initializeOptimization();
  optimizer.optimize(n_iteration);
}