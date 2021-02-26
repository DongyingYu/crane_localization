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
#include "utils.h"
#include <cmath> // for M_PI

void G2oOptimizer::optimizeFramePose(Frame::Ptr frame, Map *map,
                                     const int &n_iteration) {

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
    auto kp = frame->un_keypoints_[i];
    Eigen::Vector2d uv;
    uv << kp.pt.x, kp.pt.y;

    auto e = new g2o::EdgeSE3ProjectXYZOnlyPose();
    e->setVertex(
        0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
    e->setMeasurement(uv);
    e->setInformation(Eigen::Matrix2d::Identity());

    std::vector<double> intr_vec = frame->camera_model_->getNewIntrinsicVec();
    e->fx = intr_vec[0];
    e->fy = intr_vec[1];
    e->cx = intr_vec[2];
    e->cy = intr_vec[3];

    e->Xw = map->mappoints_[mp_idx]->toEigenVector3d();

    optimizer.addEdge(e);
  }

  // optimize
  optimizer.initializeOptimization();
  optimizer.optimize(n_iteration);

  // optimizatioin result
  auto v = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
  Eigen::Matrix4d mat = v->estimate().to_homogeneous_matrix();

  std::cout << "[INFO]: before optimization frame->Tcw_: " << std::endl;
  std::cout << frame->Tcw_ << std::endl;
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
  std::cout << "[DEBUG]: g2o ba map: map->frames_.size() = "
            << map->frames_.size() << std::endl;
  int frame_id_max = -1;
  for (int i = 0; i < map->frames_.size(); ++i) {
    auto v = new g2o::VertexSE3Expmap();
    int frame_id = map->frames_[i]->frame_id_;
    frame_id_max = std::max(frame_id_max, frame_id);
    v->setId(frame_id);
    v->setFixed(frame_id == 0);
    v->setEstimate(g2o::SE3Quat(map->frames_[i]->getEigenR(),
                                map->frames_[i]->getEigenT()));
    optimizer.addVertex(v);
  }

  std::vector<g2o::EdgeSE3ProjectXYZ *> edges;
  // mappoint vertex and e
  for (int i = 0; i < map->mappoints_.size(); ++i) {
    int id = frame_id_max + 1 + i;
    auto v = new g2o::VertexSBAPointXYZ();
    v->setId(id);
    v->setEstimate(map->mappoints_[i]->toEigenVector3d());
    v->setMarginalized(true);
    optimizer.addVertex(v);

    for (const std::pair<int, int> &obs : map->mappoints_[i]->observations_) {
      int frame_id = obs.first;
      if (!optimizer.vertex(frame_id)) {
        continue;
      }
      int keypoint_id = obs.second;
      auto kp = map->frames_[frame_id]->un_keypoints_[keypoint_id];
      Eigen::Vector2d uv;
      uv << kp.pt.x, kp.pt.y;

      auto e = new g2o::EdgeSE3ProjectXYZ();
      e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                          optimizer.vertex(id)));
      e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                          optimizer.vertex(frame_id)));
      e->setMeasurement(uv);
      e->setInformation(Eigen::Matrix2d::Identity());

      std::vector<double> intr_vec =
          map->frames_[frame_id]->camera_model_->getNewIntrinsicVec();
      e->fx = intr_vec[0];
      e->fy = intr_vec[1];
      e->cx = intr_vec[2];
      e->cy = intr_vec[3];

      edges.emplace_back(e);
      optimizer.addEdge(e);
    }
  }

  // optimize
  optimizer.initializeOptimization();

  std::vector<double> chi2s;
  for (auto &e : edges) {
    e->computeError();
    chi2s.emplace_back(e->chi2());
  }
  statistic(chi2s, "edges' chi2 before optimize");

  optimizer.optimize(n_iteration);

  chi2s.clear();
  for (auto &e : edges) {
    chi2s.emplace_back(e->chi2());
  }
  statistic(chi2s, "edges' chi2 after optimize");

  // optimize result

  // camera pose
  for (int i = 0; i < map->frames_.size(); ++i) {
    int frame_id = map->frames_[i]->frame_id_;

    auto v = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(frame_id));
    Eigen::Matrix4d emat = v->estimate().to_homogeneous_matrix();
    map->frames_[i]->setPose(emat);
  }
  // mappoints
  for (int i = 0; i < map->mappoints_.size(); ++i) {
    auto &mp = map->mappoints_[i];
    int id = frame_id_max + 1 + i;
    auto v = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(id));
    Eigen::Vector3d evec = v->estimate();

    // 输出第一个数据，做debug用
    if (i == 0) {
      Frame::Ptr frame = map->frames_.front();
      cv::Point2f proj_before = frame->project(mp->x_, mp->y_, mp->z_);
      cv::Point2f proj_after = frame->project(evec[0], evec[1], evec[2]);

      std::cout << "mp: before " << mp->toEigenVector3d().transpose()
                << std::endl;
      std::cout << "mp: after " << evec.transpose() << std::endl;

      int kp_idx = -1;
      for (const auto &obs : mp->observations_) {
        if (obs.first == frame->frame_id_) {
          kp_idx = obs.second;
          break;
        }
      }
      if (kp_idx < 0) {
        std::cout << "[ERROR]: not found kp " << std::endl;
        exit(-1);
      }
      std::cout << "uv: un_kp.pt " << frame->un_keypoints_[kp_idx].pt
                << std::endl;
      std::cout << "uv: proj before " << proj_before << std::endl;
      std::cout << "uv: proj after " << proj_after << std::endl << std::endl;
    }
    mp->fromEigenVector3d(evec);
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

  std::cout << "[DEBUG]: src " << std::endl
            << src / s << std::endl
            << src << std::endl;
  std::cout << "         dst " << std::endl
            << dst / d << std::endl
            << dst << std::endl;
  std::cout << "         rotated src " << std::endl << q * src / s << std::endl;

  return q;
}

void G2oOptimizerForLinearMotion::mapBundleAdjustment(Map::Ptr map,
                                                      const int &n_iteration) {

  // map->rotateFrameToXTranslation();

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
    auto q = Eigen::Quaterniond(map->frames_.front()->getEigenR());
    std::cout << "[DEBUG]: shared rotation: " << toString(q) << std::endl;
    v->setEstimate(q);
    optimizer.addVertex(v);
  }

  using EdgeVec = std::vector<g2o::EdgeSO3LinearMotionProjectXYZOnlyPose *>;
  EdgeVec edges_all;
  std::unordered_map<int, EdgeVec> edges_frame;

  // 顶点类型2：每一帧的平移量（设定为x方向）
  for (int i = 0; i < map->frames_.size(); ++i) {
    int frame_id = map->frames_[i]->frame_id_;
    edges_frame[frame_id] = EdgeVec();

    Eigen::Vector3d trans = map->frames_[i]->getEigenT();
    std::cout << "[DEBUG]: trans of frame " << frame_id << ": "
              << toString(trans) << std::endl;

    auto v = new g2o::VertexLineTranslation();
    v->setId(1 + frame_id);
    v->setFixed(frame_id == 0);
    v->setEstimate(trans[0]);
    v->setMarginalized(true);
    optimizer.addVertex(v);
  }

  // 边
  for (int i = 0; i < map->mappoints_.size(); ++i) {
    MapPoint::Ptr &mp = map->mappoints_[i];
    Eigen::Vector3d Xw = mp->toEigenVector3d();

    for (const std::pair<int, int> &obs : mp->observations_) {
      int frame_id = obs.first;
      if (!optimizer.vertex(frame_id + 1)) {
        std::cout << "[WARNING]: some observations not used in g2o"
                  << std::endl;
        continue;
      }
      int keypoint_id = obs.second;
      Frame::Ptr frame = map->frames_[frame_id];
      assert(frame_id == frame->frame_id_);
      auto kp = frame->un_keypoints_[keypoint_id];
      Eigen::Vector2d uv(kp.pt.x, kp.pt.y);

      Eigen::Matrix3d K;
      cv::cv2eigen(frame->camera_model_->getNewK(), K);
      auto e = new g2o::EdgeSO3LinearMotionProjectXYZOnlyPose(Xw, K);
      e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                          optimizer.vertex(0)));
      e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                          optimizer.vertex(1 + frame_id)));
      e->setMeasurement(uv);
      e->setInformation(Eigen::Matrix2d::Identity());

      edges_frame[frame_id].emplace_back(e);
      edges_all.emplace_back(e);
      optimizer.addEdge(e);
    }
  }

  // optimize
  optimizer.initializeOptimization();
  {
    std::vector<double> chi2s;
    for (auto &e : edges_all) {
      e->computeError();
      chi2s.emplace_back(e->chi2());
    }
    statistic(chi2s, "edges' chi2 before optimize");
  }
  for (auto &it : edges_frame) {
    int frame_id = it.first;
    EdgeVec edges = it.second;
    std::vector<double> chi2s;
    for (auto &e : edges) {
      chi2s.emplace_back(e->chi2());
    }
    statistic(chi2s, "frame " + std::to_string(frame_id) +
                         ": edges' chi2 before optimize");
  }

  optimizer.optimize(n_iteration);

  {
    std::vector<double> chi2s;
    for (auto &e : edges_all) {
      chi2s.emplace_back(e->chi2());
    }
    statistic(chi2s, "edges' chi2 after optimize");
  }
  for (auto &it : edges_frame) {
    int frame_id = it.first;
    EdgeVec edges = it.second;
    std::vector<double> chi2s;
    for (auto &e : edges) {
      chi2s.emplace_back(e->chi2());
    }
    statistic(chi2s, "frame " + std::to_string(frame_id) +
                         ": edges' chi2 before optimize");
  }
}