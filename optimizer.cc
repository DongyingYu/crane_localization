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

#define G2O_OPT_VERBOSE false

/**
 * @brief 统计输出边的卡方信息
 *
 * @tparam EdgeType g2o边的类型
 * @param[in out] compute_error 是否需要调用edge的computeError函数
 * @param[in out] edges_data 按frame_id索引的所有边
 */
template <typename EdgeType>
static void
debugPrintEdges(bool compute_error,
                std::map<size_t, std::vector<EdgeType *>> edges_data) {
  std::vector<double> chi2s_all;
  for (auto &it : edges_data) {
    const size_t &frame_id = it.first;
    auto &edges = it.second;
    std::vector<double> chi2s;
    for (auto &e : edges) {
      if (compute_error) {
        e->computeError();
      }
      double chi2 = e->chi2();
      chi2s.emplace_back(chi2);
      chi2s_all.emplace_back(chi2);
    }
    statistic(chi2s, "[DEBUG]: G2o Optimization, Chi2 for frame " +
                         std::to_string(frame_id));
  }
  statistic(chi2s_all, "[DEBUG]: G2o Optimization, Chi2 for all frames");
}

void G2oOptimizer::optimizeFramePose(Frame::Ptr frame, Map *map,
                                     const int &n_iteration) {

  // create g2o optimizer
  g2o::SparseOptimizer optimizer;
  optimizer.setVerbose(G2O_OPT_VERBOSE);

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
    v->setEstimate(g2o::SE3Quat(frame->getEigenRot(), frame->getEigenTrans()));
    optimizer.addVertex(v);
  }

  // edges
  int mp_cnt = 0;
  for (int i = 0; i < frame->getMappointIdx().size(); ++i) {
    int mp_idx = frame->getMappointIdx(i);
    if (mp_idx < 0) {
      continue;
    }
    auto kp = frame->getUnKeyPoints(i);
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

    e->Xw = map->getMapPointById(mp_idx)->toEigenVector3d();

    optimizer.addEdge(e);
  }

  // optimize
  optimizer.initializeOptimization();
  optimizer.optimize(n_iteration);

  // optimizatioin result
  auto v = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
  Eigen::Matrix4d mat = v->estimate().to_homogeneous_matrix();

  // std::cout << "[INFO]: before optimization frame->Tcw_: " << std::endl;
  // std::cout << frame->getPose() << std::endl;
  frame->setPose(mat);
  // std::cout << "[INFO]: after optimization frame->Tcw_: " << std::endl;
  // std::cout << frame->getPose() << std::endl;

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
  optimizer.setVerbose(G2O_OPT_VERBOSE);

  // solver algorithm
  using BlockSolverType = g2o::BlockSolver_6_3;
  // using LinearSolverType =
  //     g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;
  using LinearSolverType =
      g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>;
  auto solver = new g2o::OptimizationAlgorithmLevenberg(
      g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
  optimizer.setAlgorithm(solver);

  using EdgeType = g2o::EdgeSE3ProjectXYZ;
  std::map<size_t, std::vector<EdgeType *>> edges_data;

  // camera vertex
  int frame_id_max = -1;
  for (int i = 0; i < map->recent_frames_.size(); ++i) {
    auto v = new g2o::VertexSE3Expmap();
    int frame_id = map->recent_frames_[i]->getFrameId();
    frame_id_max = std::max(frame_id_max, frame_id);
    edges_data[frame_id] = std::vector<EdgeType *>();
    v->setId(frame_id);
    v->setFixed(frame_id == 0);
    v->setEstimate(g2o::SE3Quat(map->recent_frames_[i]->getEigenRot(),
                                map->recent_frames_[i]->getEigenTrans()));
    optimizer.addVertex(v);
  }

  // mappoint vertex and e
  // size_t mp_size = map->getMapPointSize();
  // for (int i = 0; i < int(mp_size); ++i) {
  std::vector<MapPoint::Ptr> mps = map->getMapPoints();
  for (auto &mp : mps) {
    int i = mp->getId();
    if (!mp) {
      continue;
    }
    int id = frame_id_max + 1 + i;
    auto v = new g2o::VertexSBAPointXYZ();
    v->setId(id);
    v->setEstimate(mp->toEigenVector3d());
    v->setMarginalized(true);
    optimizer.addVertex(v);

    for (const std::pair<int, int> &obs : mp->observations_) {
      int frame_id = obs.first;
      if (!optimizer.vertex(frame_id)) {
        continue;
      }
      int keypoint_id = obs.second;
      auto kp = map->recent_frames_[frame_id]->getUnKeyPoints(keypoint_id);
      Eigen::Vector2d uv;
      uv << kp.pt.x, kp.pt.y;

      using GVO = g2o::OptimizableGraph::Vertex;
      auto e = new g2o::EdgeSE3ProjectXYZ();
      e->setVertex(0, dynamic_cast<GVO *>(optimizer.vertex(id)));
      e->setVertex(1, dynamic_cast<GVO *>(optimizer.vertex(frame_id)));
      e->setMeasurement(uv);
      e->setInformation(Eigen::Matrix2d::Identity());

      std::vector<double> intr_vec =
          map->recent_frames_[frame_id]->camera_model_->getNewIntrinsicVec();
      e->fx = intr_vec[0];
      e->fy = intr_vec[1];
      e->cx = intr_vec[2];
      e->cy = intr_vec[3];

      edges_data[frame_id].emplace_back(e);
      optimizer.addEdge(e);
    }
  }

  // optimize
  optimizer.initializeOptimization();
  debugPrintEdges(true, edges_data);
  optimizer.optimize(n_iteration);
  debugPrintEdges(false, edges_data);

  // optimize result
  // camera pose
  for (int i = 0; i < map->recent_frames_.size(); ++i) {
    int frame_id = map->recent_frames_[i]->getFrameId();

    auto v = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(frame_id));
    Eigen::Matrix4d emat = v->estimate().to_homogeneous_matrix();
    map->recent_frames_[i]->setPose(emat);
  }
  // mappoints
  for (auto &mp : mps) {
    int i = mp->getId();
    if (!mp) {
      continue;
    }
    int id = frame_id_max + 1 + i;
    auto v = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(id));
    Eigen::Vector3d evec = v->estimate();

    // 输出第一个数据，做debug用
    if (0) {
      Frame::Ptr frame = map->recent_frames_.front();
      cv::Point2f proj_before = frame->project(mp->toEigenVector3d());
      cv::Point2f proj_after = frame->project(evec);

      std::cout << "mp: before " << mp->toEigenVector3d().transpose()
                << std::endl;
      std::cout << "mp: after " << evec.transpose() << std::endl;

      int kp_idx = -1;
      for (const auto &obs : mp->observations_) {
        if (obs.first == frame->getFrameId()) {
          kp_idx = obs.second;
          break;
        }
      }
      if (kp_idx < 0) {
        std::cout << "[ERROR]: not found kp " << std::endl;
        exit(-1);
      }
      std::cout << "uv: un_kp.pt " << frame->getUnKeyPoints(kp_idx).pt
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

void G2oOptimizerForLinearMotion::mapBundleAdjustmentOnlyPose(
    Map::Ptr map, const int &n_iteration) {

  // map->rotateFrameToXTranslation();

  // optimizer
  g2o::SparseOptimizer optimizer;
  optimizer.setVerbose(G2O_OPT_VERBOSE);
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
    auto q = Eigen::Quaterniond(map->recent_frames_.front()->getEigenRot());
    // std::cout << "[DEBUG]: shared rotation: " << toString(q) << std::endl;
    v->setEstimate(q);
    optimizer.addVertex(v);
  }

  using EdgeType = g2o::EdgeLinearMotionOnlyPose;
  std::map<size_t, std::vector<EdgeType *>> edges_data;

  // 顶点类型2：每一帧的平移量（设定为x方向）
  for (int i = 0; i < map->recent_frames_.size(); ++i) {
    int frame_id = map->recent_frames_[i]->getFrameId();
    edges_data[frame_id] = std::vector<EdgeType *>();

    Eigen::Vector3d trans = map->recent_frames_[i]->getEigenTrans();
    // std::cout << "[DEBUG]: trans of frame " << frame_id << ": "
    //           << toString(trans) << std::endl;

    auto v = new g2o::VertexLineTranslation();
    v->setId(1 + frame_id);
    v->setFixed(frame_id == 0);
    v->setEstimate(trans[0]);
    v->setMarginalized(true);
    optimizer.addVertex(v);
  }

  // 边
  size_t mp_size = map->getMapPointSize();
  for (int i = 0; i < int(mp_size); ++i) {
    MapPoint::Ptr mp = map->getMapPointById(i);
    if (!mp) {
      continue;
    }
    Eigen::Vector3d Xw = mp->toEigenVector3d();

    for (const std::pair<int, int> &obs : mp->observations_) {
      int frame_id = obs.first;
      if (!optimizer.vertex(frame_id + 1)) {
        std::cout << "[WARNING]: some observations not used in g2o"
                  << std::endl;
        continue;
      }
      int keypoint_id = obs.second;
      Frame::Ptr frame = map->recent_frames_[frame_id];
      assert(frame_id == frame->getFrameId());
      auto kp = frame->getUnKeyPoints(keypoint_id);
      Eigen::Vector2d uv(kp.pt.x, kp.pt.y);

      Eigen::Matrix3d K;
      cv::cv2eigen(frame->camera_model_->getNewK(), K);
      auto e = new g2o::EdgeLinearMotionOnlyPose(Xw, K);
      e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                          optimizer.vertex(0)));
      e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                          optimizer.vertex(1 + frame_id)));
      e->setMeasurement(uv);
      e->setInformation(Eigen::Matrix2d::Identity());

      edges_data[frame_id].emplace_back(e);
      optimizer.addEdge(e);
    }
  }

  // optimize
  optimizer.initializeOptimization();
  debugPrintEdges<EdgeType>(true, edges_data);
  optimizer.optimize(n_iteration);
  debugPrintEdges<EdgeType>(false, edges_data);
}

void G2oOptimizerForLinearMotion::mapBundleAdjustment(Map::Ptr map,
                                                      const int &n_iteration) {

  // map->rotateFrameToXTranslation();

  // optimizer
  g2o::SparseOptimizer optimizer;
  optimizer.setVerbose(G2O_OPT_VERBOSE);
  using BlockSolverType = g2o::BlockSolverX;
  using LinearSolverType =
      g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;
  auto solver = new g2o::OptimizationAlgorithmLevenberg(
      g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
  optimizer.setAlgorithm(solver);

  // 顶点类型1：公共的旋转量
  {
    auto v = new g2o::VertexSO3Expmap();
    v->setId(0);
    auto q = Eigen::Quaterniond(map->recent_frames_.front()->getEigenRot());
    // std::cout << "[DEBUG]: shared rotation: " << toString(q) << std::endl;
    v->setEstimate(q);
    optimizer.addVertex(v);
  }

  using EdgeType = g2o::EdgeLinearMotion;
  std::map<size_t, std::vector<EdgeType *>> edges_data;

  // 顶点类型2：每一帧的平移量（设定为x方向）
  int frame_id_max = -1;
  for (int i = 0; i < map->recent_frames_.size(); ++i) {
    int frame_id = map->recent_frames_[i]->getFrameId();
    frame_id_max = std::max(frame_id_max, frame_id);
    edges_data[frame_id] = std::vector<EdgeType *>();

    Eigen::Vector3d trans = map->recent_frames_[i]->getEigenTrans();
    // std::cout << "[DEBUG]: trans of frame " << frame_id << ": "
    //           << toString(trans) << std::endl;

    auto v = new g2o::VertexLineTranslation();
    v->setId(1 + frame_id);
    v->setFixed(frame_id == 0);
    v->setEstimate(trans[0]);
    optimizer.addVertex(v);
  }

  // 顶点类型3：地图点
  std::vector<MapPoint::Ptr> mps = map->getMapPoints();
  for (auto &mp : mps) {
    int i = mp->getId();
    if (!mp) {
      continue;
    }
    int id = frame_id_max + 1 + 1 + i;
    auto v = new g2o::VertexSBAPointXYZ();
    v->setId(id);
    v->setEstimate(mp->toEigenVector3d());
    v->setMarginalized(true);
    optimizer.addVertex(v);

    Eigen::Vector3d Xw = mp->toEigenVector3d();

    for (const std::pair<int, int> &obs : mp->observations_) {
      int frame_id = obs.first;
      if (!optimizer.vertex(frame_id + 1)) {
        std::cout << "[WARNING]: some observations not used in g2o"
                  << std::endl;
        continue;
      }
      int keypoint_id = obs.second;
      Frame::Ptr frame = map->recent_frames_[frame_id];
      assert(frame_id == frame->getFrameId());
      auto kp = frame->getUnKeyPoints(keypoint_id);
      Eigen::Vector2d uv(kp.pt.x, kp.pt.y);

      Eigen::Matrix3d K;
      cv::cv2eigen(frame->camera_model_->getNewK(), K);

      using GOV = g2o::OptimizableGraph::Vertex;
      auto e = new g2o::EdgeLinearMotion(K);
      e->setVertex(0, dynamic_cast<GOV *>(optimizer.vertex(0)));
      e->setVertex(1, dynamic_cast<GOV *>(optimizer.vertex(1 + frame_id)));
      e->setVertex(2, dynamic_cast<GOV *>(optimizer.vertex(id)));
      e->setMeasurement(uv);
      e->setInformation(Eigen::Matrix2d::Identity());

      edges_data[frame_id].emplace_back(e);
      optimizer.addEdge(e);
    }
  }

  // optimize
  optimizer.initializeOptimization();
  debugPrintEdges<EdgeType>(true, edges_data);
  optimizer.optimize(n_iteration);
  debugPrintEdges<EdgeType>(false, edges_data);

  // optimize result
  Eigen::Matrix3d rotation;
  {
    auto v = static_cast<g2o::VertexSO3Expmap *>(optimizer.vertex(0));
    rotation = v->estimate().toRotationMatrix();
  }

  // camera pose
  for (int i = 0; i < map->recent_frames_.size(); ++i) {
    int frame_id = map->recent_frames_[i]->getFrameId();
    auto v = static_cast<g2o::VertexLineTranslation *>(
        optimizer.vertex(frame_id + 1));
    Eigen::Vector3d trans = Eigen::Vector3d(0, 0, 0);
    trans[0] = v->estimate();
    map->recent_frames_[i]->setPose(rotation, trans);
  }

  // mappoints
  for (auto &mp : mps) {
    int i = mp->getId();
    if (!mp) {
      continue;
    }
    int id = frame_id_max + 1 + 1 + i;
    auto v = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(id));
    Eigen::Vector3d evec = v->estimate();

    // 输出第一个数据，做debug用
    if (0) {
      Frame::Ptr frame = map->recent_frames_.front();
      cv::Point2f proj_before = frame->project(mp->toEigenVector3d());
      cv::Point2f proj_after = frame->project(evec);

      std::cout << "mp: before " << mp->toEigenVector3d().transpose()
                << std::endl;
      std::cout << "mp: after " << evec.transpose() << std::endl;

      int kp_idx = -1;
      for (const auto &obs : mp->observations_) {
        if (obs.first == frame->getFrameId()) {
          kp_idx = obs.second;
          break;
        }
      }
      if (kp_idx < 0) {
        std::cout << "[ERROR]: not found kp " << std::endl;
        exit(-1);
      }
      std::cout << "uv: un_kp.pt " << frame->getUnKeyPoints(kp_idx).pt
                << std::endl;
      std::cout << "uv: proj before " << proj_before << std::endl;
      std::cout << "uv: proj after " << proj_after << std::endl << std::endl;
    }
    mp->fromEigenVector3d(evec);
  }
}

void G2oOptimizerForLinearMotion::optimize(
    std::map<size_t, std::pair<Frame::Ptr, bool>> &frames,
    std::map<size_t, std::pair<MapPoint::Ptr, bool>> &mps,
    std::map<size_t, std::vector<std::pair<size_t, size_t>>> &observations,
    const int &n_iteration) {
  // optimizer
  g2o::SparseOptimizer optimizer;
  optimizer.setVerbose(G2O_OPT_VERBOSE);
  using BlockSolverType = g2o::BlockSolverX;
  using LinearSolverType =
      g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;
  auto solver = new g2o::OptimizationAlgorithmLevenberg(
      g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
  optimizer.setAlgorithm(solver);

  // 顶点类型1：公共的旋转量
  auto v_rot = new g2o::VertexSO3Expmap();
  v_rot->setId(0);
  Frame::Ptr &frame = frames.begin()->second.first;
  const bool &fixed = frames.begin()->second.second;
  auto q = Eigen::Quaterniond(frame->getEigenRotWc());
  // std::cout << "[DEBUG]: shared rotation: " << toString(q) << std::endl;
  // v_rot->setFixed(fixed);
  v_rot->setEstimate(q);
  optimizer.addVertex(v_rot);

  using EdgeType = g2o::EdgeLinearMotion;
  std::map<size_t, std::vector<EdgeType *>> edges_data;

  // 顶点类型2：每一帧的平移量（设定为x方向）
  int frame_id_max = -1;
  for (const auto &it : frames) {
    const size_t &frame_id = it.first;
    const auto &frame = it.second.first;
    const bool &fixed = it.second.second;
    frame_id_max = std::max(frame_id_max, int(frame_id));
    edges_data[frame_id] = std::vector<EdgeType *>();
    Eigen::Vector3d twc = frame->getEigenTransWc();
    // std::cout << "[DEBUG]: trans of frame " << frame_id << ": "
    //           << toString(trans) << std::endl;

    auto v = new g2o::VertexLineTranslation();
    v->setId(1 + frame_id);
    v->setFixed(fixed);
    v->setEstimate(twc[0]);
    optimizer.addVertex(v);
  }

  // 顶点类型3：地图点
  for (const auto &it : mps) {
    const size_t &mp_id = it.first;
    const auto &mp = it.second.first;
    const bool &fixed = it.second.second;
    const size_t id = size_t(1 + frame_id_max) + 1 + mp_id;
    const Eigen::Vector3d &Xw = mp->toEigenVector3d();

    auto v = new g2o::VertexSBAPointXYZ();
    v->setId(id);
    v->setFixed(fixed);
    v->setEstimate(Xw);
    v->setMarginalized(true);
    optimizer.addVertex(v);

    // 边
    if (fixed) {
      // not implemented yet
    } else {
      for (const auto &obs : observations[mp_id]) {
        size_t frame_id = obs.first;
        size_t kp_id = obs.second;
        Frame::Ptr &frame = frames[frame_id].first;
        Eigen::Matrix3d K = frame->getEigenNewK();
        const cv::KeyPoint &kp = frame->getUnKeyPoints(kp_id);
        Eigen::Vector2d uv(kp.pt.x, kp.pt.y);

        using GOV = g2o::OptimizableGraph::Vertex;
        auto e = new g2o::EdgeLinearMotion(K);
        e->setVertex(0, dynamic_cast<GOV *>(optimizer.vertex(0)));
        e->setVertex(1, dynamic_cast<GOV *>(optimizer.vertex(1 + frame_id)));
        e->setVertex(2, dynamic_cast<GOV *>(optimizer.vertex(id)));
        e->setMeasurement(uv);
        e->setInformation(Eigen::Matrix2d::Identity());

        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
        e->setRobustKernel(rk);
        rk->setDelta(25);

        edges_data[frame_id].emplace_back(e);
        optimizer.addEdge(e);
      }
    }
  }

  // optimize
  optimizer.initializeOptimization();
  debugPrintEdges(true, edges_data);
  optimizer.optimize(n_iteration);
  debugPrintEdges(false, edges_data);

  // recover result
  Eigen::Matrix3d Rwc = v_rot->estimate().toRotationMatrix();

  for (auto &it : frames) {
    const size_t frame_id = it.first;
    auto &frame = it.second.first;
    // const auto &fixed = it.second.second;
    // if (fixed) {
    //   continue;
    // }
    auto v = static_cast<g2o::VertexLineTranslation *>(
        optimizer.vertex(frame_id + 1));
    Eigen::Vector3d twc = Eigen::Vector3d(0, 0, 0);
    twc[0] = v->estimate();
    Eigen::Vector3d tcw = -Rwc.transpose() * twc;
    // std::cout << "[DEBUG]: frame " << frame_id << std::endl;
    // std::cout << "         twc " << twc.transpose() << std::endl;
    // std::cout << "         tcw " << tcw.transpose() << std::endl;
    frame->setPose(Rwc.transpose(), tcw);
  }

  for (auto &it : mps) {
    const auto &mp = it.second.first;
    const bool &fixed = it.second.second;
    if (fixed) {
      continue;
    }
    const size_t &mp_id = it.first;

    const size_t id = size_t(1 + frame_id_max) + 1 + mp_id;
    auto v = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(id));
    Eigen::Vector3d evec = v->estimate();

    mp->fromEigenVector3d(evec);
  }
}
