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
#include <cmath>  // for M_PI
#include "utils.h"

// 去注释，可查看优化调试信息
// #define G2O_OPT_VERBOSE

#ifdef G2O_OPT_VERBOSE
#define VERBOSE__ true
#else
#define VERBOSE__ false
#endif

/**
 * @brief 统计输出边的卡方信息
 *
 * @tparam EdgeType g2o边的类型
 * @param[in out] compute_error 是否需要调用edge的computeError函数
 * @param[in out] edges_data 按frame_id索引的所有边
 */
template <typename EdgeType>
static void debugPrintEdges(
    bool compute_error, std::map<size_t, std::vector<EdgeType *>> edges_data) {
#ifdef G2O_OPT_VERBOSE
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
#endif  // G2O_OPT_VERBOSE
}

void G2oOptimizer::optimize(const int &n_iteration) {
  // create optimizer
  g2o::SparseOptimizer optimizer;
  optimizer.setVerbose(VERBOSE__);

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
  for (auto &it : frames_data_) {
    size_t frame_id = it.first;
    Frame::Ptr frame = it.second.first;
    bool fixed = it.second.second;
    frame_id_max = std::max(frame_id_max, int(frame_id));

    edges_data[frame_id] = std::vector<EdgeType *>();

    auto v = new g2o::VertexSE3Expmap();
    v->setId(frame_id);
    v->setFixed(fixed);
    v->setEstimate(g2o::SE3Quat(frame->getEigenRot(), frame->getEigenTrans()));
    optimizer.addVertex(v);
  }

  // mappoint vertex and e
  for (auto &it : mps_data_) {
    size_t mp_id = it.first;
    MapPoint::Ptr mp = it.second.first;
    if (!mp) {
      std::cout << "[WARNING]: empty mappoint " << mp_id << std::endl;
    }
    bool fixed = it.second.second;
    int id = (frame_id_max + 1) + mp_id;

    auto v = new g2o::VertexSBAPointXYZ();
    v->setId(id);
    v->setFixed(fixed);
    v->setEstimate(mp->toEigenVector3d());
    v->setMarginalized(true);
    optimizer.addVertex(v);

    // 边
    for (const auto &obs : observations_data_[mp_id]) {
      size_t frame_id = obs.first;
      size_t kp_id = obs.second;
      Frame::Ptr &frame = frames_data_[frame_id].first;
      const cv::KeyPoint &kp = frame->getUnKeyPoints(kp_id);
      Eigen::Vector2d uv(kp.pt.x, kp.pt.y);

      if (!optimizer.vertex(frame_id)) {
        continue;
      }

      using GVO = g2o::OptimizableGraph::Vertex;
      auto e = new g2o::EdgeSE3ProjectXYZ();
      e->setVertex(0, dynamic_cast<GVO *>(optimizer.vertex(id)));
      e->setVertex(1, dynamic_cast<GVO *>(optimizer.vertex(frame_id)));
      // 决定了边类中computeError()函数和linearizeOplus()函数中的_measurement的值
      e->setMeasurement(uv);
      // 信息矩阵的设定
      e->setInformation(Eigen::Matrix2d::Identity());

      std::vector<double> intr_vec = frame->camera_model_->getNewIntrinsicVec();
      e->fx = intr_vec[0];
      e->fy = intr_vec[1];
      e->cx = intr_vec[2];
      e->cy = intr_vec[3];

      edges_data[frame_id].emplace_back(e);
      optimizer.addEdge(e);
    }
  }

  // optimize
  // 设置优化次数
  for (size_t i = 0; i < 2; i++) {
    optimizer.initializeOptimization();
    debugPrintEdges(true, edges_data);
    optimizer.optimize(n_iteration);
    debugPrintEdges(false, edges_data);

    for (auto &it : edges_data) {
      auto &edges = it.second;
      for (auto &e : edges) {
        e->computeError();
        double chi2 = e->chi2();
        if (chi2 > 20) {
          e->setLevel(1);
        }
      }
    }
  }
  // optimize result
  // camera pose
  for (auto &it : frames_data_) {
    size_t frame_id = it.first;
    Frame::Ptr frame = it.second.first;
    bool fixed = it.second.second;
    if (fixed) {
      continue;
    }
    auto v = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(frame_id));
    Eigen::Matrix4d emat = v->estimate().to_homogeneous_matrix();
    frame->setPose(emat);
  }
  // mappoints
  for (auto &it : mps_data_) {
    size_t mp_id = it.first;
    MapPoint::Ptr mp = it.second.first;
    bool fixed = it.second.second;
    int id = (frame_id_max + 1) + mp_id;

    auto v = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(id));
    Eigen::Vector3d evec = v->estimate();
    mp->fromEigenVector3d(evec);
  }
}

/**
 * @brief calculate the minimal rotation, which can convert src to dst.
 * @return g2o::Quaternion
 */
Eigen::Quaterniond G2oOptimizer::calMinRotation(const Eigen::Vector3d &src,
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

void G2oOptimizer::optimizeLinearMotion(const int &n_iteration) {
  // optimizer
  g2o::SparseOptimizer optimizer;
  optimizer.setVerbose(VERBOSE__);
  using BlockSolverType = g2o::BlockSolverX;
  using LinearSolverType =
      g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;
  auto solver = new g2o::OptimizationAlgorithmLevenberg(
      g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
  optimizer.setAlgorithm(solver);

  // 顶点类型1：公共的旋转量
  auto v_rot = new g2o::VertexSO3Expmap();
  v_rot->setId(0);
  Frame::Ptr &frame = frames_data_.begin()->second.first;
  const bool &fixed = frames_data_.begin()->second.second;
  auto q = Eigen::Quaterniond(frame->getEigenRotWc());
  // std::cout << "[DEBUG]: shared rotation: " << toString(q) << std::endl;
  // v_rot->setFixed(fixed);
  v_rot->setEstimate(q);
  optimizer.addVertex(v_rot);

  using EdgeType = g2o::EdgeLinearMotion;
  std::map<size_t, std::vector<EdgeType *>> edges_data;

  // 顶点类型2：每一帧的平移量（设定为x方向）
  int frame_id_max = -1;
  for (const auto &it : frames_data_) {
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
  for (const auto &it : mps_data_) {
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

    // 不对先前地图点位置及关键帧位姿做调整
    // 边 对每个地图点查询frame_id、kp_id
    if (1) {
      for (const auto &obs : observations_data_[mp_id]) {
        size_t frame_id = obs.first;
        size_t kp_id = obs.second;
        Frame::Ptr &frame = frames_data_[frame_id].first;
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
  //  mp_id frame_id
  std::map<size_t, size_t> e_index;
  // optimize
  for (size_t i = 0; i < 2; i++) {
    optimizer.initializeOptimization();
    debugPrintEdges(true, edges_data);
    optimizer.optimize(n_iteration);
    debugPrintEdges(false, edges_data);

    for (auto &it : edges_data) {
      auto &edges = it.second;
      for (auto &e : edges) {
        e->computeError();
        double chi2 = e->chi2();
        if (i == 0) {
          if (chi2 > 20) {
            e->setLevel(1);
            // 进行了两轮优化，这里需要设置只使用第一次优化中获取值，需做改动
            // 这里frame_id、mp_id的计算关系首先要里理清楚
            size_t frame_id = e->vertices()[1]->id() - 1;
            // 获取地图点id值
            size_t mp_id = e->vertices()[2]->id() - frame_id_max - 2;
            e_index[mp_id] = frame_id;
          }
        }
      }
    }
  }

  // recover result
  Eigen::Matrix3d Rwc = v_rot->estimate().toRotationMatrix();

  for (auto &it : frames_data_) {
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
  bool erase_flag;
  if (e_index.empty())
    erase_flag = false;
  else
    erase_flag = true;

  for (auto &it : mps_data_) {
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
    // 删除无效的共视地图点
    if (erase_flag) {
      mp->eraseObservation(e_index);
    }
    erase_flag = false;
  }
}

Eigen::Vector3d G2oOptimizer::calAveMapPoint() {
  Eigen::Vector3d ave_kf_mp = Eigen::Vector3d::Zero();
  // 标准的定义容器方法,Eigen管理内存和C++11中的方法不一样，需要单独强调元素的内存分配和管理
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
      kf_mps;
  for (auto &it : mps_data_) {
    auto &mp = it.second.first;
    if (mp->toEigenVector3d()[2] > 9 || mp->toEigenVector3d()[2] < 0.2)
      continue;
    kf_mps.emplace_back(mp->toEigenVector3d());
    ave_kf_mp += mp->toEigenVector3d();
  }
  // 有点问题
  double ave_z = statistic(kf_mps, "ave mp");
  ave_kf_mp /= kf_mps.size();
  return ave_kf_mp;
}