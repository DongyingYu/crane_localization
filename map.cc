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

void Map::clear() {
  {
    std::unique_lock<std::mutex> lock(mutex_mappoints_);
    mappoints_.clear();
  }
  {
    std::unique_lock<std::mutex> lock(mutex_frames_);
    frames_.clear();
  }
}

bool Map::trackNewFrame(Frame::Ptr curr_frame) {
  Frame::Ptr last_frame = frames_.back();

  // 1. 特征点匹配
  std::vector<cv::DMatch> good_matches;
  std::vector<cv::Point2f> points1, points2;
  int n_match =
      last_frame->matchWith(curr_frame, good_matches, points1, points2, true);
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
      curr_frame->mappoint_idx_[m.trainIdx] = x3D_idx;
    } else {
      ;
    }
  }
  std::cout << "[INFO]: cnt_3d=" << cnt_3d << " cnt_not_3d=" << n_match - cnt_3d
            << std::endl;
  curr_frame->setPose(last_frame->Tcw_);
  G2oOptimizer::optimizeFramePose(curr_frame, this, 10);

  cv::Mat P1 = last_frame->getProjectionMatrix();
  cv::Mat P2 = curr_frame->getProjectionMatrix();

  // 3. 将剩余配对特征点三角化
  for (const cv::DMatch &m : good_matches) {
    int x3D_idx = last_frame->mappoint_idx_[m.queryIdx];
    if (x3D_idx >= 0) {
      ;
    } else {
      cv::KeyPoint kp1 = last_frame->keypoints_[m.queryIdx];
      cv::KeyPoint kp2 = curr_frame->keypoints_[m.trainIdx];
      cv::Mat x3D;
      triangulate(kp1, kp2, P1, P2, x3D);
    }
  }

  // todo
  // 计算重投影误差，排除外点，之后，重新优化；或者采用类似orbslam2的方式，四次迭代，每次迭代中判断内点和外点
}


bool Map::checkInitialized() {
  int n_mps = 0, n_frames = 0;
  {
    std::unique_lock<std::mutex> lock(mutex_mappoints_);
    n_mps = mappoints_.size();
  }
  {
    std::unique_lock<std::mutex> lock(mutex_frames_);
    n_frames = frames_.size();
  }
  return n_frames >= 2 && n_mps > 0;
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

bool Map::initialize(Frame::Ptr frame1, Frame::Ptr frame2) {
  std::cout << "[INFO]: trying to initialize a map " << std::endl;
  clear();

  assert(frame1->camera_model_ == frame2->camera_model_);
  cv::Mat K = frame1->camera_model_->getNewK();

  // 1. 匹配两帧图像的特征点，计算单应矩阵
  std::vector<cv::DMatch> good_matches;
  std::vector<cv::Point2f> points1, points2;
  frame1->matchWith(frame2, good_matches, points1, points2, true);

  // todo: 增大误差阈值，因为没有矫正畸变参数
  std::vector<uchar> ransac_status;
  cv::Mat H =
      cv::findHomography(points1, points2, ransac_status, cv::RANSAC, 10.0);
  std::cout << "H: " << std::endl << H << std::endl;
  int h_inliers = std::accumulate(ransac_status.begin(), ransac_status.end(), 0,
                                  [](int c1, int c2) { return c1 + c2; });
  std::cout << "[INFO]: Find H inliers: " << h_inliers << std::endl;

  // 尝试利用先验知识（旋转为单位阵），手动分解H
  // K*(R-t*n/d)*K.inv() = H
  // rtnd = K.inv() * H * K
  // tnd = R - rtnd
  if (0) {
    cv::Mat rtnd = K.inv() * H * K;
    std::cout << "rtnd: " << std::endl << rtnd << std::endl;
    cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat tnd = R - rtnd;
    std::cout << "tnd: " << std::endl << tnd << std::endl;

    cv::Mat t(3, 1, CV_64F);
    cv::Mat n(1, 3, CV_64F);
    cv::reduce(tnd, t, 1, cv::REDUCE_SUM);
    cv::reduce(tnd, n, 0, cv::REDUCE_SUM);
    std::cout << "t: " << std::endl << t << std::endl;
    std::cout << "n: " << std::endl << n << std::endl;

    std::vector<uchar> mask = ransac_status;
    std::vector<MapPoint::Ptr> mps_tmp;
    int inliers = checkRtn(R, t, n, K, points1, points2, mask, mps_tmp, false);
    std::cout << "[INFO]: checkRt, inliers: " << inliers << "of " << h_inliers
              << std::endl;
  }

  // // recover pose from E
  // std::vector<uchar> ransac_status_e;
  // cv::Mat F =
  //     cv::findFundamentalMat(points1, points2, ransac_status_e, cv::RANSAC);
  // cv::Mat E = K.t() * F * K;
  // std::cout << "E: " << std::endl << E << std::endl;
  // cv::Mat R_e, t_e;
  // std::vector<uchar> mask = ransac_status_e;
  // int inliers = cv::recoverPose(E, points1, points2, K, R_e, t_e, mask);
  // std::cout << "R_e: " << std::endl << R_e << std::endl;
  // std::cout << "t_e: " << std::endl << t_e << std::endl;

  // 2. 利用单应矩阵计算R和t，挑选出正确的R和t，初始化地图点
  std::vector<cv::Mat> Rs, ts, normals;
  cv::decomposeHomographyMat(H, K, Rs, ts, normals);

  cv::Mat R_h, t_h;
  int inliers = 0;
  std::vector<uchar> mask;
  std::vector<MapPoint::Ptr> mappoints;
  for (int i = 0; i < Rs.size(); ++i) {
    std::vector<uchar> mask_tmp = ransac_status;
    std::vector<MapPoint::Ptr> mps_tmp;
    // 检查R，t，n，统计内点数目
    int inliers_tmp = checkRtn(Rs[i], ts[i], normals[i], K, points1, points2,
                               mask_tmp, mps_tmp);
    if (inliers_tmp > inliers) {
      inliers = inliers_tmp;
      R_h = Rs[i];
      t_h = ts[i];
      mask = mask_tmp;
      mappoints = mps_tmp;
    }
  }

  if (inliers == 0) {
    return false;
  }
  if (inliers < x3D_inliers_threshold_) {
    std::cout << "[WARNING]: Too few mappoint inliers" << std::endl;
  }

  std::cout << "[INFO]: Recover Rt, mappoint inliers: " << inliers << std::endl;
  std::cout << "[INFO]: R_h: " << std::endl << R_h << std::endl;
  std::cout << "[INFO]: t: " << t_h.t() << std::endl;

  // 4. 初始化地图，建立特征点与地图点之间的关联
  frame1->setPose(cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(3, 1, CV_64F));
  frame2->setPose(R_h, t_h);
  std::cout << "[INFO]: twc: " << toString(frame2->getEigenTwc()) << std::endl;
  frames_.emplace_back(frame1);
  frames_.emplace_back(frame2);

  mappoints_ = mappoints;
  int mp_idx = 0;
  for (int i = 0; i < mask.size(); ++i) {
    if (!mask[i]) {
      continue;
    }
    const cv::DMatch &m = good_matches[i];
    int kp_idx1 = m.queryIdx;
    int kp_idx2 = m.trainIdx;
    auto obs1 = std::pair<int, int>(0, kp_idx1);
    auto obs2 = std::pair<int, int>(1, kp_idx2);
    mappoints[mp_idx]->observations_.emplace_back(obs1);
    mappoints[mp_idx]->observations_.emplace_back(obs2);
    frame1->mappoint_idx_[kp_idx1] = mp_idx;
    frame2->mappoint_idx_[kp_idx2] = mp_idx;
    mp_idx++;
  }

  std::cout << "[INFO]: Initialize map finished " << std::endl;
  return true;
}

int Map::checkRtn(const cv::Mat &R, const cv::Mat &t, const cv::Mat &n,
                  const cv::Mat &K, std::vector<cv::Point2f> points1,
                  std::vector<cv::Point2f> points2, std::vector<uchar> &mask,
                  std::vector<MapPoint::Ptr> &mappoints, bool verbose) {

  // 因为相机是俯视地面，法向量必须是大致沿z轴的（z轴分量绝对值最大）
  if (std::fabs(n.at<double>(2, 0)) <= std::fabs(n.at<double>(0, 0)) ||
      std::fabs(n.at<double>(2, 0)) <= std::fabs(n.at<double>(1, 0))) {
    // return 0;
  }

  // 计算地图点，地图点在两个相机坐标系下的z值必须都为正

  // 在相机位置1参考系中，两相机光心
  cv::Mat O1 = cv::Mat::zeros(3, 1, CV_64F);
  cv::Mat O2 = -R.t() * t;

  // Camera 1 Projection Matrix K[I|0]
  cv::Mat P1(3, 4, CV_64F, cv::Scalar(0));
  K.copyTo(P1.rowRange(0, 3).colRange(0, 3));

  // Camera 2 Projection Matrix K[R|t]
  cv::Mat P2(3, 4, CV_64F);
  R.copyTo(P2.rowRange(0, 3).colRange(0, 3));
  t.copyTo(P2.rowRange(0, 3).col(3));
  P2 = K * P2;

  // debug message
  std::vector<double> e1s, e2s, cos_thetas;

  int x3D_cnt = 0;
  mappoints.clear();
  for (int j = 0; j < mask.size(); ++j) {
    // 如果不是计算H矩阵时的内点，则不参与计算
    if (!mask[j]) {
      continue;
    }
    // 空间点在相机位置1参考系中的坐标
    auto pt1 = points1[j];
    auto pt2 = points2[j];
    cv::Mat x3D_C1;

    triangulate(pt1, pt2, P1, P2, x3D_C1);

    // 空间点在相机位置2参考系中的坐标
    cv::Mat x3D_C2 = R * x3D_C1 + t;

    // 判断是否为有效的地图点
    if (!(std::isfinite(x3D_C1.at<double>(0)) &&
          std::isfinite(x3D_C1.at<double>(1)) &&
          std::isfinite(x3D_C1.at<double>(2)) && x3D_C1.at<double>(2) > 0 &&
          x3D_C2.at<double>(2) > 0)) {
      if (verbose) {
        std::cout << "[WARNING]: invalid x3D " << j << ": in C1, " << x3D_C1.t()
                  << " in C2, " << x3D_C2.t() << std::endl;
      }
      continue;
    }

    // 如何使用夹角？
    cv::Mat N1 = x3D_C1 - O1;
    cv::Mat N2 = x3D_C1 - O2;
    double cos_theta = N1.dot(N1) / (cv::norm(N1) * cv::norm(N2));
    cos_thetas.emplace_back(cos_theta);

    // 计算空间点的投影误差，误差平方值应当在允许范围之内
    auto proj_pt1 = project(x3D_C1, K);
    double e1 = (pt1.x - proj_pt1.x) * (pt1.x - proj_pt1.x) +
                (pt1.y - proj_pt1.y) * (pt1.y - proj_pt1.y);
    auto proj_pt2 = project(x3D_C2, K);
    double e2 = (pt2.x - proj_pt2.x) * (pt2.x - proj_pt2.x) +
                (pt2.y - proj_pt2.y) * (pt2.y - proj_pt2.y);

    e1s.emplace_back(e1);
    e2s.emplace_back(e2);

    if (e1 > square_projection_error_threshold_ ||
        e2 > square_projection_error_threshold_) {
      if (verbose) {
        std::cout << "[WARNING]: big reprojection error " << j << ": e1=" << e1
                  << " e2=" << e2 << std::endl;
      }
      continue;
    }

    mappoints.emplace_back(std::make_shared<MapPoint>(
        x3D_C1.at<double>(0), x3D_C1.at<double>(1), x3D_C1.at<double>(2)));
    x3D_cnt++;
    mask[j] = '\1';
  }

  // debug info
  statistic(cos_thetas, "  cos_theta");
  statistic(e1s, "  e1");
  statistic(e2s, "  e2");

  return x3D_cnt;
}

void Map::triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2,
                      const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D) {
  triangulate(kp1.pt, kp2.pt, P1, P2, x3D);
}

void Map::triangulate(const cv::Point2f &pt1, const cv::Point2f &pt2,
                      const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D) {
  // 三角化
  cv::Mat A(4, 4, CV_64F);

  A.row(0) = pt1.x * P1.row(2) - P1.row(0);
  A.row(1) = pt1.y * P1.row(2) - P1.row(1);
  A.row(2) = pt2.x * P2.row(2) - P2.row(0);
  A.row(3) = pt2.y * P2.row(2) - P2.row(1);

  cv::Mat u, w, vt;
  cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
  x3D = vt.row(3).t();
  x3D = x3D.rowRange(0, 3) / x3D.at<double>(3);
}

cv::Point2f Map::project(const cv::Mat &x3D, const cv::Mat K) {
  double fx = K.at<double>(0, 0);
  double fy = K.at<double>(1, 1);
  double cx = K.at<double>(0, 2);
  double cy = K.at<double>(1, 2);
  double X = x3D.at<double>(0);
  double Y = x3D.at<double>(1);
  double Z = x3D.at<double>(2);
  double x = fx * X / Z + cx;
  double y = fy * Y / Z + cy;
  return cv::Point2f(x, y);
}
