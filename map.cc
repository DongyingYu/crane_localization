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
#include <numeric>
#include <utility>
#include "utils.h"

Map::Map() {}

Map::Map(const int &sliding_window_local, const int &sliding_window_global,
         const int &crane_id)
    : sliding_window_local_(sliding_window_local),
      sliding_window_global_(sliding_window_global),
      crane_id_(crane_id) {
  std::cout
      << "[INFO]: \033[33m The value of sliding_window_local_ is:   \033[0m "
      << sliding_window_local_ << std::endl;
  std::cout
      << "[INFO]: \033[33m The value of sliding_window_global_ is: \033[0m "
      << sliding_window_global_ << std::endl;
}

Map::~Map() {}

void Map::clear() {
  {
    std::unique_lock<std::mutex> lock(mutex_mappoints_);
    mappoints_.clear();
  }
  {
    std::unique_lock<std::mutex> lock(mutex_recent_frames_);
    recent_frames_.clear();
  }
  {
    std::unique_lock<std::mutex> lock(mutex_keyframes_);
    keyframes_.clear();
  }
}

int Map::trackNewFrameByKeyFrame(Frame::Ptr curr_frame,
                                 const double &debug_draw) {
  std::cout << "[TRACK]: track new frame " << curr_frame->getFrameId()
            << std::endl;
  Frame::Ptr last_kf = getLastKeyFrame();

  // 1. 特征点匹配
  std::vector<cv::DMatch> good_matches;
  std::vector<cv::Point2f> points1, points2;
  float ave_x;
  int n_match = last_kf->matchWith(curr_frame, good_matches, points1, points2,
                                   ave_x, debug_draw);
  if (n_match < 50) {
    std::cout
        << "[WARNING]: Too less matched keypoint, this may lead to wrong pose: "
        << n_match << std::endl;
  }
  diff_ave_ = ave_x;
  // 对系统CPU占用减少作用并不明显，可见位姿计算部分消耗不大
  // if(abs(ave_x - diff_ave_) < 0.1){
  //   // 静止的情况下，仅获取位姿，将位姿输出
  //   std::cout << "\033[31m [INFO]: The diff of ave_x and diffave : \033[0m"
  //   << abs(ave_x - diff_ave_) << std::endl;
  //   std::cout << "\033[31m [INFO]: The diff_x_ave is small enough, use the
  //   pose of lastframe. \033[0m" << std::endl;
  //   curr_frame->setPose(getLastFrame()->getPose());
  //   // insertRecentFrame(curr_frame);

  //   return 3;
  // }
  // diff_ave_ = ave_x;
  // std::cout << "\033[31m [INFO]: The value of diff_ave_: \033[0m" <<
  // diff_ave_
  // << std::endl;
  if (n_match < 10) return 1;

  // 2. 使用PnP给出当前帧的相机位姿
  int cnt_3d = 0;
  for (const cv::DMatch &m : good_matches) {
    int mp_idx = last_kf->getMappointId(m.queryIdx);
    if (mp_idx >= 0) {
      cnt_3d++;
      curr_frame->setMappointIdx(m.trainIdx, mp_idx);
      auto obs = std::pair<int, int>(curr_frame->getFrameId(), m.trainIdx);
      auto mp = getMapPointById(mp_idx);
      mp->observations_.emplace_back(obs);
    } else {
      ;
    }
  }
  std::cout << "[INFO]: cnt_3d=" << cnt_3d << " cnt_not_3d=" << n_match - cnt_3d
            << std::endl;

  // 设定所需的最小地图点
  if (cnt_3d < 3) {
    std::cout << "[ERROR]: too few matched, trackByKeyFrame failed! " << cnt_3d
              << std::endl;
    // 地图点过少，直接下一帧
    return 1;
  } else {
    // 如果上一帧是关键帧，则返回不对当前帧进行处理，需重新初始化地图，否则继续进行后续(待添加)

    // 对普通帧就是如下的处理情况，对未在地图点中的特征点进行三角化，并进行g2o优化求取位姿。
    // 获取的初始位姿，以上一帧位姿作为初值，是在相机坐标系下，因对两针之间优化求解，故是在像极坐标系下。
    curr_frame->setPose(getLastFrame()->getPose());
    // 优化当前帧curr_frame,及其相关的地图点
    G2oOptimizer::Ptr opt = buildG2oOptForFrame(curr_frame);
    opt->optimizeLinearMotion();

    insertRecentFrame(curr_frame);

    if (cnt_3d == n_match) {
      return 3;
    }
  }

  cv::Mat P1 = last_kf->getProjectionMatrix();
  cv::Mat P2 = curr_frame->getProjectionMatrix();

  // 3. 将剩余配对特征点三角化
  int invalid_3D_cnt = 0;
  for (const cv::DMatch &m : good_matches) {
    int mp_idx = last_kf->getMappointId(m.queryIdx);
    if (mp_idx >= 0) {
      ;
    } else {
      cv::KeyPoint kp1 = last_kf->getUnKeyPoints(m.queryIdx);
      cv::KeyPoint kp2 = curr_frame->getUnKeyPoints(m.trainIdx);
      cv::Mat x3D;

      // 避免出现无穷远点
      cv::Point2f d = kp1.pt - kp2.pt;
      if (d.x * d.x + d.y * d.y < std::numeric_limits<float>::epsilon()) {
        continue;
      }

      triangulate(kp1, kp2, P1, P2, x3D);

      // 判断是否为有效的地图点（貌似失效了，不起作用）
      bool is_finite = std::isfinite(x3D.at<double>(0)) &&
                       std::isfinite(x3D.at<double>(1)) &&
                       std::isfinite(x3D.at<double>(2));
      bool is_depth_valid =
          last_kf->checkDepthValid(x3D) && curr_frame->checkDepthValid(x3D);

      if (!(is_finite && is_depth_valid)) {
        std::cout << "[WARNING]: invalid x3D " << x3D.t() << std::endl;
        invalid_3D_cnt++;
        continue;
      }

      cv::Point2f proj_pt1 = last_kf->project(x3D);
      cv::Point2f proj_pt2 = curr_frame->project(x3D);
      float e1 = squareUvError(proj_pt1 - kp1.pt);
      float e2 = squareUvError(proj_pt2 - kp2.pt);
      if (e1 > 3 * square_projection_error_threshold_ ||
          e2 > 3 * square_projection_error_threshold_) {
        std::cout << "[WARNING]: big reprojection error "
                  << ": e1=" << e1 << " e2=" << e2 << std::endl;
        continue;
      }
      if (std::abs(x3D.at<double>(0)) > 1e10) {
        throw std::runtime_error("error");
      }
      auto mp = std::make_shared<MapPoint>(x3D.at<double>(0), x3D.at<double>(1),
                                           x3D.at<double>(2));
      auto obs1 = std::pair<int, int>(last_kf->getFrameId(), m.queryIdx);
      auto obs2 = std::pair<int, int>(curr_frame->getFrameId(), m.trainIdx);
      mp->observations_.emplace_back(obs1);
      mp->observations_.emplace_back(obs2);
      insertMapPoint(mp);
      last_kf->setMappointIdx(m.queryIdx, mp->getId());
      curr_frame->setMappointIdx(m.trainIdx, mp->getId());
    }
  }
  // 再次优化当前帧curr_frame,及其相关的地图点
  G2oOptimizer::Ptr opt = buildG2oOptForFrame(curr_frame);
  opt->optimizeLinearMotion();

  // 计算重投影误差，排除外点，之后，重新优化；或者采用类似orbslam2的方式，四次迭代，每次迭代中判断内点和外点
  return 3;
}

bool Map::checkInitialized() { return is_initialized_; }

G2oOptimizer::Ptr Map::buildG2oOptForFrame(const Frame::Ptr frame) {
  std::map<size_t, std::pair<Frame::Ptr, bool>> frames_data;
  std::map<size_t, std::pair<MapPoint::Ptr, bool>> mps_data;
  std::map<size_t, std::vector<std::pair<size_t, size_t>>> obs_data;

  // pose固定的frame
  {
    std::unique_lock<std::mutex> lock(mutex_keyframes_);
    std::map<size_t, Frame::Ptr>::reverse_iterator rit;
    for (rit = keyframes_.rbegin(); rit != keyframes_.rend(); ++rit) {
      if (frames_data.size() > sliding_window_local_) {
        break;
      }
      frames_data[rit->first] = std::pair<Frame::Ptr, bool>(rit->second, true);
    }
  }

  // pose待优化的frame
  frames_data[frame->getFrameId()] = std::pair<Frame::Ptr, bool>(frame, false);

  // 待优化的mappoint
  std::vector<int> mp_indixes = frame->getMappointId();
  {
    std::unique_lock<std::mutex> lock(mutex_mappoints_);
    for (const int &mp_idx : mp_indixes) {
      if (mp_idx < 0) {
        continue;
      }
      auto z_value = mappoints_[mp_idx].get()->getPointValue(3);
      if (z_value > 10 || z_value < 0) continue;
      const auto &mp = mappoints_[mp_idx];
      mps_data[mp_idx] = std::pair<MapPoint::Ptr, bool>(mp, false);
    }
  }

  // 以上frame和mappoint之间所有的observation
  for (const auto &it : mps_data) {
    const size_t &mp_idx = it.first;
    const auto &mp = it.second.first;
    const auto &observation_tmp = mp->getObservation();

    std::vector<std::pair<size_t, size_t>> observation;
    for (const auto &obs : observation_tmp) {
      auto frame_id = obs.first;
      if (frames_data.find(frame_id) != frames_data.end()) {
        observation.emplace_back(obs);
      }
    }
    obs_data[mp_idx] = observation;
    // 若地图点被两个以上关键帧观察到，不对地图点位置做出调整
    if (observation.size() > 2) mps_data[mp_idx].second = true;
  }

  return std::make_shared<G2oOptimizer>(frames_data, mps_data, obs_data);
}

G2oOptimizer::Ptr Map::buildG2oOptKeyFrameBa() {
  std::map<size_t, std::pair<Frame::Ptr, bool>> frames_data;
  std::map<size_t, std::pair<MapPoint::Ptr, bool>> mps_data;
  std::map<size_t, std::vector<std::pair<size_t, size_t>>> obs_data;
  // frames
  {
    std::unique_lock<std::mutex> lock(mutex_keyframes_);
    std::map<size_t, Frame::Ptr>::reverse_iterator rit;
    for (rit = keyframes_.rbegin(); rit != keyframes_.rend(); ++rit) {
      if (frames_data.size() > sliding_window_global_) {
        break;
      }
      frames_data[rit->first] = std::pair<Frame::Ptr, bool>(rit->second, false);
    }
  }

  // 将id最小的一帧，置为fixed
  frames_data.begin()->second.second = true;

  // mappoints
  for (auto &it : frames_data) {
    const auto &frame_id = it.first;
    auto &frame = it.second.first;
    // 获取图像帧特征点对应3D空间点id
    std::vector<int> mp_indixes = frame->getMappointId();
    {
      // 获取id存储在set中
      std::unique_lock<std::mutex> lock(mutex_mappoints_);
      for (const int &mp_idx : mp_indixes) {
        // mp_idx小于0为无效点
        if (mp_idx < 0) {
          continue;
        }
        auto z_value = mappoints_[mp_idx].get()->getPointValue(3);
        if (z_value > 10 || z_value < 0) continue;
        const auto &mp = mappoints_[mp_idx];
        mps_data[mp_idx] = std::pair<MapPoint::Ptr, bool>(mp, false);
      }
    }
  }

  // 以上frame和mappoint之间所有的observation
  for (const auto &it : mps_data) {
    const auto &mp = it.second.first;
    const auto &observation_tmp = mp->getObservation();
    std::vector<std::pair<size_t, size_t>> observation;
    for (const auto &obs : observation_tmp) {
      if (frames_data.find(obs.first) != frames_data.end()) {
        observation.emplace_back(obs);
      }
    }
    // 如果该地图点，有超过两帧观测到，则加入优化
    if (observation.size() >= 2) {
      obs_data[mp->getId()] = observation;
    }
  }

  return std::make_shared<G2oOptimizer>(frames_data, mps_data, obs_data);
}

void Map::clearRecentFrames() {
  std::unique_lock<std::mutex> lock(mutex_recent_frames_);
  recent_frames_.clear();
}

void Map::debugPrintMap() {
  double scale = getScale();
  {
    std::unique_lock<std::mutex> lock(mutex_recent_frames_);
    std::cout << "[INFO]: recent_frames_.size()=" << recent_frames_.size()
              << std::endl;
  }
  // 输出地图点的均值
  {
    std::unique_lock<std::mutex> lock(mutex_mappoints_);
    Eigen::Vector3d mp_ave = Eigen::Vector3d::Zero();
    for (const auto &it : mappoints_) {
      mp_ave += it.second->toEigenVector3d();
    }
    mp_ave = mp_ave / mappoints_.size();
    std::cout << "[INFO]: MapPoint size " << mappoints_.size()
              << " scaled mean: " << toString(mp_ave * scale) << std::endl;
    std::cout << "[INFO]: scaled mean of kf_mp: "
              << toString(ave_kf_mp_ * scale) << std::endl;
  }
  // 输出关键帧信息，最多五帧
  {
    std::unique_lock<std::mutex> lock_frames(mutex_keyframes_);
    std::cout << "[INFO]: Num KeyFrames=" << keyframes_.size() << std::endl;
    int cnt = 0;
    std::map<size_t, Frame::Ptr>::reverse_iterator rit;
    for (rit = keyframes_.rbegin(); rit != keyframes_.rend(); ++rit) {
      const size_t &frame_id = rit->first;
      Frame::Ptr &frame = rit->second;
      if (cnt == 0) {
        // 输出公用旋转量
        Eigen::Quaterniond q(frame->getEigenRot());
        std::cout << "[INFO]: Shared Rotation " << toString(q) << std::endl;
      } else if (cnt >= 5) {
        break;
      }
      cnt++;
      // 每一帧的平移量
      Eigen::Vector3d twc = frame->getEigenTransWc();
      std::cout << "[INFO]: Frame " << frame_id
                << "          scaled twc = " << toString(twc * scale)
                << std::endl;
    }
  }
  // 最近一帧的信息
  Frame::Ptr last_frame = getLastFrame();
  Eigen::Vector3d twc = last_frame->getEigenTransWc();
  std::cout << "[INFO]: Frame " << last_frame->getFrameId()
            << " : scaled twc = " << toString(twc * scale) << std::endl
            << std::endl;
}

const double Map::kCraneHeight = 9.0;

bool Map::initialize(const Frame::Ptr &frame1, const Frame::Ptr &frame2,
                     const double &debug_draw) {
  std::cout << "[INFO]: trying to initialize a map " << std::endl;
  // clear();

  assert(frame1->camera_model_ == frame2->camera_model_);
  cv::Mat K = frame1->camera_model_->getNewK();

  // 1. 匹配两帧图像的特征点，计算单应矩阵
  std::vector<cv::DMatch> good_matches;
  std::vector<cv::Point2f> points1, points2;
  float ave_x;
  frame1->matchWith(frame2, good_matches, points1, points2, ave_x, debug_draw);

  // todo: 增大误差阈值，因为没有矫正畸变参数
  std::vector<uchar> ransac_status;
  if (points1.size() == 0 || points2.size() == 0) return false;
  cv::Mat H = cv::findHomography(points1, points2, ransac_status, cv::RANSAC);
  std::cout << "H: " << std::endl << H << std::endl;
  int h_inliers = std::accumulate(ransac_status.begin(), ransac_status.end(), 0,
                                  [](int c1, int c2) { return c1 + c2; });
  std::cout << "[INFO]: Find H inliers: " << h_inliers << std::endl;
  if (h_inliers == 0) return false;

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
  // t_h = t_h / std::sqrt(t_h.at<double>(0) * t_h.at<double>(0) +
  //                      t_h.at<double>(1) * t_h.at<double>(1) +
  //                      t_h.at<double>(2) * t_h.at<double>(2));
  std::cout << "[INFO]: Recover Rt, mappoint inliers: " << inliers << std::endl;
  std::cout << "[INFO]: R_h: " << std::endl << R_h << std::endl;
  std::cout << "[INFO]: t: " << t_h.t() << std::endl;

  // 3. 初始化地图，建立特征点与地图点之间的关联
  // setPose()最初初始化时用到，设置第一帧图像的初始化位置，用默认值
  // frame1->setPose(cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(3, 1, CV_64F));
  // 获取相机位姿下的旋转、平移数据
  cv::Mat Tcw = frame1->getPose();
  cv::Mat R1, t1;
  Tcw.rowRange(0, 3).colRange(0, 3).copyTo(R1);
  Tcw.rowRange(0, 3).col(3).copyTo(t1);
  frame2->setPose(R_h * R1, R_h * t1 + t_h);

  // 这里仅是输出对平移量进行观察，获取第二帧图像在世界坐标系下的平移，
  // 即：-旋转矩阵的逆*平移量
  std::cout << "[INFO]: twc: " << toString(frame2->getEigenTransWc())
            << std::endl;
  insertRecentFrame(frame1);
  insertRecentFrame(frame2);

  frame1->releaseImage();
  insertKeyFrame(frame1);
  insertKeyFrame(frame2);

  int mp_idx = 0;
  for (int i = 0; i < mask.size(); ++i) {
    if (!mask[i]) {
      continue;
    }
    const cv::DMatch &m = good_matches[i];
    int kp_idx1 = m.queryIdx;
    int kp_idx2 = m.trainIdx;
    auto obs1 = std::pair<int, int>(frame1->getFrameId(), kp_idx1);
    auto obs2 = std::pair<int, int>(frame2->getFrameId(), kp_idx2);
    auto &mp = mappoints[mp_idx];
    mp->observations_.emplace_back(obs1);
    mp->observations_.emplace_back(obs2);
    frame1->setMappointIdx(kp_idx1, mp->getId());
    frame2->setMappointIdx(kp_idx2, mp->getId());
    mp_idx++;
  }
  for (const auto &mp : mappoints) {
    insertMapPoint(mp);
  }

  ofstream keyframe_position("./keyframe_position.csv", ios::app);
  keyframe_position.setf(ios::fixed, ios::floatfield);
  keyframe_position.precision(6);
  keyframe_position << frame1->getEigenTransWc()[0] << std::endl
                    << frame2->getEigenTransWc()[0] << std::endl;
  keyframe_position.close();

  saveKeyframeposition();

  // std::cout << "[Debug]: test one ... " << std::endl;
  // 4. 利用天车高度的先验，计算尺度
  ave_kf_mp_ = getAveMapPoint();
  Eigen::Vector3d ave_kf_mp = ave_kf_mp_;
  // 排除x方向上的影响
  ave_kf_mp[0] = 0;
  scale_ = kCraneHeight / ave_kf_mp.norm();
  std::cout << "[INFO]: The scale value is :         " << scale_ << std::endl;
  // std::cout << "[Debug]: test two ... " << std::endl;
  is_initialized_ = true;

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

  // cv::Mat t_normalize = t / std::sqrt(t.at<double>(0) * t.at<double>(0) +
  //                                    t.at<double>(1) * t.at<double>(1) +
  //                                    t.at<double>(2) * t.at<double>(2));
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
    mask[j] = '\0';
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
    double e1 = squareUvError(pt1 - proj_pt1);
    auto proj_pt2 = project(x3D_C2, K);
    double e2 = squareUvError(pt2 - proj_pt2);

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

float Map::squareUvError(const cv::Point2f &uv_error) {
  return uv_error.x * uv_error.x + uv_error.y * uv_error.y;
}

void Map::insertMapPoint(const MapPoint::Ptr &mp) {
  std::unique_lock<std::mutex> lock(mutex_mappoints_);
  mappoints_[mp->getId()] = mp;
}

size_t Map::removeMapPointById(const size_t &mp_idx) {
  std::unique_lock<std::mutex> lock(mutex_mappoints_);
  return mappoints_.erase(mp_idx);
}

MapPoint::Ptr Map::getMapPointById(const int &mp_idx) {
  std::unique_lock<std::mutex> lock(mutex_mappoints_);
  auto it = mappoints_.find(mp_idx);
  if (it != mappoints_.end()) {
    return it->second;
  } else {
    std::cout << "[WARNING]: " << mp_idx << " not exists!" << std::endl;
    throw std::runtime_error("error");
    return nullptr;
  }
}

std::vector<MapPoint::Ptr> Map::getMapPoints() {
  std::vector<MapPoint::Ptr> mappoints;
  std::unique_lock<std::mutex> lock(mutex_mappoints_);
  for (auto &it : mappoints_) {
    mappoints.emplace_back(it.second);
  }
  return mappoints;
}

Eigen::Vector3d Map::getAveMapPoint() {
  // 仅对关键帧3D点用来计算scale
  std::unique_lock<std::mutex> lock_keyframes(mutex_keyframes_);
  std::set<int> mappoints_index;
  for (auto &it : keyframes_) {
    const auto &frame_id = it.first;
    auto &frame = it.second;
    std::vector<int> mp_indixes = frame->getMappointId();
    {
      for (const int &mp_idx : mp_indixes) {
        // mp_idx小于0为无效点
        if (mp_idx < 0) {
          continue;
        }
        mappoints_index.insert(mp_idx);
      }
    }
  }

  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
      kf_mps;
  std::unique_lock<std::mutex> lock_mappoints(mutex_mappoints_);
  Eigen::Vector3d ret = Eigen::Vector3d::Zero();
  for (const int &mps_id : mappoints_index) {
    auto z_value = mappoints_[mps_id].get()->getPointValue(3);
    if (z_value > 9 || z_value < 0.2) continue;
    auto mp = mappoints_[mps_id];
    kf_mps.emplace_back(mp->toEigenVector3d());
    ret += mp->toEigenVector3d();
  }
  statistic(kf_mps, "ave mp");
  ret = ret / kf_mps.size();
  return ret;

  // std::unique_lock<std::mutex> lock(mutex_mappoints_);
  // Eigen::Vector3d ret = Eigen::Vector3d::Zero();
  // for (const auto &it : mappoints_) {
  //   ret += it.second->toEigenVector3d();
  // }
  // ret = ret / mappoints_.size();
  // return ret;
}

double Map::getScale() {
  std::unique_lock<std::mutex> lock(mutex_scale_);
  return scale_;
}

void Map::setScale(const double &scale) {
  std::unique_lock<std::mutex> lock(mutex_scale_);
  scale_ = scale;
}

void Map::insertRecentFrame(const Frame::Ptr &frame) {
  std::unique_lock<std::mutex> lock(mutex_recent_frames_);
  recent_frames_[frame->getFrameId()] = frame;
  while (recent_frames_.size() > max_recent_frames_) {
    recent_frames_.erase(recent_frames_.begin());
  }
}

Frame::Ptr Map::getLastFrame() {
  std::unique_lock<std::mutex> lock(mutex_recent_frames_);
  if (recent_frames_.empty()) {
    return nullptr;
  }
  return recent_frames_.rbegin()->second;
}

void Map::insertKeyFrame(const Frame::Ptr &frame) {
  std::unique_lock<std::mutex> lock(mutex_keyframes_);
  keyframes_[frame->getFrameId()] = frame;
}

Frame::Ptr Map::getLastKeyFrame() {
  std::unique_lock<std::mutex> lock(mutex_keyframes_);
  if (keyframes_.empty()) {
    return nullptr;
  }
  return keyframes_.rbegin()->second;
}

bool Map::checkIsNewKeyFrame(Frame::Ptr &frame) {
  Frame::Ptr last_kf = getLastKeyFrame();
  // 如果当前帧与关键帧之间再x方向上的像素差值大于20，必须插入关键帧
  // if (diff_ave_ > 20) {
  //   return true;
  // }

  // 当前帧与上一帧相隔时间太短，则不宜为关键帧
  // 若丢了较多帧，邻近几帧移动距离会足够可插入关键帧
  if (std::abs(frame->getFrameId() - last_kf->getFrameId() < 2)) {
    return false;
  }

  // 当前帧与上一个关键帧之间的平移量过小，也不宜关键帧
  Eigen::Vector3d trans = frame->getEigenTransWc();
  Eigen::Vector3d last_kf_trans = last_kf->getEigenTransWc();
  if (std::abs((trans[0] - last_kf_trans[0])) < 0.02) {
    return false;
  }
  // double scale = getScale();
  // if (std::abs((trans - last_kf_trans).norm() * scale) < 0.2) {
  //   return false;
  // }
  return true;
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

int Map::getKeyframesSize() {
  std::unique_lock<std::mutex> lock(mutex_keyframes_);
  return keyframes_.size();
}

std::map<size_t, Frame::Ptr> Map::getKeyframes() const { return keyframes_; }

void Map::setOffset(const double &offset) { offset_ = offset; }

double Map::getOffset() { return offset_; }

void Map::calculateOffset() {
  int cnt = 0;
  double sum = 0.0;
  std::vector<double> offset_vec;
  for (auto &v : keyframes_) {
    if (v.second->getFlag()) {
      cnt++;
      double position_abs = v.second->getAbsPosition();

      Eigen::Vector3d twc = v.second->getEigenTransWc();
      // double position_estimate = twc[0] * scale_;
      double position_estimate;
      if (crane_id_ == 4) {
        position_estimate = -twc[0] * 13.18 + 2.72;
      } else if (crane_id_ == 3) {
        position_estimate = -twc[0] * 6.75 + 66.13;
      } else if (crane_id_ == 2) {
        position_estimate = -twc[0] * 12.53 + 3.53;
      } else {
        position_estimate = -twc[0] * 12.53 + 3.53;
      }
      std::cout << "[INFO]: The value of twc[0]:  " << twc[0] << std::endl;
      std::cout << "[INFO]: The value of position_estimate:  "
                << position_estimate << std::endl;

      offset_vec.emplace_back(position_abs - position_estimate);
      std::cout << "[INFO]: The value of position_abs:  " << position_abs
                << std::endl;
    }
    statistic(offset_vec, " offset");
    double stddev;
    calAveStddev(offset_vec, offset_, stddev);
  }
}

void Map::setInitializeStatus(const bool &status) { is_initialized_ = status; }

void Map::releaseLastKeyframeimg() {
  std::unique_lock<std::mutex> lock(mutex_keyframes_);
  if (keyframes_.empty()) {
    return;
  }
  Frame::Ptr last_keyframe = keyframes_.rbegin()->second;
  last_keyframe->releaseImage();
}

void Map::saveKeyframeposition() {
  std::unique_lock<std::mutex> lock(mutex_keyframes_);
  if (keyframes_.empty()) {
    return;
  }
  ofstream keyframe_position_update("./keyframe_position_update.csv", ios::app);
  for (auto &it : keyframes_) {
    keyframe_position_update.setf(ios::fixed, ios::floatfield);
    keyframe_position_update.precision(6);
    keyframe_position_update << it.first << ","
                             << it.second->getEigenTransWc()[0] << std::endl;
  }
  keyframe_position_update << std::endl;
  keyframe_position_update.close();
}
