/**
 * @file frame.cc
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2021-02-08
 *
 * @copyright Copyright (c) 2021
 *
 */
#include "frame.h"
#include "utils.h"

Frame::Frame(const cv::Mat &img) : img_(img.clone()) { init(); }

Frame::Frame(const cv::Mat &img, const CameraModel::Ptr &camera_model)
    : img_(img.clone()), camera_model_(camera_model) {
  init();
}

void Frame::init() {
  assert(!img_.empty());

  // 1. 特征点计算
  // 1.1 计算Oriented FAST角点
  std::vector<cv::KeyPoint> keypoints;
  detector_->detect(img_, keypoints);

  // 1.2去除部分不可用的特征点， (注：图片可能被转置了)
  // ①左上角和右下角的字幕的干扰
  const double lf = (6.5 / 21.0);
  const double sf = (1.5 / 12.0);
  double x_s = img_.cols * (img_.cols > img_.rows ? lf : sf);
  double y_s = img_.rows * (img_.cols > img_.rows ? sf : lf);

  auto isSubtitle = [&](const cv::KeyPoint &kp) {
    return (kp.pt.x < x_s && kp.pt.y < y_s) ||
           (kp.pt.x > img_.cols - x_s && kp.pt.y > img_.rows - y_s);
  };

  // ②因畸变导致长边两端的区域不可用(各约1/8)
  const double outer_boarder_factor = 1.0 / 8;
  auto isOuterBoarder = [&](const cv::KeyPoint &kp) {
    if (img_.cols > img_.rows) {
      return (kp.pt.x < img_.cols * outer_boarder_factor) ||
             (kp.pt.x > img_.cols * (1 - outer_boarder_factor));
    } else {
      return (kp.pt.y < img_.rows * outer_boarder_factor) ||
             (kp.pt.y > img_.rows * (1 - outer_boarder_factor));
    }
  };

  for (const cv::KeyPoint &kp : keypoints) {
    if (isSubtitle(kp) || isOuterBoarder(kp)) {
      continue;
    } else {
      keypoints_.emplace_back(kp);
    }
  }

  // 2. 去畸变
  if (camera_model_) {
    camera_model_->undistortKeyPoint(keypoints_, un_keypoints_);
    camera_model_->undistortImage(img_, un_img_);
  } else {
    un_keypoints_ = keypoints;
    un_img_ = img_.clone();
  }

  // 3. 描述子计算（BRIEF）
  extrator_->compute(img_, keypoints_, descriptors_);
  std::cout << "[INFO]: keypoints_.size()=" << keypoints_.size();
  std::cout << " descriptors_.size()=" << descriptors_.size() << std::endl;

  // 4. 其他初始化
  mappoint_idx_ = std::vector<int>(keypoints_.size(), -1);
  frame_id_ = Frame::total_frame_cnt_++;
}

int Frame::matchWith(const Frame::Ptr frame,
                     std::vector<cv::DMatch> &good_matches,
                     std::vector<cv::Point2f> &points1,
                     std::vector<cv::Point2f> &points2,
                     const bool &debug_draw) {

  // 匹配特征点
  std::vector<cv::DMatch> all_matches;
  matcher_->match(descriptors_, frame->descriptors_, all_matches);

  // 统计匹配距离（Hamming）的最大值和最小值
  double dmin = 1;
  double dmax = 0;
  for (const cv::DMatch &m : all_matches) {
    dmin = m.distance < dmin ? m.distance : dmin;
    dmax = m.distance > dmax ? m.distance : dmax;
  }
  std::cout << "[INFO]: Descriptor distance max: " << dmax << " min: " << dmin
            << std::endl;

  // 根据经验，筛选匹配
  std::vector<cv::DMatch> tmp_matches;
  std::vector<cv::Point2f> pts1, pts2, pts_diff;
  for (const cv::DMatch &m : all_matches) {
    if (m.distance <= dmax * 0.6) {
      tmp_matches.emplace_back(m);
      cv::Point2f pt1 = un_keypoints_[m.queryIdx].pt;
      cv::Point2f pt2 = frame->un_keypoints_[m.trainIdx].pt;
      pts1.emplace_back(pt1);
      pts2.emplace_back(pt2);
      pts_diff.emplace_back(pt1 - pt2);
    }
  }
  std::cout << "[INFO]: selected " << tmp_matches.size() << " matches from "
            << all_matches.size() << " by match distance." << std::endl;

  // 根据运动约束，检测配对点是否合理
  cv::Point2f ave, stddev;
  calAveStddev(pts_diff, ave, stddev, true);
  std::cout << "[INFO]: Point uv diff, ave " << ave << " stddev " << stddev
            << std::endl;

  std::vector<cv::DMatch> better_matches;
  pts1.clear();
  pts2.clear();
  for (int i = 0; i < int(tmp_matches.size()); ++i) {
    cv::Point2f abs_diff =
        cv::Point2d(std::abs(pts_diff[i].x), std::abs(pts_diff[i].y));
    cv::Point2f ddiff = abs_diff - ave;
    if (std::abs(ddiff.y) > 1 + 3 * stddev.y ||
        std::abs(ddiff.x) > 3 + 3 * stddev.x) {
      // if (std::abs(diff.y) > stddev.y) {
      std::cout << "[INFO]: outlier, ddiff.x=" << ddiff.x
                << " ddiff.y=" << ddiff.y << std::endl;
      continue;
    }
    auto m = tmp_matches[i];
    cv::Point2f pt1 = un_keypoints_[m.queryIdx].pt;
    cv::Point2f pt2 = frame->un_keypoints_[m.trainIdx].pt;
    better_matches.emplace_back(m);
    pts1.emplace_back(pt1);
    pts2.emplace_back(pt2);
  }

  std::cout << "[INFO]: selected " << better_matches.size() << " matches from "
            << tmp_matches.size() << " by stddev " << std::endl;

  pts_diff.clear();
  for (int i = 0; i < int(pts1.size()); ++i) {
    pts_diff.emplace_back(pts1[i] - pts2[i]);
  }

  calAveStddev(pts_diff, ave, stddev, true);
  std::cout << "[INFO]: Point uv diff, ave " << ave << " stddev " << stddev
            << std::endl;

  // 优先使用靠近图像中间的特征点（越往边缘，畸变越严重）
  good_matches.clear();
  points1.clear();
  points2.clear();
  for (const auto &m : better_matches) {
    auto kp1 = un_keypoints_[m.queryIdx];
    auto kp2 = frame->un_keypoints_[m.trainIdx];
    // 0.5表示认为整个图像都可以，即无任何筛选
    double half_center_factor = 0.5;
    if (isCentralKp(kp1, half_center_factor) &&
        isCentralKp(kp2, half_center_factor)) {
      good_matches.emplace_back(m);
      points1.emplace_back(kp1.pt);
      points2.emplace_back(kp2.pt);
    }
  }

  std::cout << "[INFO]: selected " << good_matches.size() << " matches from "
            << better_matches.size() << " by position(central is better)"
            << std::endl;

  // debug draw
  if (debug_draw) {
    std::cout << "[DEBUG]: debug draw for frame " << frame_id_ << " and "
              << frame->frame_id_ << std::endl;
    cv::Mat img_match, img_good_match;
    cv::drawMatches(img_, keypoints_, frame->img_, frame->keypoints_,
                    all_matches, img_match);
    cv::drawMatches(img_, keypoints_, frame->img_, frame->keypoints_,
                    good_matches, img_good_match);
    // cv::imshow("all_matches", img_match);
    cv::resize(img_good_match, img_good_match, {0, 0}, 0.4, 0.4);
    cv::imshow("good_matches", img_good_match);

    cv::Mat un_img_good_match;
    cv::drawMatches(un_img_, un_keypoints_, frame->un_img_,
                    frame->un_keypoints_, good_matches, un_img_good_match);
    cv::resize(un_img_good_match, un_img_good_match, {0, 0}, 0.4, 0.4);
    cv::imshow("undistorted_good_matches", un_img_good_match);
  }

  return good_matches.size();
}

cv::Point2f Frame::project(const cv::Mat &x3D) {
  cv::Mat ret = camera_model_->getNewK() * (Rcw_ * x3D + tcw_);
  ret = ret / ret.at<double>(2);
  return cv::Point2f(ret.at<double>(0), ret.at<double>(1));
}

cv::Point2f Frame::project(const Eigen::Vector3d &mappoint) {
  cv::Mat x3D(3, 1, CV_64F);
  x3D.at<double>(0) = mappoint[0];
  x3D.at<double>(1) = mappoint[1];
  x3D.at<double>(2) = mappoint[2];
  return project(x3D);
}

bool Frame::checkDepthValid(const cv::Mat &x3D) {
  cv::Mat x3Dc = Rcw_ * x3D + tcw_;
  return x3Dc.at<double>(2) > 0;
}

std::vector<cv::KeyPoint> Frame::getUnKeyPoints() const {
  return un_keypoints_;
}

cv::KeyPoint Frame::getUnKeyPoints(const int &keypoint_idx) const {
  return un_keypoints_[keypoint_idx];
}

std::vector<int> Frame::getMappointIdx() const { return mappoint_idx_; }

int Frame::getMappointIdx(const int &keypoint_idx) const {
  return mappoint_idx_[keypoint_idx];
}

void Frame::setMappointIdx(const int &keypoint_idx, const int &mappoint_idx) {
  mappoint_idx_[keypoint_idx] = mappoint_idx;
}

Eigen::Matrix3d Frame::getEigenRot() {
  Eigen::Matrix3d ret;
  std::unique_lock<std::mutex> lock(mutex_pose_);
  cv::cv2eigen(Rcw_, ret);
  return ret;
}

Eigen::Vector3d Frame::getEigenTrans() {
  Eigen::Vector3d ret;
  std::unique_lock<std::mutex> lock(mutex_pose_);
  cv::cv2eigen(tcw_, ret);
  return ret;
}

Eigen::Matrix3d Frame::getEigenRotWc() { return getEigenRot().inverse(); }

Eigen::Vector3d Frame::getEigenTransWc() {
  return -getEigenRot().inverse() * getEigenTrans();
}

cv::Mat Frame::getPose() {
  std::unique_lock<std::mutex> lock(mutex_pose_);
  return Tcw_;
}

cv::Mat Frame::getProjectionMatrix() {
  std::unique_lock<std::mutex> lock(mutex_pose_);
  cv::Mat P = camera_model_->getNewK() * Tcw_.rowRange(0, 3);
  return P;
}

void Frame::setPose(const Eigen::Matrix4d &mat) {
  std::unique_lock<std::mutex> lock(mutex_pose_);
  cv::eigen2cv(mat, Tcw_);
  Tcw_.rowRange(0, 3).colRange(0, 3).copyTo(Rcw_);
  Tcw_.rowRange(0, 3).col(3).copyTo(tcw_);
}

void Frame::setPose(const Eigen::Matrix3d &R, const Eigen::Vector3d &t) {
  std::unique_lock<std::mutex> lock(mutex_pose_);
  cv::eigen2cv(R, Rcw_);
  cv::eigen2cv(t, tcw_);
  Tcw_ = cv::Mat::zeros(4, 4, CV_64F);
  Rcw_.copyTo(Tcw_.rowRange(0, 3).colRange(0, 3));
  tcw_.copyTo(Tcw_.rowRange(0, 3).col(3));
}

void Frame::setPose(const cv::Mat &mat) {
  std::unique_lock<std::mutex> lock(mutex_pose_);
  mat.copyTo(Tcw_);
  Tcw_.rowRange(0, 3).colRange(0, 3).copyTo(Rcw_);
  Tcw_.rowRange(0, 3).col(3).copyTo(tcw_);
}

void Frame::setPose(const cv::Mat &R, const cv::Mat &t) {
  std::unique_lock<std::mutex> lock(mutex_pose_);
  R.copyTo(Rcw_);
  t.copyTo(tcw_);
  Tcw_ = cv::Mat::zeros(4, 4, CV_64F);
  Rcw_.copyTo(Tcw_.rowRange(0, 3).colRange(0, 3));
  tcw_.copyTo(Tcw_.rowRange(0, 3).col(3));
}

void Frame::rotateWorld(const Eigen::Quaterniond &q_ds) {
  Eigen::Isometry3d Tcs = Eigen::Isometry3d::Identity();
  Tcs.rotate(getEigenRot());
  Tcs.pretranslate(getEigenTrans());

  Eigen::Isometry3d Tsc = Tcs.inverse();
  std::cout << "Tcs: " << std::endl << Tcs.matrix() << std::endl;
  std::cout << "Tsc: " << std::endl << Tsc.matrix() << std::endl;

  Eigen::Isometry3d Tds = Eigen::Isometry3d::Identity();
  Tds.rotate(q_ds);

  std::cout << "Tds: " << std::endl << Tds.matrix() << std::endl;

  Eigen::Isometry3d Tdc = Tds * Tsc;
  std::cout << "Tdc: " << std::endl << Tdc.matrix() << std::endl;

  Eigen::Isometry3d Tcd = Tdc.inverse();
  std::cout << "Tcd: " << std::endl << Tcd.matrix() << std::endl;

  setPose(Tcd.matrix());
}

int Frame::getFrameId() const { return frame_id_; }

bool Frame::isCentralKp(const cv::KeyPoint &kp,
                        const double &half_center_factor) {
  int c = img_.cols;
  int r = img_.rows;
  if (std::abs(kp.pt.x - (c / 2.0)) < c * half_center_factor &&
      std::abs(kp.pt.y - (r / 2.0)) < r * half_center_factor) {
    return true;
  }
  return false;
}

void Frame::debugDraw(const double &scale_image) {
  std::cout << "[DEBUG]: debug draw" << std::endl;
  std::vector<cv::DMatch> all_matches;
  matcher_->match(descriptors_, descriptors_, all_matches);

  cv::Mat mat;
  cv::drawMatches(img_, keypoints_, un_img_, un_keypoints_, all_matches, mat);
  cv::resize(mat, mat, {0, 0}, scale_image, scale_image);
  cv::imshow("debug draw", mat);
  cv::waitKey();
}

int Frame::debugCountMappoints() {
  int cnt = 0;
  for (const int &i : mappoint_idx_) {
    if (i >= 0) {
      cnt++;
    }
  }
  return cnt;
}

int Frame::total_frame_cnt_ = 0;

cv::Ptr<cv::FeatureDetector> Frame::detector_ = cv::ORB::create(1000);
cv::Ptr<cv::DescriptorExtractor> Frame::extrator_ = cv::ORB::create(1000);
cv::Ptr<cv::DescriptorMatcher> Frame::matcher_ =
    cv::DescriptorMatcher::create("BruteForce-Hamming");
