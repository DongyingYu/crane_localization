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

void Frame::init() {
  assert(!img_.empty());

  // 计算Oriented FAST角点
  std::vector<cv::KeyPoint> keypoints;
  detector_->detect(img_, keypoints);

  // 去除左上角和右下角的字幕的干扰 (注：图片可能被转置了)
  const double lf = (6.5 / 21.0);
  const double sf = (1.5 / 12.0);
  double x_s = img_.cols * (img_.cols > img_.rows ? lf : sf);
  double y_s = img_.rows * (img_.cols > img_.rows ? sf : lf);

  for (const cv::KeyPoint &kp : keypoints) {
    if ((kp.pt.x < x_s && kp.pt.y < y_s) ||
        (kp.pt.x > img_.cols - x_s && kp.pt.y > img_.rows - y_s)) {
      continue;
    } else {
      keypoints_.emplace_back(kp);
    }
  }

  // BRIEF
  extrator_->compute(img_, keypoints_, descriptors_);
  std::cout << "[INFO]: keypoints_.size()=" << keypoints_.size();
  std::cout << " descriptors_.size()=" << descriptors_.size() << std::endl;

  // 初始化
  mappoint_idx_ = std::vector<int>(keypoints_.size(), -1);

  frame_id_ = Frame::total_frame_cnt_++;
}

Frame::Frame(const cv::Mat &img) : img_(img.clone()) { init(); }

Frame::Frame(const cv::Mat &img, const Intrinsic &intrinsic)
    : img_(img.clone()), intrinsic_(intrinsic) {
  init();
}

void Frame::matchWith(const Frame::Ptr frame,
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
      cv::Point2f pt1 = keypoints_[m.queryIdx].pt;
      cv::Point2f pt2 = frame->keypoints_[m.trainIdx].pt;
      pts1.emplace_back(pt1);
      pts2.emplace_back(pt2);
      pts_diff.emplace_back(pt1 - pt2);
    }
  }
  std::cout << "[INFO]: selected " << tmp_matches.size() << " matches from "
            << all_matches.size() << std::endl;

  // 根据运动约束，检测配对点是否合理
  cv::Point2f ave, stddev;
  calAveStddev(pts_diff, ave, stddev);
  std::cout << "[INFO]: Point uv diff, ave " << ave << " stddev " << stddev
            << std::endl;

  std::vector<cv::DMatch> better_matches;
  pts1.clear();
  pts2.clear();
  for (int i = 0; i < int(tmp_matches.size()); ++i) {
    cv::Point2f diff = pts_diff[i];
    if (std::abs(diff.y) > stddev.y || std::abs(diff.x) > 3 * stddev.x) {
      continue;
    }
    auto m = tmp_matches[i];
    cv::Point2f pt1 = keypoints_[m.queryIdx].pt;
    cv::Point2f pt2 = frame->keypoints_[m.trainIdx].pt;
    better_matches.emplace_back(m);
    pts1.emplace_back(pt1);
    pts2.emplace_back(pt2);
  }

  std::cout << "[INFO]: selected " << better_matches.size() << " matches from "
            << tmp_matches.size() << std::endl;

  pts_diff.clear();
  for (int i = 0; i < int(pts1.size()); ++i) {
    pts_diff.emplace_back(pts1[i] - pts2[i]);
  }

  calAveStddev(pts_diff, ave, stddev);
  std::cout << "[INFO]: Point uv diff, ave " << ave << " stddev " << stddev
            << std::endl;

  // 优先使用靠近图像中间的特征点（越往边缘，畸变越严重）
  good_matches.clear();
  points1.clear();
  points2.clear();
  for (const auto &m : better_matches) {
    auto kp1 = keypoints_[m.queryIdx];
    auto kp2 = frame->keypoints_[m.trainIdx];
    // 0.5表示认为整个图像都可以，即无任何筛选
    double half_center_factor = 0.45;
    if (isCentralKp(kp1, half_center_factor) &&
        isCentralKp(kp2, half_center_factor)) {
      good_matches.emplace_back(m);
      points1.emplace_back(kp1.pt);
      points2.emplace_back(kp2.pt);
    }
  }

  std::cout << "[INFO]: selected " << good_matches.size() << " matches from "
            << better_matches.size() << std::endl;

  // debug draw
  if (debug_draw) {
    cv::Mat img_match, img_good_match;
    cv::drawMatches(img_, keypoints_, frame->img_, frame->keypoints_,
                    all_matches, img_match);
    cv::drawMatches(img_, keypoints_, frame->img_, frame->keypoints_,
                    good_matches, img_good_match);
    // cv::imshow("all_matches", img_match);
    cv::imshow("good_matches", img_good_match);
  }
}

Eigen::Matrix3d Frame::getEigenR() const {
  Eigen::Matrix3d ret;
  cv::cv2eigen(Rcw_, ret);
  return ret;
}

Eigen::Vector3d Frame::getEigenT() const {
  Eigen::Vector3d ret;
  cv::cv2eigen(tcw_, ret);
  return ret;
}

Eigen::Matrix3d Frame::getEigenRwc() const { return getEigenR().inverse(); }
Eigen::Vector3d Frame::getEigenTwc() const {
  return -getEigenR().inverse() * getEigenT();
}

void Frame::setPose(const Eigen::Matrix4d &mat) {
  cv::eigen2cv(mat, Tcw_);
  Tcw_.rowRange(0, 3).colRange(0, 3).copyTo(Rcw_);
  Tcw_.rowRange(0, 3).col(3).copyTo(tcw_);
}

void Frame::setPose(const cv::Mat &mat) {
  mat.copyTo(Tcw_);
  Tcw_.rowRange(0, 3).colRange(0, 3).copyTo(Rcw_);
  Tcw_.rowRange(0, 3).col(3).copyTo(tcw_);
}

void Frame::setPose(const cv::Mat &R, const cv::Mat &t) {
  R.copyTo(Rcw_);
  t.copyTo(tcw_);
  Tcw_ = cv::Mat::zeros(4, 4, CV_64F);
  Rcw_.copyTo(Tcw_.rowRange(0, 3).colRange(0, 3));
  tcw_.copyTo(Tcw_.rowRange(0, 3).col(3));
}

void Frame::rotateWorld(const Eigen::Quaterniond &q_ds) {
  Eigen::Isometry3d Tcs = Eigen::Isometry3d::Identity();
  Tcs.rotate(getEigenR());
  Tcs.pretranslate(getEigenT());

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

int Frame::total_frame_cnt_ = 0;

cv::Ptr<cv::FeatureDetector> Frame::detector_ = cv::ORB::create();
cv::Ptr<cv::DescriptorExtractor> Frame::extrator_ = cv::ORB::create();
cv::Ptr<cv::DescriptorMatcher> Frame::matcher_ =
    cv::DescriptorMatcher::create("BruteForce-Hamming");
