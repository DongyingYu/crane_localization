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
#include "gridmatcher.h"
#include "utils.h"

Frame::Frame(const cv::Mat &img) : img_(img.clone()) { init(); }

Frame::Frame(const cv::Mat &img, const CameraModel::Ptr &camera_model)
    : img_(img.clone()), camera_model_(camera_model) {
  init();
}

Frame::Frame(const cv::Mat &img, Vocabulary *voc)
    : vocabulary_(voc), img_(img.clone()) {
  init();
}

Frame::Frame(const cv::Mat &img, const CameraModel::Ptr &camera_model,
             Vocabulary *voc)
    : img_(img), camera_model_(camera_model), vocabulary_(voc) {
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
      keypoints_bow_.emplace_back(kp);

      if (isCentralKp(kp, 0.8)) {
        keypoints_.emplace_back(kp);
      }
    }
  }

  // 2. 去畸变
  if (camera_model_) {
    camera_model_->undistortKeyPoint(keypoints_, un_keypoints_);
    // camera_model_->undistortImage(img_, un_img_);
    un_img_ = img_.clone();
  } else {
    un_keypoints_ = keypoints;
    un_img_ = img_.clone();
  }

  // 3. 描述子计算（BRIEF）
  extrator_->compute(img_, keypoints_, descriptors_);
  extrator_->compute(img_, keypoints_bow_, descriptors_bow_);
  if (keypoints_.size() < 50) {
    std::cout << "[WARNING]: too few keypoints detected " << keypoints_.size()
              << std::endl;
  }

  // 4. 其他初始化
  mappoints_id_ = std::vector<int>(keypoints_.size(), -1);
  frame_id_ = Frame::total_frame_cnt_++;
}

int Frame::matchWith(const Frame::Ptr frame,
                     std::vector<cv::DMatch> &good_matches,
                     std::vector<cv::Point2f> &points1,
                     std::vector<cv::Point2f> &points2, float &ave_x,
                     const double &debug_draw) {
  std::cout << "[INFO]: The value of debug_draw is :  " << debug_draw
            << std::endl;
  // 匹配特征点
  std::vector<cv::DMatch> all_matches;
  matcher_->match(descriptors_, frame->descriptors_, all_matches);

  /*
  // 统计匹配距离（Hamming）的最大值和最小值
  double dmin = 1;
  double dmax = 0;
  for (const cv::DMatch &m : all_matches) {
    dmin = m.distance < dmin ? m.distance : dmin;
    dmax = m.distance > dmax ? m.distance : dmax;
  }

  // 根据经验，筛选匹配
  std::vector<cv::DMatch> tmp_matches;
  std::vector<cv::Point2f> pts1, pts2, pts_diff;
  for (const cv::DMatch &m : all_matches) {
    if (m.distance <= dmax * 0.8) {
      cv::Point2f pt1 = un_keypoints_[m.queryIdx].pt;
      cv::Point2f pt2 = frame->un_keypoints_[m.trainIdx].pt;
      auto pt_diff = pt1 - pt2;
      if (std::abs(pt_diff.y) > 5) {
        continue;
      }
      tmp_matches.emplace_back(m);
      pts1.emplace_back(pt1);
      pts2.emplace_back(pt2);
      pts_diff.emplace_back(pt_diff);
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
  int n_outliers = 0;
  for (int i = 0; i < int(tmp_matches.size()); ++i) {
    cv::Point2f abs_diff =
        cv::Point2d(std::abs(pts_diff[i].x), std::abs(pts_diff[i].y));
    cv::Point2f ddiff = abs_diff - ave;
    // 限制匹配点对在y方向上的偏移量
    if (std::abs(ddiff.y) > 3 + 3 * stddev.y ||
        std::abs(ddiff.x) > 3 + 3 * stddev.x) {
      n_outliers++;
      // std::cout << "[INFO]: outlier, ddiff.x=" << ddiff.x
      //           << " ddiff.y=" << ddiff.y << std::endl;
      continue;
    }
    auto m = tmp_matches[i];
    cv::Point2f pt1 = un_keypoints_[m.queryIdx].pt;
    cv::Point2f pt2 = frame->un_keypoints_[m.trainIdx].pt;
    better_matches.emplace_back(m);
    pts1.emplace_back(pt1);
    pts2.emplace_back(pt2);
  }

  if (1.0 * n_outliers / tmp_matches.size() > 0.3) {
    std::cout << "[WARNING]: too much outliers " << n_outliers << std::endl;
  }

  std::cout << "[INFO]: selected " << better_matches.size() << " matches from "
            << tmp_matches.size() << " by stddev " << std::endl;

  pts_diff.clear();
  for (int i = 0; i < int(pts1.size()); ++i) {
    pts_diff.emplace_back(pts1[i] - pts2[i]);
  }
  // 对特征点筛选后，在经过一轮计算，打印输出结果
  calAveStddev(pts_diff, ave, stddev, true);
  std::cout << "[INFO]: Point uv diff, ave " << ave << " stddev " << stddev
            << std::endl;

  ave_x = ave.x;
  */

  ave_x = 0;
  std::vector<bool> vbInliers;
  Gridmatcher::Ptr gridmatch = std::make_shared<Gridmatcher>(
      this->un_keypoints_, this->img_.size(), frame->un_keypoints_,
      frame->img_.size(), all_matches);
  int num_inliers =
      gridmatch->GetInlierMask(vbInliers, cv::Size(20, 20), false, false);
  cout << "[INFO]: Get total " << num_inliers << " matches." << endl;

  // collect matches
  std::vector<cv::DMatch> matches_good;
  for (size_t i = 0; i < vbInliers.size(); ++i) {
    if (vbInliers[i] == true) {
      matches_good.push_back(all_matches[i]);
    }
  }
  std::cout << "[INFO]: matches_good size   " << matches_good.size()
            << std::endl;

  // 优先使用靠近图像中间的特征点（越往边缘，畸变越严重）
  good_matches.clear();
  points1.clear();
  points2.clear();
  for (const auto &m : matches_good /*better_matches*/) {
    auto kp1 = un_keypoints_[m.queryIdx];
    auto kp2 = frame->un_keypoints_[m.trainIdx];
    auto pt_diff = kp1.pt - kp2.pt;
    if (abs(pt_diff.y) > 10) continue;
    good_matches.emplace_back(m);
    points1.emplace_back(kp1.pt);
    points2.emplace_back(kp2.pt);
  }

  std::cout << "[INFO]: selected " << good_matches.size()
            << " by position(central is better)" << std::endl;

  // debug draw
  if (debug_draw > 0) {
    std::cout << "[DEBUG]: debug draw for frame " << frame_id_ << " and "
              << frame->frame_id_ << std::endl;
    cv::Mat un_img_good_match;
    cv::drawMatches(un_img_, un_keypoints_, frame->un_img_,
                    frame->un_keypoints_, good_matches, un_img_good_match);
    cv::resize(un_img_good_match, un_img_good_match, {0, 0}, debug_draw,
               debug_draw);
    cv::imshow("undistorted_good_matches", un_img_good_match);
    // cv::waitKey();
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

int Frame::getUnKeyPointsSize() const { return un_keypoints_.size(); }

std::vector<int> Frame::getMappointId() const { return mappoints_id_; }

int Frame::getMappointId(const int &keypoint_idx) const {
  return mappoints_id_[keypoint_idx];
}

void Frame::setMappointIdx(const int &keypoint_idx, const int &mappoint_idx) {
  mappoints_id_[keypoint_idx] = mappoint_idx;
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

std::vector<cv::Mat> Frame::toDescriptorVector() {
  std::vector<cv::Mat> vDesc;
  vDesc.reserve(descriptors_bow_.rows);
  for (int j = 0; j < descriptors_bow_.rows; j++)
    vDesc.push_back(descriptors_bow_.row(j));
  return vDesc;
}

void Frame::computeBoW() {
  std::vector<cv::Mat> vCurrentDesc = toDescriptorVector();
  // BdWVec为Bow特征向量，FeatVec为正向索引
  vocabulary_->transform(vCurrentDesc, bow_vec_, feat_vec_, 4);
}

DBoW2::BowVector Frame::getBowVoc() { return bow_vec_; }

void Frame::createVocabulary(
    Vocabulary &voc, std::string &filename,
    const std::vector<std::vector<cv::Mat>> &descriptors) {
  std::cout << " Creating vocabulary. May take some time ... " << std::endl;
  voc.create(descriptors);
  std::cout << " Creating Done ! " << std::endl;
  std::cout << " vocabulary infirmation: " << std::endl
            << voc << std::endl
            << std::endl;
  // 保存词典
  std::cout << " Saving vocabulary ... " << std::endl;
  voc.saveToTextFile(filename);
  std::cout << " saved to file: " << filename << std::endl;
}

size_t Frame::getFrameId() const { return frame_id_; }

Eigen::Matrix3d Frame::getEigenNewK() const {
  Eigen::Matrix3d ret;
  cv::cv2eigen(camera_model_->getNewK(), ret);
  return ret;
}

cv::Mat Frame::getImage() const { return img_; }

bool Frame::isCentralKp(const cv::KeyPoint &kp,
                        const double &half_center_factor) {
  float a = img_.cols / 2.0;
  float b = img_.rows / 2.0;
  float x = kp.pt.x - a;
  float y = kp.pt.y - b;

  if ((x * x) / (a * a) + (y * y) / (b * b) <
      half_center_factor * half_center_factor) {
    return true;
  } else {
    return false;
  }
}

void Frame::setFlag(const bool cal_flag) { offset_flag_ = cal_flag; }

bool Frame::getFlag() const { return offset_flag_; }

void Frame::setAbsPosition(const double &position) { abs_position_ = position; }

double Frame::getAbsPosition() const { return abs_position_; }

void Frame::debugDraw(const double &scale_image) {
  std::cout << "[DEBUG]: debug draw" << std::endl;
  std::vector<cv::DMatch> all_matches;
  matcher_->match(descriptors_, descriptors_, all_matches);

  cv::Mat mat;
  cv::drawMatches(img_, keypoints_, un_img_, un_keypoints_, all_matches, mat);
  cv::resize(mat, mat, {0, 0}, scale_image, scale_image);
  cv::imshow("debug draw", mat);
  // cv::waitKey();
}

void Frame::debugPrintPose() {
  Eigen::Quaterniond q(getEigenRot());
  Eigen::Vector3d t = getEigenTrans();
  std::cout << "[INFO]: Frame " << frame_id_ << ": " << toString(q) << ", "
            << toString(t) << std::endl;
}

void Frame::setVocabulary(Vocabulary *voc) { vocabulary_ = voc; }

void Frame::releaseImage() {
  img_.release();
  un_img_.release();
}

size_t Frame::total_frame_cnt_ = 0;

cv::Ptr<cv::FeatureDetector> Frame::detector_ = cv::ORB::create(1000);
cv::Ptr<cv::DescriptorExtractor> Frame::extrator_ = cv::ORB::create(1000);
cv::Ptr<cv::DescriptorMatcher> Frame::matcher_ =
    cv::DescriptorMatcher::create("BruteForce-Hamming");
