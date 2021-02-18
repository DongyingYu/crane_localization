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
  map_points_.resize(keypoints_.size());
}

Frame::Frame(const cv::Mat &img) : img_(img.clone()) { init(); }

Frame::Frame(const cv::Mat &img, const Intrinsic &intrinsic)
    : img_(img.clone()), intrinsic_(intrinsic) {
  init();
}

void Frame::matchWith(const Frame::Ptr frame,
                      std::vector<cv::DMatch> &good_matches,
                      const bool &debug_draw) {

  std::vector<cv::DMatch> all_matches;

  // 匹配特征点
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
  good_matches.clear();
  for (const cv::DMatch &m : all_matches) {
    if (m.distance <= dmax * 0.6) {
      good_matches.emplace_back(m);
    }
  }
  std::cout << "[INFO]: all match: " << all_matches.size()
            << " good match: " << good_matches.size() << std::endl;

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

cv::Ptr<cv::FeatureDetector> Frame::detector_ = cv::ORB::create();
cv::Ptr<cv::DescriptorExtractor> Frame::extrator_ = cv::ORB::create();
cv::Ptr<cv::DescriptorMatcher> Frame::matcher_ =
    cv::DescriptorMatcher::create("BruteForce-Hamming");
