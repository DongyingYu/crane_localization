#pragma once
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

class Intrinsic;

/**
 * @brief 使用ORB特征，基于hamming距离的暴力匹配
 *
 */
class Frame {
public:
  Frame(const cv::Mat &img) : img_(img.clone()) {
    // Oriented FAST
    std::vector<cv::KeyPoint> keypoints;
    detector_->detect(img_, keypoints);

    // 去除左上角和右下角的字幕的干扰
    // 注：图片可能被转置了
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
  }

  /**
   * @brief 与另一帧进行特征点匹配，并简单筛选
   *
   * @param frame [IN]
   * @param good_matches [OUT]
   * @param debug_draw [IN]
   */
  inline void matchWith(const cv::Ptr<Frame> frame,
                        std::vector<cv::DMatch> &good_matches,
                        const bool debug_draw) {

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

public:
  cv::Mat img_;
  std::vector<cv::KeyPoint> keypoints_;
  cv::Mat descriptors_;

  static cv::Ptr<cv::FeatureDetector> detector_;
  static cv::Ptr<cv::DescriptorExtractor> extrator_;
  static cv::Ptr<cv::DescriptorMatcher> matcher_;
};

cv::Ptr<cv::FeatureDetector> Frame::detector_ = cv::ORB::create();
cv::Ptr<cv::DescriptorExtractor> Frame::extrator_ = cv::ORB::create();
cv::Ptr<cv::DescriptorMatcher> Frame::matcher_ =
    cv::DescriptorMatcher::create("BruteForce-Hamming");

class Intrinsic {
public:
  double fx;
  double fy;
  double cx;
  double cy;

  Intrinsic(const double &vfx, const double &vfy, const double &vcx,
            const double &vcy)
      : fx(vfx), fy(vfy), cx(vcx), cy(vcy) {}

  inline Intrinsic scale(const double &s) {
    return Intrinsic(fx * s, fy * s, cx * s, cy * s);
  }

  inline Intrinsic transpose() { return Intrinsic(fy, fx, cy, cx); }

  inline cv::Mat K() {
    return (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
  }
};