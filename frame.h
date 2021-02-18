/**
 * @file frame.h
 * @author xiaotaw (you@domain.com)
 * @brief 普通帧
 * @version 0.1
 * @date 2021-02-08
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once
#include "mappoint.h"
#include <Eigen/Core>
#include <iostream>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

/**
 * @brief 相机内参(不包含畸变系数)
 */
class Intrinsic {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  Intrinsic() {}

  Intrinsic(const double &vfx, const double &vfy, const double &vcx,
            const double &vcy)
      : fx(vfx), fy(vfy), cx(vcx), cy(vcy) {}

  /**
   * @brief 返回缩放后的图像对应的内参
   * @param s 缩放系数，s>1代表图像放大，s<1代表图像缩小。
   */
  inline Intrinsic scale(const double &s) {
    return Intrinsic(fx * s, fy * s, cx * s, cy * s);
  }

  /**
   * @brief 返回转置后的图像对应的内参
   */
  inline Intrinsic transpose() { return Intrinsic(fy, fx, cy, cx); }

  /**
   * @brief 获取内参矩阵
   */
  inline cv::Mat K() {
    return (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
  }

  inline Eigen::Matrix3d getEigenK() {
    Eigen::Matrix3d ret;
    cv::cv2eigen(K(), ret);
    return ret;
  }

public:
  double fx;
  double fy;
  double cx;
  double cy;
};

/**
 * @brief 普通图像帧
 * @note 使用ORB特征，基于hamming距离的暴力匹配
 */
class Frame {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  
  using Ptr = std::shared_ptr<Frame>;

  // ctor
  Frame(const cv::Mat &img);
  Frame(const cv::Mat &img, const Intrinsic &intrinsic);

  /**
   * @brief 与另一帧进行特征点匹配，并根据距离，进行简单筛选
   *
   * @param frame [IN] 另一帧图像
   * @param good_matches [OUT] 好的匹配
   * @param debug_draw [IN] 是否画出匹配图
   */
  void matchWith(const Frame::Ptr frame, std::vector<cv::DMatch> &good_matches,
                 const bool &debug_draw);

  inline Eigen::Matrix3d getEigenR() const {
    Eigen::Matrix3d ret;
    cv::cv2eigen(R_cw_, ret);
    return ret;
  }

  inline Eigen::Vector3d getEigenT() const {
    Eigen::Vector3d ret;
    cv::cv2eigen(t_cw_, ret);
    return ret;
  }

public:
  // 相机内参
  Intrinsic intrinsic_;

  // 图片，特征点，描述符
  cv::Mat img_;
  std::vector<cv::KeyPoint> keypoints_;
  cv::Mat descriptors_;

  // 特征点对应的3D空间点
  std::vector<MapPoint::WPtr> map_points_;

  static cv::Ptr<cv::FeatureDetector> detector_;
  static cv::Ptr<cv::DescriptorExtractor> extrator_;
  static cv::Ptr<cv::DescriptorMatcher> matcher_;

  // 代码中T_cw，表示位姿T^c_w。
  // 假设：点在相机坐标系下的值P_c，点在世界坐标系下的坐标值P_w，则 P_w = T^c_w
  // * P_c
  cv::Mat R_cw_;
  cv::Mat t_cw_;

private:
  /**
   * @brief 仅供构造函数使用，进行初始化
   */
  void init();
};