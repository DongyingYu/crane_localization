/**
 * @file intrinsic.h
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2021-02-24
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
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

  inline Intrinsic fromK(const cv::Mat &K) {
    fx = K.at<double>(0, 0);
    fy = K.at<double>(1, 1);
    cx = K.at<double>(0, 2);
    cy = K.at<double>(1, 2);
    return *this;
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
