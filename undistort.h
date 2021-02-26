/**
 * @file undistort.h
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2021-02-23
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once
#include "intrinsic.h"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <memory>

class UndistorterFisheye {
public:
  using Ptr = std::shared_ptr<UndistorterFisheye>;
  UndistorterFisheye(const int &cols, const int &rows, const double &fx,
                     const double &fy, const double &cx, const double &cy,
                     const double &k1, const double &k2, const double &k3,
                     const double &k4);

  UndistorterFisheye(const cv::Size &img_size, const Intrinsic &intrinsic,
                     const double &k1, const double &k2, const double &k3,
                     const double &k4);

  cv::Mat getD() const;
  cv::Mat getK() const;
  cv::Mat getNewK() const;
  Intrinsic getNewIntrinsic() const;

  // Note: 结果与undistortPoint不符，不可用
  void undistort(const cv::Mat &img, cv::Mat &img_undistorted);

  void undistortPoint(const std::vector<cv::KeyPoint> &kps,
                      std::vector<cv::KeyPoint> &un_kps);

  void undistortImage(const cv::Mat &img, cv::Mat &un_img);

private:
  Intrinsic intrinsic_, new_intrinsic_;
  cv::Size img_size_;
  cv::Mat K_, newK_, D_, map1_, map2_;
};