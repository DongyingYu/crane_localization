/**
 * @file undistort.cc
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2021-02-23
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "undistort.h"

UndistorterFisheye::UndistorterFisheye(const int &cols, const int &rows,
                                       const double &fx, const double &fy,
                                       const double &cx, const double &cy,
                                       const double &k1, const double &k2,
                                       const double &k3, const double &k4) {
  img_size_ = cv::Size(cols, rows);
  K_ = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

  std::vector<double> _dist_coeffs = {k1, k2, k3, k4};
  cv::Mat D = cv::Mat(4, 1, CV_64F, &_dist_coeffs[0]);

  cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
      K_, D, img_size_, cv::Matx33d::eye(), newK_, 1);

  std::cout << "[INFO]: K=" << std::endl << K_ << std::endl;
  std::cout << "[INFO]: new K " << std::endl << newK_ << std::endl;

  cv::Mat_<double> I = cv::Mat_<double>::eye(3, 3);
  map1_ = cv::Mat::zeros(img_size_, CV_16SC2);
  map2_ = cv::Mat::zeros(img_size_, CV_16UC1);
  cv::fisheye::initUndistortRectifyMap(K_, D, I, newK_, img_size_, map1_.type(),
                                       map1_, map2_);
}

cv::Mat UndistorterFisheye::getNewK() const { return newK_; }

void UndistorterFisheye::undistort(const cv::Mat &img,
                                   cv::Mat &img_undistorted) {
  cv::remap(img, img_undistorted, map1_, map2_, cv::INTER_LINEAR,
            cv::BORDER_CONSTANT);
}