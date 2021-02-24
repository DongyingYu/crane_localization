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
  intrinsic_ = Intrinsic(fx, fy, cx, cy);
  K_ = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

  std::vector<double> _dist_coeffs = {k1, k2, k3, k4};
  D_ = cv::Mat(4, 1, CV_64F, &_dist_coeffs[0]);

  cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
      K_, D_, img_size_, cv::Matx33d::eye(), newK_, 1);

  new_intrinsic_.fromK(newK_);

  std::cout << "[INFO]: K=" << std::endl << K_ << std::endl;
  std::cout << "[INFO]: new K " << std::endl << newK_ << std::endl;

  cv::Mat_<double> I = cv::Mat_<double>::eye(3, 3);
  map1_ = cv::Mat::zeros(img_size_, CV_16SC2);
  map2_ = cv::Mat::zeros(img_size_, CV_16UC1);
  cv::fisheye::initUndistortRectifyMap(K_, D_, I, newK_, img_size_,
                                       map1_.type(), map1_, map2_);
}

UndistorterFisheye::UndistorterFisheye(const cv::Size &img_size,
                                       const Intrinsic &intrinsic,
                                       const double &k1, const double &k2,
                                       const double &k3, const double &k4)
    : img_size_(img_size), intrinsic_(intrinsic) {
  K_ = intrinsic_.K();
  std::vector<double> _dist_coeffs = {k1, k2, k3, k4};
  D_ = cv::Mat(4, 1, CV_64F, &_dist_coeffs[0]);

  cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
      K_, D_, img_size_, cv::Matx33d::eye(), newK_, 1);
  new_intrinsic_.fromK(newK_);

  std::cout << "[INFO]: K=" << std::endl << K_ << std::endl;
  std::cout << "[INFO]: new K " << std::endl << newK_ << std::endl;

  cv::Mat_<double> I = cv::Mat_<double>::eye(3, 3);
  map1_ = cv::Mat::zeros(img_size_, CV_16SC2);
  map2_ = cv::Mat::zeros(img_size_, CV_16UC1);
  cv::fisheye::initUndistortRectifyMap(K_, D_, I, newK_, img_size_,
                                       map1_.type(), map1_, map2_);
}

cv::Mat UndistorterFisheye::getD() const { return D_; }
cv::Mat UndistorterFisheye::getK() const { return K_; }
cv::Mat UndistorterFisheye::getNewK() const { return newK_; }

Intrinsic UndistorterFisheye::getNewIntrinsic() const { return new_intrinsic_; }

void UndistorterFisheye::undistort(const cv::Mat &img,
                                   cv::Mat &img_undistorted) {
  std::cout << "[ERROR]: this function does not match with "
               "cv::fisheye::undistortPoint"
            << std::endl;
  std::cout << "         use cv::fisheye::undistortImage instead please."
            << std::endl;
  cv::remap(img, img_undistorted, map1_, map2_, cv::INTER_LINEAR,
            cv::BORDER_CONSTANT);
}

void UndistorterFisheye::undistortImage(const cv::Mat &img, cv::Mat &un_img) {
  assert(img.size() == img_size_);
  cv::fisheye::undistortImage(img, un_img, K_, D_, newK_, img_size_);
}

void UndistorterFisheye::undistortPoint(const std::vector<cv::KeyPoint> &kps,
                                        std::vector<cv::KeyPoint> &un_kps) {
  if (kps.empty()) {
    return;
  }

  // Fill matrix with points
  cv::Mat mat(kps.size(), 2, CV_32F);
  for (int i = 0; i < kps.size(); i++) {
    mat.at<float>(i, 0) = kps[i].pt.x;
    mat.at<float>(i, 1) = kps[i].pt.y;
  }

  // Undistort points
  mat = mat.reshape(2);
  cv::Mat_<double> I = cv::Mat_<double>::eye(3, 3);
  cv::fisheye::undistortPoints(mat, mat, K_, D_, I, newK_);
  mat = mat.reshape(1);

  // Fill undistorted keypoint vector
  un_kps.resize(kps.size());
  for (int i = 0; i < kps.size(); i++) {
    cv::KeyPoint kp = kps[i];
    kp.pt.x = mat.at<float>(i, 0);
    kp.pt.y = mat.at<float>(i, 1);
    un_kps[i] = kp;
  }
}