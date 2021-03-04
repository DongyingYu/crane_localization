/**
 * @file camera_model.cc
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2021-02-23
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "camera_model.h"

CameraModel::CameraModel(const std::string &camera_model,
                         const std::vector<double> &intrinsic_vector,
                         const cv::Size &img_size)
    : camera_model_(camera_model), intr_vec_(intrinsic_vector),
      img_size_(img_size) {}

CameraModel::CameraModel(const std::string &camera_model,
                         const std::vector<double> &intrinsic_vector,
                         const cv::Size &img_size,
                         const std::string &distortion_model,
                         const std::vector<double> &distortion_coeffs)
    : camera_model_(camera_model), intr_vec_(intrinsic_vector),
      img_size_(img_size), dist_model_(distortion_model),
      dist_coef_(distortion_coeffs) {}

cv::Size CameraModel::getImageSize() const { return img_size_; }

std::vector<double> CameraModel::getIntrinsicVec() const { return intr_vec_; }

void CameraModelPinholeEqui::init() {
  double fx = intr_vec_[0], fy = intr_vec_[1];
  double cx = intr_vec_[2], cy = intr_vec_[3];
  K_ = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

  D_ = cv::Mat(dist_coef_.size(), 1, CV_64F, &dist_coef_);

  cv::Mat_<double> I = cv::Mat_<double>::eye(3, 3);
  cv::fisheye::estimateNewCameraMatrixForUndistortRectify(K_, D_, img_size_, I,
                                                          newK_, 1);

  new_intr_vec_ = {
      newK_.at<double>(0, 0),
      newK_.at<double>(1, 1),
      newK_.at<double>(0, 2),
      newK_.at<double>(1, 2),
  };

  map1_ = cv::Mat::zeros(img_size_, CV_16SC2);
  map2_ = cv::Mat::zeros(img_size_, CV_16UC1);
  cv::fisheye::initUndistortRectifyMap(K_, D_, I, newK_, img_size_,
                                       map1_.type(), map1_, map2_);
}

CameraModelPinholeEqui::CameraModelPinholeEqui(
    const std::vector<double> &intrinsic_vector, const cv::Size &img_size,
    const std::vector<double> &distortion_coeffs)
    : CameraModel("pinhole", intrinsic_vector, img_size, "equi",
                  distortion_coeffs) {
  init();
}

void CameraModelPinholeEqui::scale(const double &scale_factor) {
  for (double &it : intr_vec_) {
    it *= scale_factor;
  }
  img_size_ = cv::Size2d(img_size_) * scale_factor;
  init();
}

CameraModelPinholeEqui
CameraModelPinholeEqui::scaled(const double &scale_factor) {
  CameraModelPinholeEqui model = *this;
  model.scale(scale_factor);
  return model;
}

void CameraModelPinholeEqui::transpose() {
  std::swap(intr_vec_[0], intr_vec_[1]);
  std::swap(intr_vec_[2], intr_vec_[3]);
  img_size_ = cv::Size(img_size_.height, img_size_.width);
  init();
}

void CameraModelPinholeEqui::undistortImage(const cv::Mat &img,
                                            cv::Mat &un_img) {
  if (img.size() != img_size_) {
    std::cout << "[ERROR]: expected image_size=" << img_size_ << " , bug got "
              << img.size() << std::endl;
  }
  cv::fisheye::undistortImage(img, un_img, K_, D_, newK_, img_size_);
}

void CameraModelPinholeEqui::undistortKeyPoint(
    const std::vector<cv::KeyPoint> &kps, std::vector<cv::KeyPoint> &un_kps) {

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

void CameraModelPinholeEqui::undistort(const cv::Mat &img, cv::Mat &un_img) {
  std::cout << "[WARNING]: This function does not match with "
               "cv::fisheye::undistortPoint， please use "
               "cv::fisheye::undistortImage instead."
            << std::endl;
  cv::remap(img, un_img, map1_, map2_, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
}

std::vector<double> CameraModelPinholeEqui::getNewIntrinsicVec() const {
  return new_intr_vec_;
}

cv::Mat CameraModelPinholeEqui::getD() const { return D_; }

cv::Mat CameraModelPinholeEqui::getK() const { return K_; }

cv::Mat CameraModelPinholeEqui::getNewK() const { return newK_; }
