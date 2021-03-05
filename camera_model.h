/**
 * @file camera_model.h
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2021-02-23
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class CameraModel {
public:
  using Ptr = std::shared_ptr<CameraModel>;

  CameraModel();

  // 无畸变
  CameraModel(const std::string &camera_model,
              const std::vector<double> &intrinsic_vector,
              const cv::Size &img_size);

  // 有畸变
  CameraModel(const std::string &camera_model,
              const std::vector<double> &intrinsic_vector,
              const cv::Size &img_size, const std::string &distortion_model,
              const std::vector<double> &distortion_coeffs);

  virtual void scale(const double &scale_factor) = 0;
  virtual void transpose() = 0;

  virtual void undistortImage(const cv::Mat &img, cv::Mat &un_img) = 0;
  virtual void undistortKeyPoint(const std::vector<cv::KeyPoint> &kps,
                                 std::vector<cv::KeyPoint> &un_kps) = 0;
  virtual void undistort(const cv::Mat &img, cv::Mat &un_img) = 0;

  cv::Size getImageSize() const;

  std::vector<double> getIntrinsicVec() const;
  virtual std::vector<double> getNewIntrinsicVec() const = 0;

  virtual cv::Mat getD() const = 0;

  virtual cv::Mat getK() const = 0;

  virtual cv::Mat getNewK() const = 0;

protected:
  std::string camera_model_;
  std::vector<double> intr_vec_;
  cv::Size img_size_;
  std::string dist_model_;
  std::vector<double> dist_coef_;
};

/**
 * @brief Kalibr中的pinhole-equi即opencv中的fisheye
 */
class CameraModelPinholeEqui : public CameraModel {
public:
  using Ptr = std::shared_ptr<CameraModelPinholeEqui>;

  CameraModelPinholeEqui(const std::vector<double> &intrinsic_vector,
                         const cv::Size &img_size,
                         const std::vector<double> &distortion_coeffs);
  
  CameraModelPinholeEqui(const std::string &kalibr_camchain_yaml);

  void scale(const double &scale_factor) override;

  void transpose() override;

  void undistortImage(const cv::Mat &img, cv::Mat &un_img) override;

  void undistortKeyPoint(const std::vector<cv::KeyPoint> &kps,
                         std::vector<cv::KeyPoint> &un_kps) override;

  void undistort(const cv::Mat &img, cv::Mat &un_img);

  std::vector<double> getNewIntrinsicVec() const override;

  cv::Mat getD() const override;

  cv::Mat getK() const override;

  cv::Mat getNewK() const override;

private:
  cv::Mat K_, newK_, D_;
  cv::Mat map1_, map2_;
  std::vector<double> new_intr_vec_;

  // scale和scaled函数，需要调整和优化，scaled函数暂不对外
  CameraModelPinholeEqui scaled(const double &scale_factor);

  void init();
};
