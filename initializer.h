/**
 * @file initializer.h
 * @author xiaotaw (you@domain.com)
 * @brief 单目初始化
 * @version 0.1
 * @date 2021-02-08
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once
#include "frame.h"
#include "map.h"
#include <numeric> // for std::accumulate

/**
 * @brief 单目初始化步骤：
 *        1. 匹配两帧图像的特征点，计算单应矩阵
 *        2. 利用单应矩阵计算R和t，挑选出正确的R和t
 *        3. 利用天车高度9米的先验，得到尺度，初始化地图点。
 *
 * @todo 将initializer整合成map的一个方法成员 initializeMono。
 */
class Initializer {
public:
  using Ptr = std::shared_ptr<Initializer>;

  /**
   * @brief 利用两帧进行单目初始化，frame1和frame2中的相机内参应当相同。重载
   * Map::Ptr initialize(Frame::Ptr frame1, Frame::Ptr frame2, const cv::Mat &K)
   *
   * @param[in] frame1
   * @param[in] frame2
   * @return Map::Ptr 返回初始化成功的地图，初始化失败则返回nullptr
   */
  Map::Ptr initialize(Frame::Ptr frame1, Frame::Ptr frame2);

  /**
   * @brief 利用两帧进行单目初始化
   *
   * @param frame1 [IN]
   * @param frame2 [IN]
   * @param K [IN]
   * @return 返回初始化成功的地图，初始化失败则返回nullptr
   */
  Map::Ptr initialize(Frame::Ptr frame1, Frame::Ptr frame2, const cv::Mat &K);

  /**
   * @brief 检查R, t, n是否正确
   *
   * @param R 旋转
   * @param t 平移
   * @param n 法向量
   * @param K 相机内参
   * @param x3D_sum 空间点（内点）的坐标值总和（在相机位置1坐标系中）
   * @return int 内点数目
   */
  int checkRtn(const cv::Mat &R, const cv::Mat &t, const cv::Mat &n,
               const cv::Mat &K, cv::Mat &x3D_sum,
               std::vector<uchar> &inlier_mask,
               std::vector<MapPoint::Ptr> &x3Ds);

  /**
   * @brief 三角化求空间点在相机1中的坐标
   *
   * @param kp1 空间点投影在在图像1中的坐标
   * @param kp2 空间点投影在在图像2中的坐标
   * @param R 旋转 R^1_2, i.e. n_2 = R^1_2 * n_1;
   * @param t 平移，与R一致
   * @param K 相机内参
   * @param x3D 空间点坐标
   */
  void triangulate(const cv::Point2f &kp1, const cv::Point2f &kp2,
                   const cv::Mat &R, const cv::Mat &t, const cv::Mat K,
                   cv::Mat &x3D);

  /**
   * @brief 将空间点x3D，投影到像素坐标
   *
   * @param x3D 空间点坐标
   * @param K 相机内参
   * @return cv::Point2f 像素坐标
   */
  cv::Point2f project(const cv::Mat &x3D, const cv::Mat K);

public:
  // 匹配上的特征点，在两帧图像中的坐标
  std::vector<cv::Point2f> points1_, points2_;
  std::vector<uchar> ransac_status_;

  // 初始化时，重投影地图点时，允许的误差最大值的平方(ORBSLAM2值为4)
  double square_projection_error_threshold_ = 4;
  // 初始化时，至少应当有100个地图点
  int x3D_inliers_threshold_ = 100;
};