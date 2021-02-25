/**
 * @file map.h
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2021-02-09
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once
#include "frame.h"
#include "mappoint.h"
#include <mutex>
#include <numeric> // for std::accumulate
#include <vector>

class Map {

public:
  using Ptr = std::shared_ptr<Map>;

  Map() {}

  /**
   * @brief 清空地图
   */
  void clear();

  /**
   * @brief 跟踪新的一帧
   */
  bool trackNewFrame(Frame::Ptr frame);

  /**
   * @brief 利用两帧进行单目初始化，frame1和frame2中的相机内参应当相同。重载
   * Map::Ptr initialize(Frame::Ptr frame1, Frame::Ptr frame2, const cv::Mat &K)
   *
   * @param[in] frame1
   * @param[in] frame2
   * @return Map::Ptr 返回初始化成功的地图，初始化失败则返回nullptr
   */
  bool initialize(Frame::Ptr frame1, Frame::Ptr frame2);

  bool checkInitialized();

  /**
   * @brief 旋转相机的pose，使得每一帧的旋转都相同，且平移量在X轴
   */
  void rotateFrameToXTranslation();

  // debug
  void printMap() const;

public:
  std::mutex mutex_mappoints_;
  std::vector<MapPoint::Ptr> mappoints_;

  std::mutex mutex_frames_;
  std::vector<Frame::Ptr> frames_;

private:
  /**
   * @brief 三角测量计算地图点在世界坐标系中的坐标 (参照ORB_SLAM2，稍有改动)
   *
   * @param kp1 地图点在图像1中对应特征点
   * @param kp2 地图点在图像2中对应特征点
   * @param P1 图像1对应的投影矩阵，即K[R1,t1]
   * @param P2 图像2对应的投影矩阵，即K[R2,t2]
   * @param[out] x3D 世界坐标系中的地图点
   */
  void triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2,
                   const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);

  void triangulate(const cv::Point2f &pt1, const cv::Point2f &pt2,
                   const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);

  /**
   * @brief 检查R, t, n是否正确(目前没有检查法向量)
   *
   * @param R 旋转
   * @param t 平移
   * @param n 法向量
   * @param K 相机内参
   * @param points1 特征点在图像1中的坐标
   * @param points2 特征点在图像2中的坐标
   * @param[in out] mask 标志着特征点是否被用于计算Rt
   * @param mappoints 空间点（内点）
   * @param verbose
   * @return int 内点数目
   */
  int checkRtn(const cv::Mat &R, const cv::Mat &t, const cv::Mat &n,
               const cv::Mat &K, std::vector<cv::Point2f> points1,
               std::vector<cv::Point2f> points2, std::vector<uchar> &mask,
               std::vector<MapPoint::Ptr> &mappoints, bool verbose = false);

  /**
   * @brief 将空间点x3D，投影到像素坐标
   *
   * @param x3D 空间点坐标
   * @param K 相机内参
   * @return cv::Point2f 像素坐标
   */
  cv::Point2f project(const cv::Mat &x3D, const cv::Mat K);

  // 初始化时，重投影地图点时，允许的误差最大值的平方(ORBSLAM2值为4)
  double square_projection_error_threshold_ = 10;
  // 初始化时，至少应当有100个地图点
  int x3D_inliers_threshold_ = 50;
};
