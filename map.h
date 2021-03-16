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
#include <mutex>
#include <numeric>  // for std::accumulate
#include <vector>
#include "frame.h"
#include "mappoint.h"
#include "optimizer.h"

class Map {
 public:
  using Ptr = std::shared_ptr<Map>;

  Map();

  Map(const int &sliding_window_local, const int &sliding_window_global);


  ~Map();
  /**
   * @brief 跟踪新的一帧
   */
  int trackNewFrameByKeyFrame(Frame::Ptr frame, const double &debug_draw = -1);

  /**
   * @brief 利用两帧进行单目初始化
   * @note frame1和frame2中的相机内参应当相同。
   *
   * @param[in] frame1
   * @param[in] frame2
   * @return Map::Ptr 返回初始化成功的地图，初始化失败则返回nullptr
   */
  bool initialize(const Frame::Ptr &frame1, const Frame::Ptr &frame2,
                  const double &debug_draw = -1);

  bool checkInitialized();

  /**
   * 地图点相关
   */
  void insertMapPoint(const MapPoint::Ptr &mp);

  size_t removeMapPointById(const size_t &mp_idx);

  MapPoint::Ptr getMapPointById(const int &mp_idx);

  std::vector<MapPoint::Ptr> getMapPoints();

  /**
   * @brief 地图点的坐标的平均值
   */
  Eigen::Vector3d getAveMapPoint();

  double getScale();

  void setScale(const double &scale);

  /**
   * @brief 插入新的一帧，更新recent_frames_
   */
  void insertRecentFrame(const Frame::Ptr &frame);

  Frame::Ptr getLastFrame();

  void insertKeyFrame(const Frame::Ptr &frame);

  Frame::Ptr getLastKeyFrame();

  bool checkIsNewKeyFrame(Frame::Ptr &frame);

  /**
   * @brief 获取与新的一帧相关的帧、地图点、观测数据，构建优化器，
   * @note 该帧相关的地图点是否需要优化？
   *
   * @param[in out] frame
   * @return G2oOptimizer::Ptr
   */
  G2oOptimizer::Ptr buildG2oOptForFrame(const Frame::Ptr frame);

  /**
   * @brief 获取G2oInput，用于关键帧滑窗优化
   *
   * @return G2oOptimizer::Ptr
   */
  G2oOptimizer::Ptr buildG2oOptKeyFrameBa();

  /**
   * @brief 清空地图
   */
  void clear();

  void clearRecentFrames();

  /**
   * @brief 常量，天车高度
   *
   */
  const static double kCraneHeight;

  /**
   * @brief 统计输出当前map的信息，包括每一帧的pose，地图点的坐标均值，等等
   */
  void debugPrintMap();

  /**
   * @brief 关键帧的地图点坐标平均值，用于debug
   */
  Eigen::Vector3d ave_kf_mp_;

  /**
   * @brief 获取关键帧数据库大小
   */
  int getKeyframesSize();

  /**
   * @brief 获取关键帧数据
   */
  std::map<size_t, Frame::Ptr> getKeyframes() const;

  /**
   * @brief设置偏移数据
   */
  void setOffset(const double &offset);

  /**
   * @brief 获取数据
   */
  double getOffset();

  /**
   * @brief 由关键帧数据计算偏移量
   */
  void calculateOffset();

  /**
  * @brief 设置系统是否初始化状态
  */
  void setInitializeStatus(const bool &status);

  void releaseLastKeyframeimg();

 private:
  std::mutex mutex_recent_frames_;
  std::map<size_t, Frame::Ptr> recent_frames_;
  int max_recent_frames_ = 50;

  std::mutex mutex_keyframes_;
  std::map<size_t, Frame::Ptr> keyframes_;

  std::mutex mutex_mappoints_;
  std::map<size_t, MapPoint::Ptr> mappoints_;

  bool is_initialized_ = false;

  std::mutex mutex_scale_;
  double scale_ = 1.0;

  // 相对位置与绝对位置之间的偏移量
  double offset_ = 0.0;

  // 图像帧与关键帧x方向上的统计值差
  float diff_ave_ = 0.0;

  //设置优化窗口大小
  int sliding_window_local_ = 5;
  int sliding_window_global_ = 5;

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
   * @brief 计算像素误差平方和
   *
   * @param[in] uv_error 像素误差
   * @return float 范围误差平方和
   */
  float squareUvError(const cv::Point2f &uv_error);

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
