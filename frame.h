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
#include "ORBVocabulary.h"
#include "camera_model.h"
#include "third_party/DBoW2/DBoW2/BowVector.h"
#include "third_party/DBoW2/DBoW2/FeatureVector.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

/**
 * @brief 普通图像帧
 * @note 使用ORB特征，基于hamming距离的暴力匹配
 */
class Frame {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  using Ptr = std::shared_ptr<Frame>;

  // ctor
  Frame();
  Frame(const cv::Mat &img);
  Frame(const cv::Mat &img, const CameraModel::Ptr &camera_model);
  Frame(const cv::Mat &img, ORBVocabulary *voc);
  /**
   * @brief 与另一帧进行特征点匹配，并根据距离，进行简单筛选
   *
   * @param frame [IN] 另一帧图像
   * @param good_matches [OUT] 好的匹配
   * @param debug_draw [IN] 是否画出匹配图
   * @return int 返回配对的特征点对数，即good_matches.size()
   */
  int matchWith(const Frame::Ptr frame, std::vector<cv::DMatch> &good_matches,
                std::vector<cv::Point2f> &points1,
                std::vector<cv::Point2f> &points2,
                const bool &debug_draw = false);

  /**
   * @brief 切换世界坐标系，将世界坐标系src切换为dst，src和dst之间只差一个旋转
   *        Rcd = Rcs * Rsd = Rcs * Rds.inverse()，Rds用四元数表示q_ds。
   *
   * @param[in] q_ds, 两个世界坐标系src、dst之间的旋转，P_dst = q_ds * P_src
   */
  void rotateWorld(const Eigen::Quaterniond &q_ds);

  // 描述子格式转换
  std::vector<cv::Mat> toDescriptorVector();

  // Compute Bag of Words representation.
  void computeBoW();

  // 相似性得分计算
  float computeScore(const DBoW2::BowVector &v1, const DBoW2::BowVector &v2);

  DBoW2::BowVector getBowVoc();

  // 创建词典
  static void
  createVocabulary(ORBVocabulary &voc, std::string &filename,
                   const std::vector<std::vector<cv::Mat>> &descriptors);

  /**
   * @brief 将地图点投影到当前帧
   *
   * @param[in] x3D 地图点的世界坐标
   * @return cv::Point2f 投影到相机上的像素坐标
   */
  cv::Point2f project(const cv::Mat &x3D);
  cv::Point2f project(const Eigen::Vector3d &mappoint);
  cv::Point2f project(const double &x, const double &y, const double &z);

  /**
   * @brief 检查地图点在当前相机位姿下，深度是否为正
   *
   * @param[in] x3D 地图点的世界坐标
   * @return true 深度为正，否则为负
   */
  bool checkDepthValid(const cv::Mat &x3D);

  // 特征点 access
  std::vector<cv::KeyPoint> getUnKeyPoints() const;
  cv::KeyPoint getUnKeyPoints(const int &keypoint_idx) const;

  // 地图点索引 access
  std::vector<int> getMappointId() const;
  int getMappointId(const int &keypoint_idx) const;
  void setMappointIdx(const int &keypoint_idx, const int &mappoint_idx);

  // 位姿相关 access
  Eigen::Matrix3d getEigenRot();
  Eigen::Vector3d getEigenTrans();
  Eigen::Matrix3d getEigenRotWc();
  Eigen::Vector3d getEigenTransWc();
  cv::Mat getPose();
  void setPose(const Eigen::Matrix4d &mat);
  void setPose(const Eigen::Matrix3d &R, const Eigen::Vector3d &t);
  void setPose(const cv::Mat &mat);
  void setPose(const cv::Mat &R, const cv::Mat &t);

  // frame id access
  size_t getFrameId() const;

  // 获取去畸变后的K
  Eigen::Matrix3d getEigenNewK() const;

  /**
   * @brief 获取投影矩阵 K[R, t]，大小为3行4列
   */
  cv::Mat getProjectionMatrix();

  // debug
  void debugDraw(const double &scale_image = 1.0);
  void debugPrintPose();

  // 特征点、描述符、匹配相关
  static cv::Ptr<cv::FeatureDetector> detector_;
  static cv::Ptr<cv::DescriptorExtractor> extrator_;
  static cv::Ptr<cv::DescriptorMatcher> matcher_;

  // 相机模型（包含畸变模型）
  CameraModel::Ptr camera_model_;

  // private:
public:
  // 图片，特征点，描述符
  cv::Mat img_;
  cv::Mat un_img_; // 去畸变后的图像
  std::vector<cv::KeyPoint> keypoints_;
  std::vector<cv::KeyPoint> un_keypoints_; // 去畸变后的特征点
  cv::Mat descriptors_;

  // 特征点对应的3D空间点的id
  std::vector<int> mappoints_id_;

  // 代码中Tcw，表示位姿T^w_c。
  // 假设：点在相机坐标系下的值P_c，点在世界坐标系下的坐标值P_w，则
  // P_c = T^w_c * P_w；代码中即：Pc = Tcw * Pw。
  std::mutex mutex_pose_;
  cv::Mat Tcw_;
  cv::Mat Rcw_;
  cv::Mat tcw_;

  // id
  size_t frame_id_;
  static size_t total_frame_cnt_;

  // Vocabulary used for relocalization.
  ORBVocabulary *pORBvocabulary;

  // Bag of Words Vector structures.
  DBoW2::BowVector bow_vec_;
  DBoW2::FeatureVector feat_vec_;

  // 相似性得分
  float score_;

private:
  /**
   * @brief 仅供构造函数使用，进行初始化
   */
  void init();

  /**
   * @brief 判断特征点是否位于图像中心
   */
  bool isCentralKp(const cv::KeyPoint &kp,
                   const double &half_center_factor = 0.2);
};
