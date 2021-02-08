/**
 * @file initializer.cc
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2021-02-08
 *
 * @copyright Copyright (c) 2021
 *
 */
#include "initializer.h"

/**
 * @brief 统计数据中的最大值、最小值、平均值，并输出到屏幕
 */
static void statistic(const std::vector<double> &data,
                      const std::string &name) {
  int cnt = 0;
  double ave = 0, min_v = 0, max_v = 0;

  if (!data.empty()) {
    cnt = data.size();
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    auto mm = std::minmax_element(data.begin(), data.end());
    ave = sum / data.size();
    min_v = *mm.first;
    max_v = *mm.second;
  }

  std::cout << "[INFO]: " << name << " cnt: " << cnt << " min: " << min_v
            << " max: " << max_v << " ave: " << ave << std::endl;
}

bool Initializer::initialize(Frame::Ptr frame1, Frame::Ptr frame2,
                             const cv::Mat &K, cv::Mat &R, cv::Mat &t) {
  // 1. 匹配两帧图像的特征点，计算单应矩阵
  std::vector<cv::DMatch> good_matches;
  frame1->matchWith(frame2, good_matches, true);

  points1_.clear();
  points2_.clear();
  for (const cv::DMatch &m : good_matches) {
    points1_.emplace_back(frame1->keypoints_[m.queryIdx].pt);
    points2_.emplace_back(frame2->keypoints_[m.trainIdx].pt);
  }

  // todo: H12 or H21 ???
  // todo: 增大误差阈值，因为没有矫正畸变参数
  ransac_status_.clear();
  cv::Mat H =
      cv::findHomography(points1_, points2_, ransac_status_, cv::RANSAC);
  int h_inliers = std::accumulate(ransac_status_.begin(), ransac_status_.end(),
                                  0, [](int c1, int c2) { return c1 + c2; });

  std::cout << "[INFO]: Find H inliers: " << h_inliers << std::endl;

  // 2. 利用单应矩阵计算R和t，挑选出正确的R和t
  std::vector<cv::Mat> Rs, ts, normals;
  cv::decomposeHomographyMat(H, K, Rs, ts, normals);

  cv::Mat R_h, t_h, x3D_mean;
  int x3D_inliers = 0;
  for (int i = 0; i < Rs.size(); ++i) {
    cv::Mat x3D_sum = cv::Mat::zeros(3, 1, CV_64F);
    // 检查R，t，n，统计内点数目
    int inliers = checkRtn(Rs[i], ts[i], normals[i], K, x3D_sum);

    if (inliers > x3D_inliers) {
      x3D_inliers = inliers;
      x3D_mean = x3D_sum / x3D_inliers;
      R_h = Rs[i];
      t_h = ts[i];
    }
  }
  std::cout << "[INFO]: Recover Rt, inliers: " << x3D_inliers << std::endl;

  if (x3D_inliers < x3D_inliers_threshold_) {
    std::cout << "[WARNING]: Initialize failed for too few x3D inliers"
              << std::endl;
    return false;
  } else {
    std::cout << "[INFO]: x3D_mean: " << x3D_mean.at<double>(0) << " "
              << x3D_mean.at<double>(1) << " " << x3D_mean.at<double>(2) << " "
              << std::endl;
  }

  // 3. 利用天车高度9米的先验，得到尺度，初始化地图点。
  double scale = 9 / x3D_mean.at<double>(2);
  R = R_h;
  t = t_h * scale;

  std::cout << "[INFO]: t: " << t.at<double>(0) << " " << t.at<double>(1) << " "
            << t.at<double>(2) << std::endl;
  // cv::Mat t_normed = t / cv::norm(t);
  // std::cout << "[INFO]: normed t: " << t_normed.at<double>(0) << " "
  //           << t_normed.at<double>(1) << " " << t_normed.at<double>(2)
  //           << " " << std::endl;
}

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
int Initializer::checkRtn(const cv::Mat &R, const cv::Mat &t, const cv::Mat &n,
                          const cv::Mat &K, cv::Mat &x3D_sum) {

  // 因为相机是俯视地面，法向量必须是大致沿z轴的（z轴分量绝对值最大）
  if (std::fabs(n.at<double>(2, 0)) <= std::fabs(n.at<double>(0, 0)) ||
      std::fabs(n.at<double>(2, 0)) <= std::fabs(n.at<double>(1, 0))) {
    // return 0;
  }

  // 计算地图点，地图点在两个相机坐标系下的z值必须都为正

  // 在相机位置1参考系中，两相机光心
  cv::Mat O1 = cv::Mat::zeros(3, 1, CV_64F);
  cv::Mat O2 = -R.t() * t;

  // debug message
  std::vector<double> e1s, e2s, cos_thetas;

  int x3D_cnt = 0;
  for (int j = 0; j < ransac_status_.size(); ++j) {
    // 如果不是计算H矩阵时的内点，则不参与计算
    if (!ransac_status_[j]) {
      continue;
    }
    // 空间点在相机位置1参考系中的坐标
    auto pt1 = points1_[j];
    auto pt2 = points2_[j];
    cv::Mat x3D_C1;
    triangulate(pt1, pt2, R, t, K, x3D_C1);

    // 空间点在相机位置2参考系中的坐标
    cv::Mat x3D_C2 = R * x3D_C1 + t;

    // 判断是否为有效的地图点
    if (!(std::isfinite(x3D_C1.at<double>(0)) &&
          std::isfinite(x3D_C1.at<double>(1)) &&
          std::isfinite(x3D_C1.at<double>(2)) && x3D_C1.at<double>(2) > 0 &&
          x3D_C2.at<double>(2) > 0)) {
      continue;
    }

    // 如何使用夹角？
    cv::Mat N1 = x3D_C1 - O1;
    cv::Mat N2 = x3D_C1 - O2;
    double cos_theta = N1.dot(N1) / (cv::norm(N1) * cv::norm(N2));
    cos_thetas.emplace_back(cos_theta);

    // 计算空间点的投影误差，误差平方值应当在允许范围之内
    auto proj_pt1 = project(x3D_C1, K);
    double e1 = (pt1.x - proj_pt1.x) * (pt1.x - proj_pt1.x) +
                (pt1.y - proj_pt1.y) * (pt1.y - proj_pt1.y);
    auto proj_pt2 = project(x3D_C2, K);
    double e2 = (pt2.x - proj_pt2.x) * (pt2.x - proj_pt2.x) +
                (pt2.y - proj_pt2.y) * (pt2.y - proj_pt2.y);

    e1s.emplace_back(e1);
    e2s.emplace_back(e2);

    if (e1 > square_projection_error_threshold_) {
      continue;
    }
    if (e2 > square_projection_error_threshold_) {
      continue;
    }

    x3D_sum += x3D_C1;
    x3D_cnt++;
  }

  statistic(cos_thetas, "  cos_theta");
  statistic(e1s, "  e1");
  statistic(e2s, "  e2");

  return x3D_cnt;
}

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
void Initializer::triangulate(const cv::Point2f &kp1, const cv::Point2f &kp2,
                              const cv::Mat &R, const cv::Mat &t,
                              const cv::Mat K, cv::Mat &x3D) {

  // 相机1投影矩阵: K[I|0]
  cv::Mat P1(3, 4, CV_64F, cv::Scalar(0));
  cv::Mat I = cv::Mat::eye(3, 3, CV_64F);
  I.copyTo(P1.rowRange(0, 3).colRange(0, 3));
  P1 = K * P1;
  // 相机2投影矩阵: K[R|t];
  cv::Mat P2(3, 4, CV_64F, cv::Scalar(0));
  R.copyTo(P2.rowRange(0, 3).colRange(0, 3));
  t.copyTo(P2.rowRange(0, 3).col(3));
  P2 = K * P2;

  // 三角化
  cv::Mat A(4, 4, CV_64F);

  A.row(0) = kp1.x * P1.row(2) - P1.row(0);
  A.row(1) = kp1.y * P1.row(2) - P1.row(1);
  A.row(2) = kp2.x * P2.row(2) - P2.row(0);
  A.row(3) = kp2.y * P2.row(2) - P2.row(1);

  cv::Mat u, w, vt;
  cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
  x3D = vt.row(3).t();
  x3D = x3D.rowRange(0, 3) / x3D.at<double>(3);
}

/**
 * @brief 讲空间点x3D，投影到像素坐标
 *
 * @param x3D 空间点坐标
 * @param K 相机内参
 * @return cv::Point2f 像素坐标
 */
cv::Point2f Initializer::project(const cv::Mat &x3D, const cv::Mat K) {
  double fx = K.at<double>(0, 0);
  double fy = K.at<double>(1, 1);
  double cx = K.at<double>(0, 2);
  double cy = K.at<double>(1, 2);
  double X = x3D.at<double>(0);
  double Y = x3D.at<double>(1);
  double Z = x3D.at<double>(2);
  double x = fx * X / Z + cx;
  double y = fy * Y / Z + cy;
  return cv::Point2f(x, y);
}
