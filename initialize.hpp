#include "frame.hpp"
#include <numeric>

void triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2,
                 const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D) {
  cv::Mat A(4, 4, CV_64F);

  A.row(0) = kp1.pt.x * P1.row(2) - P1.row(0);
  A.row(1) = kp1.pt.y * P1.row(2) - P1.row(1);
  A.row(2) = kp2.pt.x * P2.row(2) - P2.row(0);
  A.row(3) = kp2.pt.y * P2.row(2) - P2.row(1);

  cv::Mat u, w, vt;
  cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
  x3D = vt.row(3).t();
  x3D = x3D.rowRange(0, 3) / x3D.at<double>(3);
}

bool initialize(cv::Ptr<Frame> frame1, cv::Ptr<Frame> frame2, const cv::Mat &K,
                cv::Mat &R, cv::Mat &t) {

  std::vector<cv::DMatch> good_matches;
  frame1->matchWith(frame2, good_matches, true);

  std::vector<cv::Point2f> points1, points2;
  std::vector<cv::KeyPoint> key_points1, key_points2;
  for (const cv::DMatch &m : good_matches) {
    key_points1.emplace_back(frame1->keypoints_[m.queryIdx]);
    key_points2.emplace_back(frame2->keypoints_[m.trainIdx]);
    points1.emplace_back(frame1->keypoints_[m.queryIdx].pt);
    points2.emplace_back(frame2->keypoints_[m.trainIdx].pt);
  }

  std::cout << "[INFO]: good match: " << good_matches.size() << std::endl;

  std::vector<uchar> ransac_status, mask;

  // E = findEssentialMat(points1, points2, cameraMatrix, RANSAC, 0.999, 1.0,
  // mask); recoverPose(E, points1, points2, cameraMatrix, R, t, mask);

  // recover pose from E
  cv::Mat F =
      cv::findFundamentalMat(points1, points2, ransac_status, cv::RANSAC);

  cv::Mat E = K.t() * F * K;
  // std::cout << "E: " << std::endl << E << std::endl;

  cv::Mat R_e, t_e;
  int inliers = cv::recoverPose(E, points1, points2, K, R_e, t_e, mask);
  int cnt = 0;
  for (const uchar &c : mask) {
    if (int(c) > 0) {
      cnt++;
    }
  }
  // std::cout << "[INFO]: E inliers: " << inliers << ", " << cnt << std::endl;

  // recover pose from H
  cv::Mat H = cv::findHomography(points1, points2, ransac_status, cv::RANSAC);
  cnt = 0;
  for (const char c : ransac_status) {
    if (int(c) > 0) {
      cnt++;
    }
  }
  std::cout << "[INFO]: H inliers: " << cnt << std::endl;

  std::vector<cv::Mat> Rs, ts, normals;
  cv::decomposeHomographyMat(H, K, Rs, ts, normals);

  // check R,t
  cv::Mat R_h, t_h;
  cv::Mat mean_x3D;
  int max_x3D_cnt = 0;
  for (int i = 0; i < Rs.size(); ++i) {
    cv::Mat n = normals[i];
    double nx = n.at<double>(0, 0);
    double ny = n.at<double>(1, 0);
    double nz = n.at<double>(2, 0);

    // 因为相机是俯视地面，法向量必须是大致沿z轴的（z轴分量绝对值最大）
    if (std::fabs(nz) <= std::fabs(nx) || std::fabs(nz) <= std::fabs(ny)) {
      continue;
    }

    // projection matrix of camera1: K[I|0]
    cv::Mat P1(3, 4, CV_64F, cv::Scalar(0));
    cv::Mat I = cv::Mat::eye(3, 3, CV_64F);
    I.copyTo(P1.rowRange(0, 3).colRange(0, 3));
    P1 = K * P1;

    cv::Mat O1 = cv::Mat::zeros(3, 1, CV_64F);

    // projection Matrix of camera2: K[R|t];
    cv::Mat P2(3, 4, CV_64F, cv::Scalar(0));
    Rs[i].copyTo(P2.rowRange(0, 3).colRange(0, 3));
    ts[i].copyTo(P2.rowRange(0, 3).col(3));
    P2 = K * P2;

    cv::Mat O2 = -Rs[i].t() * ts[i];

    cv::Mat x3D_sum = cv::Mat::zeros(3, 1, CV_64F);
    int x3D_cnt = 0;
    for (int j = 0; j < key_points1.size(); ++j) {
      if (!ransac_status[j]) {
        continue;
      }
      cv::Mat x3D;
      triangulate(key_points1[j], key_points2[j], P1, P2, x3D);

      // 判断是否为有效的地图点
      if (std::isfinite(x3D.at<double>(0)) &&
          std::isfinite(x3D.at<double>(1)) &&
          std::isfinite(x3D.at<double>(2))) {
        cv::Mat N1 = x3D - O1;
        cv::Mat N2 = x3D - O2;
        double cos_parallex = N1.dot(N1) / (cv::norm(N1) * cv::norm(N2));

        cv::Mat x3D_C2 = Rs[i] * x3D + ts[i];

        // 在两个相机坐标系下，z分量均大于零
        if (x3D.at<double>(2) >= 0 && x3D_C2.at<double>(2) >= 0) {
          x3D_sum += x3D;
          x3D_cnt++;
        }
      }
    }
    if (x3D_cnt > max_x3D_cnt) {
      max_x3D_cnt = x3D_cnt;
      R_h = Rs[i];
      t_h = ts[i];
      mean_x3D = x3D_sum / x3D_cnt;
    }
  }

  // R t from H
  if (R_h.empty() && t_h.empty()) {
    R = R_e;
    t = t_e;
    // std::cout << "R_e: " << std::endl << R_e << std::endl;
    // std::cout << "t_e: " << std::endl << t_e << std::endl;
  } else {
    std::cout << "x3D size: " << max_x3D_cnt << " mean: " << std::endl
              << mean_x3D << std::endl;
    double scale = 9 / mean_x3D.at<double>(2);
    R = R_h;
    t = t_h * scale;
    // std::cout << "R_h: " << std::endl << R_h << std::endl;
    cv::Mat t_normed = t / cv::norm(t);
    std::cout << "t_h: " << std::endl
              << t.at<double>(0) << " " << t.at<double>(1) << " "
              << t.at<double>(2) << "   ";
    std::cout << t_normed.at<double>(0) << " " << t_normed.at<double>(1) << " "
              << t_normed.at<double>(2) << " " << std::endl;
  }

  return true;
}
