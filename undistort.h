/**
 * @file undistort.h
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2021-02-23
 *
 * @copyright Copyright (c) 2021
 *
 */

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

class UndistorterFisheye {
public:
  UndistorterFisheye(const int &cols, const int &rows, const double &fx,
                     const double &fy, const double &cx, const double &cy,
                     const double &k1, const double &k2, const double &k3,
                     const double &k4);

  cv::Mat getNewK() const;

  void undistort(const cv::Mat &img, cv::Mat &img_undistorted);

private:
  cv::Size img_size_; 
  cv::Mat K_, newK_, map1_, map2_;
};