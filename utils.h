/**
 * @file utils.h
 * @author xiaotaw (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2021-02-22
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

/**
 * @brief 统计数据中的最大值、最小值、平均值，并输出到屏幕
 */
void statistic(const std::vector<double> &data,
                      const std::string &name);

void calAveStddev(const std::vector<double> &data, double &ave, double &stddev);

void calAveStddev(const std::vector<cv::Point2f> &data, cv::Point2f &ave, cv::Point2f &stddev);

/**
 * @brief 将四元数转换成字符串方便debug输出，w在前。
 * 
 * @param[in] q
 * @return std::string 
 */
std::string toString(const Eigen::Quaterniond &q);

std::string toString(const Eigen::Vector3d &v);