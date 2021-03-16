/*
 * @file:
 * @Author: Dongying (yudong2817@sina.com)
 * @brief:
 * @version:
 * @date:  Do not edit
 * @copyright: Copyright (c) 2021
 */
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

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

/**
 * @brief 统计数据中的最大值、最小值、平均值，并输出到屏幕
 */
void statistic(const std::vector<double> &data, const std::string &name);

void statistic(const std::vector<Eigen::Vector3d,
                                 Eigen::aligned_allocator<Eigen::Vector3d>> &data,
               const std::string &name);

void histogram(const std::vector<double> &data, const double &min_v = 0,
               const double &max_v = 2, const double &interval = 0.1,
               const std::string &prefix_str = "histogram");

void calAveStddev(const std::vector<double> &data, double &ave, double &stddev);

/**
 * @brief 计算均值和方差
 *
 * @param[in] data
 * @param[out] ave
 * @param[out] stddev
 * @param[in] abs 是否对data中的每个元素取绝对值
 */
void calAveStddev(const std::vector<cv::Point2f> &data, cv::Point2f &ave,
                  cv::Point2f &stddev, const bool &abs = false);

/**
 * @brief 将四元数转换成字符串方便debug输出，w在前。
 *
 * @param[in] q
 * @return std::string
 */
std::string toString(const Eigen::Quaterniond &q);

std::string toString(const Eigen::Vector3d &v);