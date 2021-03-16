/**
 * @file:  gridmatcher.h
 * @author: ipsg
 * @brief:
 * @version:
 * @date:  2021-03-11
 * @copyright: Copyright (c) 2021
 */

#pragma once

#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;

// 8 possible rotation and each one is 3 X 3

const int THRESH_FACTOR = 6;

const int mRotationPatterns[8][9] = {1, 2, 3, 4, 5, 6, 7, 8, 9,

                                     4, 1, 2, 7, 5, 3, 8, 9, 6,

                                     7, 4, 1, 8, 5, 2, 9, 6, 3,

                                     8, 7, 4, 9, 5, 1, 6, 3, 2,

                                     9, 8, 7, 6, 5, 4, 3, 2, 1,

                                     6, 9, 8, 3, 5, 7, 2, 1, 4,

                                     3, 6, 9, 2, 5, 8, 1, 4, 7,

                                     2, 3, 6, 1, 5, 9, 4, 7, 8};

// 5 level scales
const double mScaleRatios[5] = {1.0, 1.0 / 2, 1.0 / sqrt(2.0), sqrt(2.0), 2.0};

class Gridmatcher {
 public:
  using Ptr = std::shared_ptr<Gridmatcher>;
  Gridmatcher(){};
  // OpenCV Keypoints & Correspond Image Size & Nearest Neighbor Matches
  Gridmatcher(const vector<cv::KeyPoint> &vkp1, const cv::Size size1,
              const vector<cv::KeyPoint> &vkp2, const cv::Size size2,
              const vector<cv::DMatch> &vDMatches) {
    // Input initialize
    NormalizePoints(vkp1, size1, mvP1);
    NormalizePoints(vkp2, size2, mvP2);
    mNumberMatches = vDMatches.size();
    ConvertMatches(vDMatches, mvMatches);
  };
  ~Gridmatcher(){};

 private:
  // Normalized Points
  vector<cv::Point2f> mvP1, mvP2;

  // Number of Matches
  size_t mNumberMatches;

  // Grid Size
  cv::Size mGridSizeLeft, mGridSizeRight;
  int mGridNumberLeft;
  int mGridNumberRight;

  // x	  : left grid idx
  // y      :  right grid idx
  // value  : how many matches from idx_left to idx_right
  cv::Mat mMotionStatistics;

  vector<int> mNumberPointsInPerCellLeft;

  // Inldex  : grid_idx_left
  // Value   : grid_idx_right
  vector<int> mCellPairs;

  // Every Matches has a cell-pair
  // first  : grid_idx_left
  // second : grid_idx_right
  vector<pair<int, int> > mvMatchPairs;

  // Inlier Mask for output
  vector<bool> mvbInlierMask;

  //
  cv::Mat mGridNeighborLeft;
  cv::Mat mGridNeighborRight;

 public:
  // Matches
  vector<pair<int, int> > mvMatches;

  // Get Inlier Mask
  // Return number of inliers
  int GetInlierMask(vector<bool> &vbInliers, const cv::Size Gridsize,
                    bool WithScale = false, bool WithRotation = false);
  bool init(const vector<cv::KeyPoint> &vkp1, const cv::Size size1,
            const vector<cv::KeyPoint> &vkp2, const cv::Size size2,
            const vector<int> &vnMatches12);
  void imresize(cv::Mat &src, int height);

 private:
  // Normalize Key Points to Range(0 - 1)
  void NormalizePoints(const vector<cv::KeyPoint> &kp, const cv::Size &size,
                       vector<cv::Point2f> &npts);
  // Convert OpenCV DMatch to Match (pair<int, int>)

  void ConvertMatches(const vector<cv::DMatch> &vDMatches,
                      vector<pair<int, int> > &vMatches);

  void ConvertMatches(const vector<int> &vnMatches12);

  int GetGridIndexLeft(const cv::Point2f &pt, int type);

  int GetGridIndexRight(const cv::Point2f &pt);

  // Assign Matches to Cell Pairs
  void AssignMatchPairs(int GridType);

  // Verify Cell Pairs
  void VerifyCellPairs(int RotationType);

  // Get Neighbor 9
  vector<int> GetNB9(const int idx, const cv::Size &GridSize);

  // 400*9 20*20
  void InitalizeNiehbors(cv::Mat &neighbor, const cv::Size &GridSize);

  void SetScale(int Scale);

  // Run
  int run(int RotationType);
};