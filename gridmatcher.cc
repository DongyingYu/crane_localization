/**
 * @file: gridmatcher.cc
 * @author ipsg
 * @brief:
 * @version:
 * @date:  2020-03-11
 * @copyright: Copyright (c) 2021
 */

#include "gridmatcher.h"
using namespace std;
using namespace cv;
bool Gridmatcher::init(const vector<KeyPoint> &vkp1, const Size size1,
                       const vector<KeyPoint> &vkp2, const Size size2,
                       const vector<int> &vnMatches12) {
  // Input initialize
  NormalizePoints(vkp1, size1, mvP1);
  NormalizePoints(vkp2, size2, mvP2);
  mNumberMatches = 0;
  ConvertMatches(vnMatches12);
  return true;
}

// Normalize Key Points to Range(0 - 1)
void Gridmatcher::NormalizePoints(const vector<KeyPoint> &kp, const Size &size,
                                  vector<Point2f> &npts) {
  const size_t numP = kp.size();
  const int width = size.width;
  const int height = size.height;
  npts.resize(numP);
  for (size_t i = 0; i < numP; i++) {
    npts[i].x = kp[i].pt.x / width;
    npts[i].y = kp[i].pt.y / height;
  }
}

// Convert OpenCV DMatch to Match (pair<int, int>)
void Gridmatcher::ConvertMatches(const vector<DMatch> &vDMatches,
                                 vector<pair<int, int> > &vMatches) {
  vMatches.clear();
  vMatches.resize(mNumberMatches);
  for (size_t i = 0; i < mNumberMatches; i++) {
    vMatches[i] = pair<int, int>(vDMatches[i].queryIdx, vDMatches[i].trainIdx);
  }
}

void Gridmatcher::ConvertMatches(const vector<int> &vnMatches12) {
  if (vnMatches12.size() == 0) {
    cout << "error input" << endl;
    return;
  }
  for (size_t i = 0; i < vnMatches12.size(); i++) {
    if (vnMatches12[i] >= 0) {
      mNumberMatches++;
    }
  }
  mvMatches.reserve(mNumberMatches);
  for (size_t i = 0; i < vnMatches12.size(); i++) {
    if (vnMatches12[i] >= 0) {
      mvMatches.push_back(pair<int, int>(i, vnMatches12[i]));
    }
  }
  if (mvMatches.size() == 0) {
    cout << "error output" << endl;
    return;
  }
}

int Gridmatcher::GetGridIndexLeft(const Point2f &pt, int type) {
  int x = 0, y = 0;
  if (type == 1) {
    x = floor(pt.x * mGridSizeLeft.width);
    y = floor(pt.y * mGridSizeLeft.height);
  }
  if (type == 2) {
    x = floor(pt.x * mGridSizeLeft.width + 0.5);
    y = floor(pt.y * mGridSizeLeft.height);
  }
  if (type == 3) {
    x = floor(pt.x * mGridSizeLeft.width);
    y = floor(pt.y * mGridSizeLeft.height + 0.5);
  }
  if (type == 4) {
    x = floor(pt.x * mGridSizeLeft.width + 0.5);
    y = floor(pt.y * mGridSizeLeft.height + 0.5);
  }
  if (x >= mGridSizeLeft.width || y >= mGridSizeLeft.height) {
    return -1;
  }
  return x + y * mGridSizeLeft.width;
}

int Gridmatcher::GetGridIndexRight(const Point2f &pt) {
  int x = floor(pt.x * mGridSizeRight.width);
  int y = floor(pt.y * mGridSizeRight.height);
  return x + y * mGridSizeRight.width;
}

// GetNB9就是获取一个格子它周围的九宫格
vector<int> Gridmatcher::GetNB9(const int idx, const Size &GridSize) {
  vector<int> NB9(9, -1);
  int idx_x = idx % GridSize.width;
  int idx_y = idx / GridSize.width;
  for (int yi = -1; yi <= 1; yi++) {
    for (int xi = -1; xi <= 1; xi++) {
      int idx_xx = idx_x + xi;
      int idx_yy = idx_y + yi;
      if (idx_xx < 0 || idx_xx >= GridSize.width || idx_yy < 0 ||
          idx_yy >= GridSize.height)
        continue;
      NB9[xi + 4 + yi * 3] = idx_xx + idx_yy * GridSize.width;
    }
  }
  return NB9;
}

// 400*9 20*20
void Gridmatcher::InitalizeNiehbors(Mat &neighbor, const Size &GridSize) {
  for (int i = 0; i < neighbor.rows; i++) {
    vector<int> NB9 = GetNB9(i, GridSize);
    int *data = neighbor.ptr<int>(i);
    memcpy(data, &NB9[0], sizeof(int) * 9);
  }
}

void Gridmatcher::SetScale(int Scale) {
  // Set Scale
  mGridSizeRight.width = mGridSizeLeft.width * mScaleRatios[Scale];
  mGridSizeRight.height = mGridSizeLeft.height * mScaleRatios[Scale];
  mGridNumberRight = mGridSizeRight.width * mGridSizeRight.height;
  // Initialize the neihbor of right grid
  mGridNeighborRight = Mat::zeros(mGridNumberRight, 9, CV_32SC1);
  InitalizeNiehbors(mGridNeighborRight, mGridSizeRight);
}

int Gridmatcher::GetInlierMask(vector<bool> &vbInliers, const Size Gridsize,
                               bool WithScale, bool WithRotation) {
  if (mvMatches.size() == 0) {
    cout << "no match!" << endl;
    return 1;
  }
  int max_inlier = 0;
  // Grid initialize
  mGridSizeLeft = Gridsize;
  mGridNumberLeft = mGridSizeLeft.width * mGridSizeLeft.height;
  // Initialize the neihbor of left grid 400*9
  mGridNeighborLeft = Mat::zeros(mGridNumberLeft, 9, CV_32SC1);
  InitalizeNiehbors(mGridNeighborLeft, mGridSizeLeft);
  if (!WithScale && !WithRotation) {
    SetScale(0);
    max_inlier = run(1);
    vbInliers = mvbInlierMask;
    return max_inlier;
  }
  if (WithRotation && WithScale) {
    for (int Scale = 0; Scale < 5; Scale++) {
      SetScale(Scale);
      for (int RotationType = 1; RotationType <= 8; RotationType++) {
        int num_inlier = run(RotationType);
        if (num_inlier > max_inlier) {
          vbInliers = mvbInlierMask;
          max_inlier = num_inlier;
        }
      }
    }
    return max_inlier;
  }
  if (WithRotation && !WithScale) {
    for (int RotationType = 1; RotationType <= 8; RotationType++) {
      int num_inlier = run(RotationType);
      if (num_inlier > max_inlier) {
        vbInliers = mvbInlierMask;
        max_inlier = num_inlier;
      }
    }
    return max_inlier;
  }
  if (!WithRotation && WithScale) {
    for (int Scale = 0; Scale < 5; Scale++) {
      SetScale(Scale);
      int num_inlier = run(1);
      if (num_inlier > max_inlier) {
        vbInliers = mvbInlierMask;
        max_inlier = num_inlier;
      }
    }
    return max_inlier;
  }
  return max_inlier;
}

void Gridmatcher::AssignMatchPairs(int GridType) {
  for (size_t i = 0; i < mNumberMatches; i++) {
    Point2f &lp = mvP1[mvMatches[i].first];
    Point2f &rp = mvP2[mvMatches[i].second];
    int lgidx = mvMatchPairs[i].first = GetGridIndexLeft(lp, GridType);
    int rgidx = -1;
    if (GridType == 1) {
      rgidx = mvMatchPairs[i].second = GetGridIndexRight(rp);
    } else {
      rgidx = mvMatchPairs[i].second;
    }
    if (lgidx < 0 || rgidx < 0) continue;
    mMotionStatistics.at<int>(lgidx, rgidx)++;
    mNumberPointsInPerCellLeft[lgidx]++;
  }
}

void Gridmatcher::VerifyCellPairs(int RotationType) {
  const int *CurrentRP = mRotationPatterns[RotationType - 1];
  for (int i = 0; i < mGridNumberLeft; i++) {
    if (sum(mMotionStatistics.row(i))[0] == 0) {
      mCellPairs[i] = -1;
      continue;
    }
    int max_number = 0;
    for (int j = 0; j < mGridNumberRight; j++) {
      int *value = mMotionStatistics.ptr<int>(i);
      if (value[j] > max_number) {
        mCellPairs[i] = j;
        max_number = value[j];
      }
    }
    int idx_grid_rt = mCellPairs[i];
    const int *NB9_lt = mGridNeighborLeft.ptr<int>(i);
    const int *NB9_rt = mGridNeighborRight.ptr<int>(idx_grid_rt);
    int score = 0;
    double thresh = 0;
    int numpair = 0;
    for (size_t j = 0; j < 9; j++) {
      int ll = NB9_lt[j];
      int rr = NB9_rt[CurrentRP[j] - 1];
      if (ll == -1 || rr == -1) continue;
      score += mMotionStatistics.at<int>(ll, rr);
      thresh += mNumberPointsInPerCellLeft[ll];
      numpair++;
    }
    thresh = THRESH_FACTOR * sqrt(thresh / numpair);
    if (score < thresh) {
      mCellPairs[i] = -2;
    }
  }
}

int Gridmatcher::run(int RotationType) {
  mvbInlierMask.assign(mNumberMatches, false);
  // Initialize Motion Statisctics
  mMotionStatistics = Mat::zeros(mGridNumberLeft, mGridNumberRight, CV_32SC1);
  mvMatchPairs.assign(mNumberMatches, pair<int, int>(0, 0));
  for (int GridType = 1; GridType <= 4; GridType++) {
    mMotionStatistics.setTo(0);
    mCellPairs.assign(mGridNumberLeft, -1);
    mNumberPointsInPerCellLeft.assign(mGridNumberLeft, 0);
    AssignMatchPairs(GridType);
    VerifyCellPairs(RotationType);
    // Mark inliers
    for (size_t i = 0; i < mNumberMatches; i++) {
      if (mCellPairs[mvMatchPairs[i].first] == mvMatchPairs[i].second) {
        mvbInlierMask[i] = true;
      }
    }
  }
  int num_inlier = accumulate(mvbInlierMask.begin(), mvbInlierMask.end(), 0);
  return num_inlier;
}

void Gridmatcher::imresize(Mat &src, int height) {
  double ratio = src.rows * 1.0 / height;
  int width = static_cast<int>(src.cols * 1.0 / ratio);
  resize(src, src, Size(width, height));
}