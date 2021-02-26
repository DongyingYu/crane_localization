/**
 * @file /**
 * @file localization.h
 * @author Dongying (yudong2817@sina.com)
 * @brief
 * @version 0.1
 * @date 2021-02-28
 *
 * @copyright Copyright (c) 2021
 */
#pragma once

#include <chrono>
#include <list>
#include <mutex>
#include <thread>
#include <utility>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include "ORBVocabulary.h"
#include "frame.h"

class Localization
{

public:
    using Ptr = std::shared_ptr<Localization>;

    Localization(int cnt);
    ~Localization();

    int videoCapturescore(const cv::Mat &video_image /*const std::string &image_address*/);
    void loadVocabulary();
    void loadImages(const std::string &strIndexFilename);
    void processKeyimages();

    static bool compareScore(const pair<float,int> a, const pair<float,int> b);

public:
    ORBVocabulary* pVocabulary_;
    std::vector<std::string> vimageIndex_;
    // Frame::Ptr cur_frame_ = nullptr;
    std::vector<Frame::Ptr> frames_;
    std::vector<std::pair<float,int>> vscore_;
    int win_size_;
    std::deque<std::vector<float>> winFrames_;
private:
    /* data */
};

