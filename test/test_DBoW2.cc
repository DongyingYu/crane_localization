/**
 * @file test_DBoW2.cc
 * @author Dongying (yudong2817@sina.com)
 * @brief
 * @version 0.1
 * @date 2021-02-24
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "frame.h"
#include "localization.h"

int main(int argc, char** argv){
    std::string video_file = "/home/ipsg/dataset_temp/78_cut.mp4";
    int skip_frames = 0;
    double scale_image = 0.6;
    bool transpose_image = true;

    // skip some frames
    cv::Mat img;
    cv::VideoCapture capture(video_file);
    while (skip_frames > 0) {
        capture >> img;
        skip_frames--;
    }

    Localization::Ptr location = std::make_shared<Localization>(3);

    int cnt = 0;
    while (1) {
    cnt++;
    capture >> img;
    cv::imshow("image_video",img);
    int index = location->videoCapturescore(img);
    cv::waitKey();
  }

    return 0;
}

