/**
 * @file localization.cc
 * @author Dongying (yudong2817@sina.com)
 * @brief
 * @version 0.1
 * @date 2021-02-28
 *
 * @copyright Copyright (c) 2021
 */

#include "localization.h"

Localization::Localization(const std::string &vocab_file,
                          const std::string &preload_keyframes,
                          const double &threshold,const bool &transpose_image,const int &win_size)
    : threshold_(threshold), win_size_(win_size) {
  // 1. load vocabulary
  pVocabulary_ = new ORBVocabulary();
  bool bVocLoad = pVocabulary_->loadFromTextFile(vocab_file);
  if (!bVocLoad) {
    std::string msg = "[ERROR]: Load vocabulary failed " + vocab_file;
    throw std::runtime_error(msg);
  }

  // 2. load offline images(keyframes)
  std::vector<cv::Mat> images = loadImages(preload_keyframes);
  if (images.empty()) {
    std::cout << "[WARNING]: No pre-saved keyframes" << std::endl;
  }
  std::cout << "The number of images:  " << images.size() << std::endl;

  for (int i = 0; i < images.size(); i++) {
    if (transpose_image) {
      cv::transpose(images[i], images[i]);
    }
    Frame::Ptr frame = std::make_shared<Frame>(images[i], pVocabulary_);
    frame->computeBoW();
    frames_.emplace_back(frame);
  }
}

Localization::~Localization() {}

bool Localization::localize(const Frame::Ptr &cur_frame/*const cv::Mat &image*/,double &position,const bool &verbose) {
  // Frame::Ptr frame = std::make_shared<Frame>(cur_frame->getImage(), pVocabulary_);
  cur_frame->setVocabulary(pVocabulary_);
  cur_frame->computeBoW();
  auto bow_vec = cur_frame->getBowVoc();
  std::vector<float> score_temp;
  for (int i = 0; i < frames_.size(); i++) {
    float s = pVocabulary_->score(frames_[i]->getBowVoc(), bow_vec);
    if (s < 0 || s > 1) {
      s = 0.0;
    }
    score_temp.push_back(s);
  }

  if (winFrames_.size() >= win_size_) {
    winFrames_.pop_front();
  }
  winFrames_.push_back(score_temp);

  auto vscore = std::vector<std::pair<float, int>>(
      frames_.size(), std::pair<float, int>(0.0, 0));

  for (int i = 0; i < score_temp.size(); i++) {
    float sum_score_temp = 0.0;
    for (int j = 0; j < winFrames_.size(); j++) {
      sum_score_temp = sum_score_temp + winFrames_[j][i];
    }
    vscore[i].first = sum_score_temp;
    vscore[i].second = i;
  }
  // 降序排列，筛选出排在前五的的分值
  std::sort(vscore.begin(), vscore.end(),
            [](const pair<float, int> &a, const pair<float, int> &b) {
              return a.first > b.first;
            });
  std::cout << "The size of vscore: " << vscore.size() << std::endl;
  for (int i = 0; i < 5; i++) {
    std::cout << "The " << i << "  number of vscore is : " << vscore[i].first
              << std::endl;
  }
  // 计算共享单词个数
  // for ( DBoW2::BowVector::const_iterator
  // vit=frames[frames.size()-1]->bow_vec_.begin(),
  //       vend=frames[frames.size()-1]->bow_vec_.end();vit != vend; vit++)
  // {
  //     std::cout << " the id of BowVoc: " << vit->first << std::endl;
  // }
  std::cout << " The max score is : " << vscore[0].first << std::endl;
  std::cout << " The index number of image is :  " << vscore[0].second
            << std::endl;

  if (verbose) {
    Frame::Ptr &best_frame = frames_[vscore[0].second];
    std::cout << cur_frame->getImage().size() << std::endl<< std::endl;
    std::cout << best_frame->getImage().size() << std::endl<< std::endl<< std::endl;

    const int height = max(best_frame->getImage().rows, cur_frame->getImage().rows);
    const int width = best_frame->getImage().cols + cur_frame->getImage().cols;

    cv::Mat output(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    std::cout << output.size() << std::endl;

    best_frame->getImage().copyTo(output(cv::Rect(
        0, 0, best_frame->getImage().cols, best_frame->getImage().rows)));
    cur_frame->getImage().copyTo(output(
        cv::Rect(best_frame->getImage().cols, 0, cur_frame->getImage().cols, cur_frame->getImage().rows)));

    cv::resize(output, output, {0, 0}, 0.6, 0.6);
    std::cout << output.size() << std::endl;
    
    cv::imshow("Image contrast", output);
  }
  position = image_position[vscore[0].second];
  std::cout << "[INFO]: The value of threshold: " << threshold_ << std::endl;
  if(vscore[0].first > threshold_)
    return true;
  else
    return false;
}

std::vector<cv::Mat> Localization::loadImages(const std::string &filename) {
  std::ifstream f_index(filename);
  if (!f_index.is_open()) {
    std::string msg = "Open file failed: " + filename;
    throw std::runtime_error(msg);
  }
  std::vector<std::string> image_indexex;
  while (!f_index.eof()) {
    std::string line;
    getline(f_index, line);
    if (!line.empty()) {
      std::stringstream ss;
      ss << line;
      std::string index;
      ss >> index;
      double true_position;
      ss >> true_position;
      image_indexex.push_back(index);
      // std::cout << "The size of image_index is:   " << image_indexex.size() << std::endl;
      image_position.push_back(true_position);
    }
  }

  // 截取目录
  int pos = filename.find_last_of('/');
  std::string dir = filename.substr(0, pos);
  if (dir.empty()) {
    dir = ".";
  }

  std::vector<cv::Mat> images;
  for (int i = 0; i < image_indexex.size(); ++i) {
    std::string path = dir + "/" + image_indexex[i];
    cv::Mat img = cv::imread(path);
    if (img.empty()) {
      std::cout << "[WARNING]: read image failed " << path << std::endl;
      continue;
    }
    images.push_back(img);
  }
  return images;
}
