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
                           const double &threshold, const bool &transpose_image,
                           const int &win_size)
    : threshold_(threshold), win_size_(win_size) {
  // 1. load vocabulary
  vocabulary_ = new Vocabulary();
  bool bVocLoad = vocabulary_->loadFromTextFile(vocab_file);
  if (!bVocLoad) {
    std::string msg = "[ERROR]: Load vocabulary failed " + vocab_file;
    throw std::runtime_error(msg);
  }

  // 2. load offline images(keyframes)
  std::vector<cv::Mat> images = loadImages(preload_keyframes);
  if (images.empty()) {
    std::cout << "[WARNING]: No pre-saved keyframes" << std::endl;
  }
  std::cout << "[INFO]: The number of images:  " << images.size() << std::endl;

  for (int i = 0; i < images.size(); i++) {
    if (transpose_image) {
      cv::transpose(images[i], images[i]);
    }
    Frame::Ptr frame = std::make_shared<Frame>(images[i], vocabulary_);
    frame->computeBoW();
    frame->releaseImage();
    frames_.emplace_back(frame);
  }
}

Localization::Localization(const std::string &preload_keyframes,
                           const double &threshold, const bool &transpose_image,
                           const int &win_size)
    : threshold_SSIM_(threshold), win_size_SSIM_(win_size) {
  // load offline images
  std::vector<cv::Mat> images = loadImages(preload_keyframes);
  if (images.empty()) {
    std::cout << "[WARNING]: No pre-saved keyframes" << std::endl;
  }
  for (int i = 0; i < images.size(); i++) {
    if (transpose_image) {
      cv::transpose(images[i], images[i]);
    }
    // 后续只需传入到frame中然后截取部分图像，用以计算SSIM即可
    Frame::Ptr frame = std::make_shared<Frame>(images[i]);
    frame->computeSSIM();
    frame->releaseImage();
    frames_.emplace_back(frame);
  }
}

Localization::~Localization() {}

bool Localization::localize(const Frame::Ptr &cur_frame, double &position,
                            const bool &verbose) {
  // Frame::Ptr frame = std::make_shared<Frame>(cur_frame->getImage(),
  // vocabulary_);
  cur_frame->setVocabulary(vocabulary_);
  cur_frame->computeBoW();
  auto bow_vec = cur_frame->getBowVoc();
  std::vector<float> score_temp;
  for (int i = 0; i < frames_.size(); i++) {
    float s = vocabulary_->score(frames_[i]->getBowVoc(), bow_vec);
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
  for (int i = 0; i < vscore.size(); i++) {
    std::cout << "The " << i << "  number of vscore is : " << vscore[i].first
              << "   "
              << "The index is: " << vscore[i].second << std::endl;
  }
  std::pair<float, int> compare_result(0.0, 0);
  int index = 0;
  for (int i = 0; i < 5; i++) {
    std::cout << "The " << i << "  number of vscore is : " << vscore[i].first
              << "The index is: " << vscore[i].second << std::endl;
    float score_compare = 0;
    if (vscore[i].second <= 2) {
      for (int j = 0; j < vscore.size(); j++) {
        if (vscore[j].second <= (vscore[i].second + 3))
          score_compare = score_compare + vscore[j].first;
      }
    } else if (vscore[i].second >= (vscore.size() - 4)) {
      for (int j = 0; j < vscore.size(); j++) {
        if (vscore[j].second >= (vscore[i].second - 3))
          score_compare = score_compare + vscore[j].first;
      }
    } else {
      for (int j = 0; j < vscore.size(); j++) {
        if (vscore[j].second >= (vscore[i].second - 3) &&
            vscore[j].second <= (vscore[i].second + 3)) {
          score_compare = score_compare + vscore[j].first;
        }
      }
    }
    score_compare = score_compare - vscore[i].first;
    if (score_compare > compare_result.first) {
      compare_result.first = score_compare;
      compare_result.second = vscore[i].second;
      index = i;
    }
  }
  if (compare_result.first == 0) {
    compare_result.first = vscore[0].first;
    compare_result.second = vscore[0].second;
    index = 0;
  }
  std::cout << "[INFO]: The final score result : "
            << vscore[index].first /*compare_result.first*/
            << "   " << compare_result.second << std::endl;
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
    Frame::Ptr &best_frame =
        frames_[compare_result.second /*vscore[0].second*/];
    std::cout << cur_frame->getImage().size() << std::endl << std::endl;
    std::cout << best_frame->getImage().size() << std::endl
              << std::endl
              << std::endl;

    const int height =
        max(best_frame->getImage().rows, cur_frame->getImage().rows);
    const int width = best_frame->getImage().cols + cur_frame->getImage().cols;

    const int height_roi =
        max(best_frame->getImageRoi().rows, cur_frame->getImageRoi().rows);
    const int width_roi =
        best_frame->getImageRoi().cols + cur_frame->getImageRoi().cols;
    cv::Mat output_roi(height_roi, width_roi, CV_8UC3, cv::Scalar(0, 0, 0));

    cv::Mat output(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    std::cout << output.size() << std::endl;

    best_frame->getImage().copyTo(output(cv::Rect(
        0, 0, best_frame->getImage().cols, best_frame->getImage().rows)));
    cur_frame->getImage().copyTo(output(cv::Rect(best_frame->getImage().cols, 0,
                                                 cur_frame->getImage().cols,
                                                 cur_frame->getImage().rows)));
    best_frame->getImageRoi().copyTo(output_roi(cv::Rect(
        0, 0, best_frame->getImageRoi().cols, best_frame->getImageRoi().rows)));
    cur_frame->getImageRoi().copyTo(output_roi(cv::Rect(
        best_frame->getImageRoi().cols, 0, cur_frame->getImageRoi().cols,
        cur_frame->getImageRoi().rows)));

    cv::resize(output, output, {0, 0}, 0.6, 0.6);
    // cv::resize(output_roi, output_roi, {0, 0}, 0.6, 0.6);
    std::cout << output.size() << std::endl;

    cv::imshow("Image contrast", output);
    cv::imshow("Image_roi contrast", output_roi);
  }
  position = image_position_[compare_result.second /*vscore[0].second*/];
  std::cout << "[INFO]: The value of threshold: " << threshold_ << std::endl;

  if (vscore[index].first > threshold_)
    return true;
  else
    return false;
}

bool Localization::localizeByMSSIM(const Frame::Ptr &cur_frame,
                                   double &position, const bool &verbose) {
  cur_frame->computeSSIM();
  // image_convert、mu、mu_2、sigma_2
  // auto ssim_cur_frame = cur_frame->getSSIMData2();
  std::vector<float> score_temp;
  for (int i = 0; i < frames_.size(); i++) {
    // auto ssim_pre_frame = frames_[i]->getSSIMData2();
    cv::Mat I1_I2 = frames_[i]->getSSIMDatax().mul(cur_frame->getSSIMDatax());
    cv::Mat mu1_mu2 = frames_[i]->getSSIMDatamu().mul(cur_frame->getSSIMDatamu());
    cv::Mat sigma_I1I2;
    cv::GaussianBlur(I1_I2, sigma_I1I2, cv::Size(11, 11), 1.5);
    sigma_I1I2 -= mu1_mu2;
    cv::Mat temp1, temp2, temp3;
    temp1 = 2 * mu1_mu2 + c1_;
    temp2 = 2 * sigma_I1I2 + c2_;
    // temp3 = temp1.mul(temp2);
    temp3 = temp2.clone();
    temp1 = cur_frame->getSSIMDatamu2() + frames_[i]->getSSIMDatamu2() + c1_;
    temp2 = cur_frame->getSSIMDatasigma2() + frames_[i]->getSSIMDatasigma2() + c2_;
    // temp1 = temp1.mul(temp2);
    temp1 = temp2.clone();
    cv::Mat ssim_map;
    cv::divide(temp3, temp1, ssim_map);
    cv::Scalar mssim = cv::mean(ssim_map);
    float score = (mssim.val[2] + mssim.val[1] + mssim.val[0]) / 3 * 100;
    if (score < 0 || score > 100) {
      score = 0.0;
    }
    score_temp.emplace_back(score);
  }

  if (winFrames_.size() >= win_size_SSIM_) {
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
              << "   "
              << "The index is: " << vscore[i].second << std::endl;
  }
  bool is_selected_one = false;
  if ((abs(vscore[0].first - 2 * vscore[1].first + vscore[2].first) > 1.0) &&
      ((vscore[0].first - vscore[1].first) /
           (vscore[1].first - vscore[2].first) <
       7.0) &&
      ((vscore[0].first - vscore[1].first) /
           (vscore[1].first - vscore[2].first) >
       3.0) &&
      (abs(image_position_[vscore[0].second] -
           image_position_[vscore[1].second]) < 1.5)) {
    is_selected_one = true;
    std::cout << "\033[31m Screening state of optimal score --one! \033[0m "
              << std::endl;
  } else {
    is_selected_one = false;
  }

  bool is_selected_two = false;
  if ((vscore[0].first - vscore[1].first) > 0.5 &&
      abs(vscore[0].second - vscore[1].second) > 2 &&
      abs(vscore[0].second - vscore[2].second) > 2) {
    is_selected_two = true;
    std::cout << "\033[31m Screening state of optimal score --two! \033[0m "
              << std::endl;
  } else {
    is_selected_two = false;
  }

  bool is_selected_three = false;
  if (vscore[0].first > 75) {
    std::cout << "\033[31m Screening state of optimal score --three! \033[0m "
              << std::endl;
    is_selected_three = true;
  } else {
    is_selected_three = false;
  }

  // float score_sum = 0.0;
  // for (int i = 0; i < 5; i++) {
  //   score_sum = score_sum + vscore[i].first;
  // }
  // if (score_sum / 5.0 > 68) {
  //   std::cout << "\033[31m Screening state of optimal score --four! \033[0m "
  //             << std::endl;
  //   is_selected = true;
  // } else {
  //   is_selected = false;
  // }
  bool selected = (is_selected_one || is_selected_two || is_selected_three);
  if (verbose) {
    Frame::Ptr &best_frame = frames_[vscore[0].second];
    std::cout << cur_frame->getImage().size() << std::endl << std::endl;
    std::cout << best_frame->getImage().size() << std::endl
              << std::endl
              << std::endl;

    const int height =
        max(best_frame->getImage().rows, cur_frame->getImage().rows);
    const int width = best_frame->getImage().cols + cur_frame->getImage().cols;

    const int height_roi =
        max(best_frame->getImageRoi().rows, cur_frame->getImageRoi().rows);
    const int width_roi =
        best_frame->getImageRoi().cols + cur_frame->getImageRoi().cols;
    cv::Mat output_roi(height_roi, width_roi, CV_8UC3, cv::Scalar(0, 0, 0));

    cv::Mat output(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    std::cout << output.size() << std::endl;

    best_frame->getImage().copyTo(output(cv::Rect(
        0, 0, best_frame->getImage().cols, best_frame->getImage().rows)));
    cur_frame->getImage().copyTo(output(cv::Rect(best_frame->getImage().cols, 0,
                                                 cur_frame->getImage().cols,
                                                 cur_frame->getImage().rows)));
    best_frame->getImageRoi().copyTo(output_roi(cv::Rect(
        0, 0, best_frame->getImageRoi().cols, best_frame->getImageRoi().rows)));
    cur_frame->getImageRoi().copyTo(output_roi(cv::Rect(
        best_frame->getImageRoi().cols, 0, cur_frame->getImageRoi().cols,
        cur_frame->getImageRoi().rows)));

    cv::resize(output, output, {0, 0}, 0.6, 0.6);
    cv::resize(output_roi, output_roi, {0, 0}, 0.6, 0.6);
    std::cout << output.size() << std::endl;

    cv::imshow("Image contrast", output);
    cv::imshow("Image_roi contrast", output_roi);
  }

  if (selected) {
    position = image_position_[vscore[0].second];
    return true;
  } else
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
      // std::cout << "The size of image_index is:   " << image_indexex.size()
      // << std::endl;
      image_position_.push_back(true_position);
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
