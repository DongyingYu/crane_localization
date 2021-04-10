/***
 * @file: test_SSIM.cc
 * @author: Dongying (yudong2817@sina.com)
 * @brief:
 * @version:
 * @date:  2021-04-09
 * @copyright: Copyright (c) 2021
 */

#include <chrono>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "frame.h"
#include "localization.h"
#include "websocket_endpoint.h"

// cv::Scalar getMSSIM(cv::Mat inputimage1, cv::Mat inputimage2);
int main(int argc, char** argv) {
  std::string video_file = "/home/ipsg/dataset_temp/74_new_test_cut.mp4";
  int skip_frames = 0;

  // skip some frames
  cv::Mat img;
  cv::VideoCapture capture(video_file);
  while (skip_frames > 0) {
    capture >> img;
    skip_frames--;
  }

  auto location = std::make_shared<Localization>(
      "./vocabulary/image_save3/rgb.txt", 0.01, true, 1);
  int cnt = 0;
  for (int cnt = 0;; ++cnt) {
    capture >> img;
    if (img.rows == 0 || img.cols == 0) continue;
    // 输入图像的尺寸变换与先验图像尺寸不同不会影响匹配,但是transpose影响非常大，如果transpose不同，会带来问题
    cv::transpose(img, img);
    // cv::resize(img,img,{0,0},0.5,0.5);
    // 现场部署时需要用
    // if(cnt%3 != 0)
    //   continue;
    // cv::imshow("image_video", img);
    Frame::Ptr frame = std::make_shared<Frame>(img);
    double position;
    bool status = location->localizeByMSSIM(frame, position, true);
    std::cout << "The frame cnt : " << cnt << std::endl;
    std::cout << "The crane position is : " << position << std::endl;
    cv::waitKey();
  }

  return 0;

  /*
  if (argc != 3) {
    std::cout << "Usage: ./test_SSIM path_to_image1 path_to_image2 "
              << std::endl;
    return 1;
  }
  cv::Mat SrcImage1 = cv::imread(argv[1]);
  cv::Mat SrcImage2 = cv::imread(argv[2]);
  double t = cv::getTickCount();
  cv::Scalar SSIM1 = getMSSIM(SrcImage1, SrcImage2);
  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  std::cout << "[INFO]: The time consuming of SSIM " << t << "  s  "
            << std::endl;
  std::cout << "[INFO]: The value of channel one: " << SSIM1.val[0] * 100
            << std::endl;
  std::cout << "[INFO]: The value of channel two: " << SSIM1.val[1] * 100
            << std::endl;
  std::cout << "[INFO]: The value of channel three: " << SSIM1.val[2] * 100
            << std::endl;
  std::cout << "[INFO]: The value of Similarity score: "
            << (SSIM1.val[2] + SSIM1.val[1] + SSIM1.val[0]) / 3 * 100
            << std::endl;
  cv::waitKey(0);
  return 0;
  */
}
/*
cv::Scalar getMSSIM(cv::Mat inputimage1, cv::Mat inputimage2) {
  cv::Mat i1 = inputimage1;
  cv::Mat i2 = inputimage2;
  const double C1 = 6.5025, C2 = 58.5225;
  cv::Mat I1, I2;
  i1.convertTo(I1, CV_32F);
  i2.convertTo(I2, CV_32F);
  std::cout << "[INFO]: image size : " << i1.size() << "\t image channel : " <<
i1.channels() << std::endl;
  cv::imshow("convert_image",i1);
  // 图像数据求平方
  cv::Mat I2_2 = I2.mul(I2);
  cv::Mat I1_2 = I1.mul(I1);
  cv::Mat I1_I2 = I1.mul(I2);
  cv::Mat mu1, mu2;
  // 直接对原图解进行高斯模糊可行？需要对图像外围进行扩展或缩减吗？
  // 高斯模糊，即对图像进行加权平均
  cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
  cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);
  cv::Mat mu1_2 = mu1.mul(mu1);
  cv::Mat mu2_2 = mu2.mul(mu2);
  cv::Mat mu1_mu2 = mu1.mul(mu2);
  cv::Mat sigma1_2, sigma2_2, sigma12;
  // 计算方差与协方差, 方差： D(x) = E{[X - E(X)]2} = E{X2 - 2XE(X) + [E(X)]2}
注：这里计算时取X = E(X)
  cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
  sigma1_2 -= mu1_2;
  cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
  sigma2_2 -= mu2_2;
  cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
  sigma12 -= mu1_mu2;
  cv::Mat t1, t2, t3;
  t1 = 2 * mu1_mu2 + C1;
  t2 = 2 * sigma12 + C2;
  t3 = t1.mul(t2);
  t1 = mu1_2 + mu2_2 + C1;
  t2 = sigma1_2 + sigma2_2 + C2;
  t1 = t1.mul(t2);
  cv::Mat ssim_map;
  // ssim_map = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_2 + mu2_2 +
C1).*(sigma1_2 + sigma2_2 + C2));
  cv::divide(t3, t1, ssim_map);
  // 将平均值作为两图像的结构相似性度量，即MSSIM
  cv::Scalar mssim = cv::mean(ssim_map);
  return mssim;
}
*/