/***
 * @file:
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

cv::Scalar getMSSIM(cv::Mat inputimage1, cv::Mat inputimage2);
int main(int argc, char** argv) {
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
}
cv::Scalar getMSSIM(cv::Mat inputimage1, cv::Mat inputimage2) {
  cv::Mat i1 = inputimage1;
  cv::Mat i2 = inputimage2;
  const double C1 = 6.5025, C2 = 58.5225;
  int d = CV_32F;
  cv::Mat I1, I2;
  i1.convertTo(I1, d);
  i2.convertTo(I2, d);
  cv::Mat I2_2 = I2.mul(I2);
  cv::Mat I1_2 = I1.mul(I1);
  cv::Mat I1_I2 = I1.mul(I2);
  cv::Mat mu1, mu2;
  cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
  cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);
  cv::Mat mu1_2 = mu1.mul(mu1);
  cv::Mat mu2_2 = mu2.mul(mu2);
  cv::Mat mu1_mu2 = mu1.mul(mu2);
  cv::Mat sigma1_2, sigma2_2, sigma12;
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
  cv::divide(t3, t1, ssim_map);
  cv::Scalar mssim = cv::mean(ssim_map);
  return mssim;
}
