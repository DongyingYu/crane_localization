#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
int main() {

  std::string video_dir =
      "/home/xt/Documents/data/DATASETS/ros_bag/BL-EX346HP-15M/";
  std::string video_file = video_dir + "192.168.1.146_01_20210205105528697.mp4";
  VideoCapture capture(video_file);

  double scale = 1;
  int width = 2560 / int(scale);
  int height = 1440 / int(scale);
  cv::Size frame_size(width, height);
  double fx = 2428.05872198 / scale;
  double fy = 1439.05448033 / scale;
  double cx = 1439.05448033 / scale;
  double cy = 846.58407292 / scale;
  double k1 = -0.85227746;
  double k2 = 0.57019705;
  double r1 = -0.0330824;
  double r2 = -0.02863118;
  Mat camera_matrix = (Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

  std::vector<double> _dist_coeffs = {k1, k2, r1, r2};
  cv::Mat dist_coeffs = cv::Mat(4, 1, CV_64F, &_dist_coeffs[0]);
  cv::Mat new_camera_matrix = cv::getOptimalNewCameraMatrix(
      camera_matrix, dist_coeffs, frame_size, 0, frame_size);

  cv::Mat_<double> I = cv::Mat_<double>::eye(3, 3);
  cv::Mat map1 = cv::Mat::zeros(frame_size, CV_16SC2);
  cv::Mat map2 = cv::Mat::zeros(frame_size, CV_16UC1);
  cv::initUndistortRectifyMap(camera_matrix, dist_coeffs, I, new_camera_matrix,
                              frame_size, map1.type(), map1, map2);
  std::cout << "Done:  get undistort map" << std::endl;

  int i = 0;
  while (1) {
    Mat frame;
    capture >> frame;
    if (frame.empty()) {
      break;
    }
    i++;
    // if (i < 1200) {
    //   continue;
    // }

    cv::Mat undistorted_frame;
    cv::remap(frame, undistorted_frame, map1, map2, cv::INTER_LINEAR,
              cv::BORDER_CONSTANT);

    resize(frame, frame, {0, 0}, 0.5, 0.5);
    resize(undistorted_frame, undistorted_frame, {0, 0}, 0.5, 0.5);
    imshow("ori", frame);
    imshow("undistorted", undistorted_frame);
    waitKey(31);
  }
  return 0;
}