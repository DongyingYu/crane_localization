#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;
using namespace cv;

int main() {

  std::string video_file = "/home/ipsg/dataset_temp/calib/calib_4.mp4";
  VideoCapture capture(video_file);

  double scale = 1;
  int width = 2560 / int(scale);
  int height = 1440 / int(scale);
  cv::Size frame_size(width, height);
  double fx = 2910.033608 / scale;
  double fy = 3258.20009 / scale;
  double cx = 1263.04576 / scale;
  double cy = 694.422101 / scale;
  double k1 = -1.45086299;
  double k2 = 3.69816983;
  double r1 = 0.01120005;
  double r2 =  0.01063321;
  double r3 = -5.41547612;
  Mat camera_matrix = (Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

  std::vector<double> _dist_coeffs = {k1, k2, r1, r2, r3};
  cv::Mat dist_coeffs = cv::Mat(5, 1, CV_64F, &_dist_coeffs[0]);
  cv::Mat new_camera_matrix = cv::getOptimalNewCameraMatrix(
      camera_matrix, dist_coeffs, frame_size, 0, frame_size);

  std::cout << "camera_matrix " << std::endl << camera_matrix << std::endl;
  std::cout << "new_camera_matrix " << std::endl
            << new_camera_matrix << std::endl;

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

    double scale_factor = 0.4;
    resize(frame, frame, {0, 0}, scale_factor, scale_factor);
    resize(undistorted_frame, undistorted_frame, {0, 0}, scale_factor,
           scale_factor);
    imshow("ori", frame);
    imshow("undistorted", undistorted_frame);
    waitKey();
  }
  return 0;
}