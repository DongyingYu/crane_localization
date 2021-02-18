#include "frame.h"
#include "initializer.h"
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

double fx = 2428.05872198;
double fy = 1439.05448033;
double cx = 1439.05448033;
double cy = 846.58407292;
Intrinsic intrinsic = Intrinsic(fx, fy, cx, cy).scale(0.5);

int main(int argc, char **argv) {
  // config
  std::string video_file =
      "/home/xt/Documents/data/DATASETS/ros_bag/BL-EX346HP-15M/crane/78.mp4";
  int skip_frames = 1200;
  double scale_image = 0.6;
  bool transpose_image = true;

  // skip some frames
  cv::Mat img;
  cv::VideoCapture capture(video_file);
  while (skip_frames > 0) {
    capture >> img;
    skip_frames--;
  }

  cv::Ptr<Frame> frame_cur, frame_prev;

  if (transpose_image) {
    intrinsic = intrinsic.transpose();
  }
  intrinsic = intrinsic.scale(scale_image);
  cv::Mat K = intrinsic.K();

  auto initializer = std::make_shared<Initializer>();

  // std::vector
  int cnt = 0;
  double speed = 0;
  std::list<cv::Ptr<Frame>> frames;
  while (1) {
    capture >> img;
    if (transpose_image) {
      cv::transpose(img, img);
    }
    cv::resize(img, img, {0, 0}, scale_image, scale_image);
    frame_cur = cv::makePtr<Frame>(img, intrinsic);

    frames.emplace_back(frame_cur);

    if (frames.size() < 15) {
      continue;
    } else {
      frame_prev = frames.front();
      Map::Ptr map = initializer->initialize(frame_cur, frame_prev, K);
      map->initial_ba();
      frames.pop_front();

      if (map) {
        cv::Mat t = map->frames_[1]->t_cw_;
        std::cout << cnt << ": current speed: " << t.at<double>(0);
        speed = (speed * cnt + t.at<double>(0)) / (++cnt);
        std::cout << ": average speed: " << speed << std::endl;
      }
    }

    cv::waitKey();
  }
}