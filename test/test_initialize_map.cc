#include "frame.h"
#include "optimizer.h"
#include "utils.h"
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

int width = 2560;
int height = 1440;
double fx = 2052.01136163;
double fy = 2299.88726199;
double cx = 1308.23302753;
double cy = 713.03063759;
double k1 = -0.47040093;
double k2 = 1.0270395;
double k3 = -2.00061705;
double k4 = 1.64023946;
Intrinsic intrinsic = Intrinsic(fx, fy, cx, cy).scale(0.5);
cv::Size img_size = cv::Size(width * 0.5, height * 0.5);

int main(int argc, char **argv) {
  // config
  std::string video_file =
      "/home/xt/Documents/data/DATASETS/ros_bag/BL-EX346HP-15M/crane/78.mp4";
  int skip_frames = 1200;
  double scale_image = 1.0;
  bool transpose_image = true;

  if (scale_image != 1.0) {
    intrinsic = intrinsic.scale(scale_image);
    img_size =
        cv::Size(img_size.width * scale_image, img_size.height * scale_image);
  }
  if (transpose_image) {
    intrinsic = intrinsic.transpose();
    img_size = cv::Size(img_size.height, img_size.width);
  }

  auto undistorter =
      std::make_shared<UndistorterFisheye>(img_size, intrinsic, k1, k2, k3, k4);
  cv::Mat K = undistorter->getNewK();

  // skip some frames
  cv::Mat img;
  cv::VideoCapture capture(video_file);
  while (skip_frames > 0) {
    capture >> img;
    skip_frames--;
  }

  Frame::Ptr frame_cur, frame_prev;

  auto initializer = std::make_shared<Initializer>();

  // std::vector
  int cnt = 0;
  double speed = 0;
  std::list<Frame::Ptr> frames;
  while (1) {
    capture >> img;
    if (transpose_image) {
      cv::transpose(img, img);
    }
    cv::resize(img, img, {0, 0}, scale_image, scale_image);

    frame_cur = std::make_shared<Frame>(img, intrinsic, undistorter);
    // frame_cur->debugDraw();

    frames.emplace_back(frame_cur);

    if (frames.size() == 1) {
      continue;
    } else if (frames.size() < 15) {
      frame_prev = frames.front();
      // frame_prev->debugDraw();
      // frame_cur->debugDraw();
      Map::Ptr map = initializer->initialize(frame_prev, frame_cur, K);

      // frames.pop_front();

      Eigen::Vector3d ave_twc(0, 0, 0);
      if (map) {
        G2oOptimizer::mapBundleAdjustment(map);
        Eigen::Vector3d twc1 = map->frames_.back()->getEigenTwc();
        std::cout << "[INFO]: twc after g2o: " << toString(twc1) << std::endl;

        G2oOptimizerForLinearMotion::mapBundleAdjustment(map);
        Eigen::Vector3d twc2 = map->frames_.back()->getEigenTwc();
        std::cout << "[INFO]: twc after constrained g2o: " << toString(twc2)
                  << std::endl;
        ave_twc += twc1 / (frames.size() - 1);
      }
      std::cout << "[INFO]: average twc: " << toString(ave_twc) << std::endl;
    }

    cv::waitKey();
  }
}