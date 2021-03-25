#include "camera_model.h"
#include "frame.h"
#include "map.h"
#include "optimizer.h"
#include "utils.h"

int main(int argc, char **argv) {
  // 默认参数
  std::string video_file =
      "/home/xt/Documents/data/DATASETS/ros_bag/BL-EX346HP-15M/crane/78.mp4";
  std::string yaml_file =
      "/home/xt/Documents/data/3D-Mapping/3D-Reconstruction/case-base/"
      "crane_localization/conf/BL-EX346HP-15M.yaml";
  int skip_frames = 1200;
  // 从命令行获取参数
  if (argc == 1) {
  } else if (argc == 3) {
    video_file = argv[1];
    yaml_file = argv[2];
  } else if (argc == 4) {
    video_file = argv[1];
    yaml_file = argv[2];
    skip_frames = atoi(argv[3]);
  } else {
    std::cout << "Usage: exec video_file yaml_file" << std::endl;
    std::cout << "       exec video_file yaml_file skip_frame" << std::endl;
  }

  std::cout << "[INFO]: video_file = " << video_file << std::endl;
  std::cout << "[INFO]: yaml_file = " << yaml_file << std::endl;
  std::cout << "[INFO]: skip_frame = " << skip_frames << std::endl;

  // 相机模型
  CameraModel::Ptr camera_model =
      std::make_shared<CameraModelPinholeEqui>(yaml_file);

  // 内部处理时，将图片转置
  bool transpose_image = true;
  if (transpose_image) {
    camera_model->transpose();
  }

  // 测试所用的视频，被缩放了0.5倍，所以需将相机模型缩放0.5倍
  double scale_camera_model = 0.5;
  if (scale_camera_model != 1) {
    camera_model->scale(scale_camera_model);
  }

  cv::Mat img;
  cv::VideoCapture capture(video_file);
  while (skip_frames > 0) {
    capture >> img;
    skip_frames--;
  }

  Frame::Ptr frame_curr, frame_prev;
  while (1) {
    capture >> img;

    if (transpose_image) {
      cv::transpose(img, img);
    }

    frame_prev = frame_curr;
    frame_curr = std::make_shared<Frame>(img, camera_model);

    if (!frame_prev) {
      continue;
    }

    Map::Ptr map = std::make_shared<Map>();
    bool status = map->initialize(frame_prev, frame_curr);

    if (status) {
      G2oOptimizer::Ptr opt = map->buildG2oOptKeyFrameBa();
      opt->optimize();
      opt->optimizeLinearMotion();

      std::cout << "[INFO]: The initialized map after g2o LinearMotion"
                << std::endl;
      map->debugPrintMap();
    }

    cv::waitKey();
  }
}