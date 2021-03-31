/**
 * @file config_parser.cc
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2021-03-08
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "config_parser.h"
#include <yaml-cpp/yaml.h>
#include <iostream>

template <typename T>
static void getValue(const YAML::Node &node, const std::string &key, T &value) {
  if (!node) {
    std::cout << "[ERROR]: input node is empty" << std::endl;
    return;
  }
  if (!node[key]) {
    throw std::runtime_error("could not find key in node: " + key);
  }
  value = node[key].as<T>();
}

ConfigParser::ConfigParser(const std::string &config_yaml) {
  YAML::Node node = YAML::LoadFile(config_yaml);
  if (!node) {
    throw std::runtime_error("open yaml failed: " + config_yaml);
  }
  getValue<bool>(node, "transpose_image", transpose_image_);
  getValue<double>(node, "scale_image", scale_image_);
  getValue<double>(node, "debug_draw", debug_draw_);
  getValue<bool>(node, "pop_frame", pop_frame_);
  getValue<bool>(node, "save_position", save_position_);

  auto cam_node = node["camera_model"];
  if (!cam_node) {
    throw std::runtime_error("could not find 'camera_model' in node");
  }
  getValue<std::string>(cam_node, "camera_yaml", camera_yaml_);
  getValue<double>(cam_node, "scale_camera_model", scale_camera_model_);

  auto loc_node = node["localization"];
  if (!loc_node) {
    throw std::runtime_error("could not find 'localization' in node");
  }
  getValue<std::string>(loc_node, "vocabulary", vocabulary_);
  getValue<std::string>(loc_node, "pre_saved_images_one",
                        pre_saved_images_one_);
  getValue<std::string>(loc_node, "pre_saved_images_two",
                        pre_saved_images_two_);
  getValue<std::string>(loc_node, "pre_saved_images_thr",
                        pre_saved_images_thr_);
  getValue<std::string>(loc_node, "pre_saved_images_fou",
                        pre_saved_images_fou_);
  getValue<double>(loc_node, "threshold", threshold_);

  auto websocket_node = node["websocket"];
  if (!websocket_node) {
    throw std::runtime_error("could not find 'websocket' in node");
  }
  getValue<std::string>(websocket_node, "server_address", server_address_);

  auto optimization_node = node["optimization"];
  if (!optimization_node) {
    throw std::runtime_error("could not find 'optimization' in node");
  }
  getValue<int>(optimization_node, "sliding_window_size_local",
                sliding_window_size_local_);
  getValue<int>(optimization_node, "sliding_window_size_global",
                sliding_window_size_global_);
}