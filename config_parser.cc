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
#include <iostream>
#include <yaml-cpp/yaml.h>

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
  getValue<std::string>(loc_node, "pre_saved_images", pre_saved_images_);
}