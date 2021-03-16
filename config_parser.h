/**
 * @file config_parser.h
 * @author xiaotaw (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2021-03-08
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#include <string>

class ConfigParser {
public:
  ConfigParser(const std::string &config_yaml);

  bool transpose_image_;
  
  std::string camera_yaml_;
  double scale_camera_model_;

  std::string vocabulary_;
  std::string pre_saved_images_;
  double threshold_;
  std::string server_address_;
};