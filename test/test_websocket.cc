/**
 * @file:
 * @author: Dongying (yudong2817@sina.com)
 * @brief:
 * @version:
 * @date:  2021-03-08
 * @copyright: Copyright (c) 2021
 */

#include <iostream>
#include <sstream>
#include <string>
#include <yaml-cpp/yaml.h>
#include "config_parser.h"
#include "websocket_endpoint.h"

int main(int argc, char **argv) {
  std::string config_yaml = "./conf/pipeline_online.yaml";
  if(argc == 2){
    config_yaml = argv[1];
  }
  auto config_parser = ConfigParser(config_yaml);
  std::string server_address = config_parser.server_address_;
  bool done = false;
  MySocket::websocket_endpoint endpoint;

  endpoint.connect(server_address);
  // endpoint.connect("ws://192.168.1.134:18001/ws?client_type=edge1&id=1");

  // 可以接受到字符串并且正常解析后，说明与服务器通信状态正常，才可进行后去数据发送
  while (!done) {
    // 返回视频的http地址
    bool link_status = endpoint.parsing();
    if (!link_status)
      continue;
    else {
      // 添加文件数据信息写入txt的操作，存储顺序：id rtsp流地址
      endpoint.craneInfosave();
      done = true;
    }
  }

  // endpoint.send(200, 200, 1, 2);

  // while (!done) {
  // 	std::cout << "Enter Command: ";
  // 	std::getline(std::cin, input);

  // 	if (input == "quit") {
  // 		done = true;
  // 	}
  // 	else if (input.substr(0, 4) == "send") {
  // 		std::stringstream ss(input);

  // 		std::string cmd;
  // 		std::string message;

  // 		ss >> cmd;
  // 		std::getline(ss, message);
  // 		endpoint.send(message);
  // 	}
  // 	else if (input.substr(0, 4) == "show") {
  // 		endpoint.show();
  // 		endpoint.parsing();
  // 	}
  // 	else {
  // 		std::cout << "> Unrecognized Command" << std::endl;
  // 	}
  // }

  //不需要输入，当主线程关闭时关闭该客户端
  // endpoint.close();

  return 0;
}
