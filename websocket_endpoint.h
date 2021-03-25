/*
 * @Author: Dongying
 * @Date: 2021-03-04 11:03:16
 * @LastEditTime: 2021-03-17 10:44:54
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /WebSocket2/websocket_endpoint.h
 */
#pragma once

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <websocketpp/client.hpp>
#include <websocketpp/common/memory.hpp>
#include <websocketpp/common/thread.hpp>
#include <websocketpp/config/asio_no_tls_client.hpp>

namespace MySocket {
using json = nlohmann::json;
class websocket_endpoint {
 public:
  websocket_endpoint();
  ~websocket_endpoint();

  int connect(std::string const& uri);
  void close();

  void send(float posi_one, float posi_two, int id_one, int id_two);
  // 发送一个天车的位置信息
  void send(float posi_one, int id_one);
  void show();
  // 数据解析
  bool parsing();
  void craneInfosave();

 private:
  std::vector<std::pair<int, std::string>> crane_info_;
};
}