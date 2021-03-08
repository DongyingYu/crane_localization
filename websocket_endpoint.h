/*
 * @Author: Dongying
 * @Date: 2021-03-04 11:03:16
 * @LastEditTime: 2021-03-08 09:48:51
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /WebSocket2/websocket_endpoint.h
 */
#pragma once
 
#include <nlohmann/json.hpp>
#include <iomanip>
#include <websocketpp/config/asio_no_tls_client.hpp>
#include <websocketpp/client.hpp>
#include <websocketpp/common/thread.hpp>
#include <websocketpp/common/memory.hpp>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <sstream> 

namespace MySocket
{
	using json = nlohmann::json;
	class websocket_endpoint {
	public:
		websocket_endpoint();
		~websocket_endpoint();
 
		int connect(std::string const & uri);
		void close();
 
		void send(float posi_one, float posi_two, int id_one, int id_two);
		void show();
		std::string parsing();
	};
}