/*
 * @Author: Dongying
 * @Date: 2021-03-04 11:03:54
 * @LastEditTime: 2021-03-16 11:28:14
 * @LastEditors: Please set LastEditors
 * @Description: 0.1
 * @FilePath: /WebSocket2/websocket_endpoint.cpp
 */

#include "websocket_endpoint.h"

typedef websocketpp::client<websocketpp::config::asio_client> ws_client;

namespace MySocket {

class connection_metadata {
 public:
  typedef websocketpp::lib::shared_ptr<connection_metadata> ptr;

  connection_metadata(websocketpp::connection_hdl hdl, std::string uri)
      : m_hdl(hdl), m_status("Connecting"), m_uri(uri), m_server("N/A") {}

  void on_open(ws_client *client, websocketpp::connection_hdl hdl) {
    m_status = "Open";

    ws_client::connection_ptr con = client->get_con_from_hdl(hdl);
    m_server = con->get_response_header("Server");
  }

  // if connection failed, the function will be invoke.
  void on_fail(ws_client *client, websocketpp::connection_hdl hdl) {
    m_status = "Failed";

    ws_client::connection_ptr con = client->get_con_from_hdl(hdl);
    m_server = con->get_response_header("Server");
    m_error_reason = con->get_ec().message();
  }

  void on_close(ws_client *client, websocketpp::connection_hdl hdl) {
    m_status = "Closed";
    ws_client::connection_ptr con = client->get_con_from_hdl(hdl);
    std::stringstream s;
    s << "close code: " << con->get_remote_close_code() << " ("
      << websocketpp::close::status::get_string(con->get_remote_close_code())
      << "), close reason: " << con->get_remote_close_reason();
    m_error_reason = s.str();
  }
  // 获取收到的信息
  void on_message(websocketpp::connection_hdl, ws_client::message_ptr msg) {
    if (msg->get_opcode() == websocketpp::frame::opcode::text) {
      // m_messages.push_back("<< " + msg->get_payload());
      m_messages.push_back(msg->get_payload());
    } else {
      // m_messages.push_back("<< " +
      // websocketpp::utility::to_hex(msg->get_payload()));
      m_messages.push_back(websocketpp::utility::to_hex(msg->get_payload()));
    }
  }

  websocketpp::connection_hdl get_hdl() const { return m_hdl; }

  std::string get_status() const { return m_status; }

  std::string get_uri() const { return m_uri; }

  void record_sent_message(std::string message) {
    // m_messages.push_back(">> " + message);
    m_messages.push_back(message);
  }

  friend std::ostream &operator<<(std::ostream &out,
                                  connection_metadata const &data);

  std::vector<std::string> GetMessage() { return m_messages; }

 private:
  websocketpp::connection_hdl m_hdl;
  std::string m_status;
  std::string m_uri;
  std::string m_server;
  std::string m_error_reason;
  // 发送的消息不存，只存接受到的消息
  std::vector<std::string> m_messages;
};

std::ostream &operator<<(std::ostream &out, connection_metadata const &data) {
  out << "> URI: " << data.m_uri << "\n"
      << "> Status: " << data.m_status << "\n"
      << "> Remote Server: "
      << (data.m_server.empty() ? "None Specified" : data.m_server) << "\n"
      << "> Error/close reason: "
      << (data.m_error_reason.empty() ? "N/A" : data.m_error_reason) << "\n";
  out << "> Messages Processed: (" << data.m_messages.size() << ") \n";

  std::vector<std::string>::const_iterator it;
  for (it = data.m_messages.begin(); it != data.m_messages.end(); ++it) {
    out << *it << "\n";
  }

  return out;
}

ws_client g_wsEndPoint;
connection_metadata::ptr g_wsClientConnection;

websocketpp::lib::shared_ptr<websocketpp::lib::thread> g_threadWS;

websocket_endpoint::websocket_endpoint() {
  g_wsEndPoint.clear_access_channels(websocketpp::log::alevel::all);
  g_wsEndPoint.clear_error_channels(websocketpp::log::elevel::all);

  g_wsEndPoint.init_asio();
  g_wsEndPoint.start_perpetual();

  g_threadWS = websocketpp::lib::make_shared<websocketpp::lib::thread>(
      &ws_client::run, &g_wsEndPoint);
}

websocket_endpoint::~websocket_endpoint() {
  g_wsEndPoint.stop_perpetual();

  if (g_wsClientConnection->get_status() == "Open") {
    // Only close open connections
    websocketpp::lib::error_code ec;
    g_wsEndPoint.close(g_wsClientConnection->get_hdl(),
                       websocketpp::close::status::going_away, "", ec);
    if (ec) {
      std::cout << "> Error closing ws connection "
                << g_wsClientConnection->get_uri() << " :" << ec.message()
                << std::endl;
    }
  }

  g_threadWS->join();
}

int websocket_endpoint::connect(std::string const &uri) {
  websocketpp::lib::error_code ec;

  ws_client::connection_ptr pConnection = g_wsEndPoint.get_connection(uri, ec);

  if (ec) {
    std::cout << "> Connect initialization error: " << ec.message()
              << std::endl;
    return -1;
  }

  g_wsClientConnection = websocketpp::lib::make_shared<connection_metadata>(
      pConnection->get_handle(), uri);

  pConnection->set_open_handler(websocketpp::lib::bind(
      &connection_metadata::on_open, g_wsClientConnection, &g_wsEndPoint,
      websocketpp::lib::placeholders::_1));
  pConnection->set_fail_handler(websocketpp::lib::bind(
      &connection_metadata::on_fail, g_wsClientConnection, &g_wsEndPoint,
      websocketpp::lib::placeholders::_1));
  pConnection->set_close_handler(websocketpp::lib::bind(
      &connection_metadata::on_close, g_wsClientConnection, &g_wsEndPoint,
      websocketpp::lib::placeholders::_1));
  pConnection->set_message_handler(websocketpp::lib::bind(
      &connection_metadata::on_message, g_wsClientConnection,
      websocketpp::lib::placeholders::_1, websocketpp::lib::placeholders::_2));

  g_wsEndPoint.connect(pConnection);

  std::cout << "Websocket连接成功" << std::endl;

  return 0;
}

void close(websocketpp::close::status::value code, std::string reason) {
  websocketpp::lib::error_code ec;

  g_wsEndPoint.close(g_wsClientConnection->get_hdl(), code, reason, ec);
  if (ec) {
    std::cout << "> Error initiating close: " << ec.message() << std::endl;
  }
}

void websocket_endpoint::close() {
  if (g_wsClientConnection->get_status() == "Open") {
    int close_code = websocketpp::close::status::normal;
    MySocket::close(close_code, "");
  }
}

void websocket_endpoint::send(float posi_one, float posi_two, int id_one,
                              int id_two) {
  websocketpp::lib::error_code ec;

  // 按照此格式可以解析
  json j1 = {{"position",
              {{{"id", id_one}, {"position", posi_one}},
               {{"id", id_two}, {"position", posi_two}}}}};
  std::string j1_string = j1.dump();
  json j = {{"type", "crane_info"}, {"data", j1_string}};
  std::cout << std::setw(4) << j << std::endl;
  std::string j_string = j.dump();

  // std::string first = " ";
  // std::string a1 = "{\"type\":\"crane_info\",\"data\":";
  // std::string a2 = "\"";
  // std::string a3 = "{\\\"position\\\":[{\\\"position\\\":";
  // std::string a5 = ",\\\"id\\\":";
  // std::string a7 = "},{\\\"position\\\":";
  // std::string a9 = ",\\\"id\\\":";
  // std::string a11 = "}]}\"}";
  // std::string end = "";
  // std::string out = first + a1 + a2 + a3 + std::to_string(position3) + a5 +
  // std::to_string(id3) + a7 + std::to_string(position4) + a9 +
  // std::to_string(id4) + a11 + end;

  g_wsEndPoint.send(g_wsClientConnection->get_hdl(), j_string,
                    websocketpp::frame::opcode::text, ec);
  if (ec) {
    std::cout << "> Error sending message: " << ec.message() << std::endl;
    return;
  }

  // g_wsClientConnection->record_sent_message(j_string);
}

void websocket_endpoint::show() {
  std::cout << *g_wsClientConnection << std::endl;
}
// clang-format off
	bool websocket_endpoint::parsing()
	{	
		int cnt =0;
		if (!(g_wsClientConnection->get_status()=="Open"))
		{
			cnt++;
			std::cout << "-----------------" << cnt << std::endl;
		}
		
		// 从服务器端接受到的信息存储在向量容器中
		std::vector<std::string> messages;
		messages = g_wsClientConnection->GetMessage();
		std::cout << "messages size:   " << messages.size() << std::endl;
		
		std::vector<std::string>::const_iterator it;
    // 对应于天车id,rtsp流地址
		
    // 实际上这里vector中只有一组数据，及从服务器端接收到的数据，仅有一次
		for (it = messages.begin(); it != messages.end(); ++it) {
			std::string j2 = *it;
			auto j3 = json::parse(j2);
			// 可以通过这种方式获取json字符串中的数据，可以读取对应id及视频码流地址的字符串等
			int crane_id0 = j3["crane_data"][0]["type_id"];
			std::cout << "----------------------------" << crane_id0 << "------------------------------" << std::endl;
			std::string crane_rtsp0 = j3["crane_data"][0]["video_urls"][0];
			std::cout << "----------------------------" << crane_rtsp0 << "------------------------------" << std::endl;
      crane_info_.push_back(std::make_pair(crane_id0,crane_rtsp0));

      int crane_id1 = j3["crane_data"][1]["type_id"];
			std::cout << "----------------------------" << crane_id1 << "------------------------------" << std::endl;
			std::string crane_rtsp1 = j3["crane_data"][1]["video_urls"][0];
			std::cout << "----------------------------" << crane_rtsp1 << "------------------------------" << std::endl;
      crane_info_.push_back(std::make_pair(crane_id1,crane_rtsp1));

      int crane_id2 = j3["crane_data"][2]["type_id"];
			std::cout << "----------------------------" << crane_id2 << "------------------------------" << std::endl;
			std::string crane_rtsp2 = j3["crane_data"][2]["video_urls"][0];
			std::cout << "----------------------------" << crane_rtsp2 << "------------------------------" << std::endl;
      crane_info_.push_back(std::make_pair(crane_id2,crane_rtsp2));

      int crane_id3 = j3["crane_data"][3]["type_id"];
			std::cout << "----------------------------" << crane_id3 << "------------------------------" << std::endl;
			std::string crane_rtsp3 = j3["crane_data"][3]["video_urls"][0];
			std::cout << "----------------------------" << crane_rtsp3 << "------------------------------" << std::endl;
      crane_info_.push_back(std::make_pair(crane_id3,crane_rtsp3));
			// 对j3输出后就是json字符串的格式，显示比较明显
			std::cout << std::setw(4) << j3 << std::endl;

			// for (auto& el : j3.items()) {
			// 	// if (el.key() == "crane_data" )
			// 	// {
			// 	// 	json j5 = el.value();
			// 	// 	std::string test_str;
    		// 	// 	test_str = j5.dump();
			// 	// 	json j = test_str;
			// 	// 	auto j4 = json::parse(std::string(j));
			// 	// 	std::cout << el.key() << ":" << std::endl;
			// 	// 	for (auto& el : j4.items()) {
			// 	// 		std::cout << el.key() << " : " << el.value() << "\n";
			// 	// 	}
			// 	// 	continue;
			// 	// 	// json j2 = el;
			// 	// 	// auto j4 = json::parse(el);
			// 	// 	// std::cout << el.key() << " : " << j4.value() << "\n";
			// 	// }
			// 	std::cout << el.key() << " : " << el.value() << "\n";
			// }
		}
		return true;
	}
// clang-format on

void websocket_endpoint::craneInfosave() {
  std::ofstream fcrane_info;
  fcrane_info.open("./conf/crane_info.txt");
  for (auto &f : crane_info_) {
    fcrane_info << f.first << " " << f.second << std::endl;
  }
  fcrane_info.close();
  std::cout << "[INFO]: The crane_info saved! " << std::endl;
}
}