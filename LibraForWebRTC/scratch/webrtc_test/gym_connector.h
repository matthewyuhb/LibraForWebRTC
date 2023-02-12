#pragma once

#include "api/transport/network_control.h"

#include <zmq.hpp>
#include <nlohmann/json.hpp>

#include <cinttypes>
#include <mutex>
#include <shared_mutex>
#include <list>
#include <string>
enum WLibraState{
    Ordinary,
    EI_c_1,
    EI_r_1,
    EI_c_2,
    EI_r_2
    // SIMU_CONTROLLER,
};


    // Ordinary = 0
    // EI_c_1 = 1 #
    // EI_r_1 = 2 #
    // EI_c_2 = 3 #receive ack of the class action and generate the reward 
    // EI_r_2 = 4 #receive ack of the rl action and generate the reward
class GymConnector {
 public:
  using BandwidthType = std::uint32_t;

  GymConnector(
    const std::string &gym_id = "gym",
    std::uint64_t report_interval_ms = 60,
    BandwidthType init_bandwidth = 0);

  virtual ~GymConnector();

  void Step(std::uint64_t delay_ms = 0);

  void ReportStats();

  void SetBandwidth(BandwidthType bandwidth,WLibraState state);

  void SetRlUpdate(bool rl_updated);
  bool GetRlUpdate();
  // int GetEISequence();
  webrtc::DataRate GetDataRate();
  void SetDataRate(webrtc::DataRate data_rate);
  WLibraState GetWState();

  webrtc::NetworkControlUpdate GetNetworkControlUpdate(const webrtc::Timestamp & at_time) const;

  void ProduceStates(
      int64_t arrival_time_ms,
      size_t payload_size,
      const webrtc::RTPHeader& header,
      const webrtc::PacketResult& packet_result);

  std::list<nlohmann::json> ConsumeStats();

 private:
  std::uint64_t base_owd_=100;//min OWD 主要是为了计算RTT  因为目前我只找到了owd的调用方法，假设下行链路不拥塞的话 RTT=min_OWD+OWD
  std::uint64_t owd_=100;
  BandwidthType current_bandwidth_;
  mutable std::shared_timed_mutex mutex_bandiwidth_;
  std::list<nlohmann::json> stats_;
  std::mutex mutex_stats_;

  const std::uint64_t report_interval_ms_;

  const std::string gym_id_;
  // int m_port_;
  zmq::context_t zmq_ctx_;
  zmq::socket_t zmq_sock_;
  bool zmq_wait_reply_;
  bool rl_updated_;
  // int EI_sequence_;
  // webrtc::NetworkControlUpdate ncu_;
  
  WLibraState w_state_;
  // webrtc::DataSize data_window_ = webrtc::DataSize::Bytes(2 * 1500);
  webrtc::DataRate target_rate_  = webrtc::DataRate::Zero();
};
