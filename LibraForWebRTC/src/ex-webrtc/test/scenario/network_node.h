/*
 *  Copyright 2018 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#ifndef TEST_SCENARIO_NETWORK_NODE_H_
#define TEST_SCENARIO_NETWORK_NODE_H_

#include <deque>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "api/call/transport.h"
#include "api/units/timestamp.h"
#include "call/call.h"
#include "call/simulated_network.h"
#include "rtc_base/constructor_magic.h"
#include "rtc_base/copy_on_write_buffer.h"
#include "rtc_base/task_queue.h"
#include "test/network/network_emulation.h"
#include "test/scenario/column_printer.h"
#include "test/scenario/scenario_config.h"
#include "test/scenario/transport_base.h"

namespace webrtc {
namespace test {

class SimulationNode {
 public:
  SimulationNode(NetworkSimulationConfig config,
                 SimulatedNetwork* behavior,
                 EmulatedNetworkNode* network_node);
  static std::unique_ptr<SimulatedNetwork> CreateBehavior(
      NetworkSimulationConfig config);

  void UpdateConfig(std::function<void(NetworkSimulationConfig*)> modifier);
  void PauseTransmissionUntil(Timestamp until);
  ColumnPrinter ConfigPrinter() const;
  EmulatedNetworkNode* node() { return network_node_; }

 private:
  NetworkSimulationConfig config_;
  SimulatedNetwork* const simulation_;
  EmulatedNetworkNode* const network_node_;
};

class NetworkNodeTransport : public TransportBase {
 public:
  NetworkNodeTransport() {}
  ~NetworkNodeTransport() override;
  void Construct(Clock* sender_clock, Call* sender_call) override;
  bool SendRtp(const uint8_t* packet,
               size_t length,
               const PacketOptions& options) override;
  bool SendRtcp(const uint8_t* packet, size_t length) override;

  void Connect(EmulatedEndpoint* endpoint,
               const rtc::SocketAddress& receiver_address,
               DataSize packet_overhead);
  void Disconnect();

  DataSize packet_overhead() {
    rtc::CritScope crit(&crit_sect_);
    return packet_overhead_;
  }

 private:
  rtc::CriticalSection crit_sect_;
  Clock* sender_clock_{nullptr};
  Call* sender_call_{nullptr};
  EmulatedEndpoint* endpoint_ RTC_GUARDED_BY(crit_sect_) = nullptr;
  rtc::SocketAddress local_address_ RTC_GUARDED_BY(crit_sect_);
  rtc::SocketAddress remote_address_ RTC_GUARDED_BY(crit_sect_);
  DataSize packet_overhead_ RTC_GUARDED_BY(crit_sect_) = DataSize::Zero();
  rtc::NetworkRoute current_network_route_ RTC_GUARDED_BY(crit_sect_);
};
}  // namespace test
}  // namespace webrtc
#endif  // TEST_SCENARIO_NETWORK_NODE_H_
