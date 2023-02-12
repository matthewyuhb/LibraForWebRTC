#include "network_estimator_proxy.h"
#include <iostream>
#include "ns3/simulator.h"

using namespace webrtc;

NetworkStateEstimatorProxy::NetworkStateEstimatorProxy(GymConnector &conn) : gym_conn_(conn) {
}

absl::optional<NetworkStateEstimate> NetworkStateEstimatorProxy::GetCurrentEstimate() {
  return absl::optional<NetworkStateEstimate>();
}

void NetworkStateEstimatorProxy::OnTransportPacketsFeedback(const TransportPacketsFeedback& feedback) {
  // std::cout<<this<<" OnTransportPacketsFeedback!"<<std::endl;

}

void NetworkStateEstimatorProxy::OnReceivedPacket(const PacketResult& packet_result) {
  // std::cout<<this<<" OnReceivedPacket!"<<std::endl;

}

void NetworkStateEstimatorProxy::OnRouteChange(const NetworkRouteChange& route_change) {
}

void NetworkStateEstimatorProxy::OnReceivedPacketDetail(
  int64_t arrival_time_ms,
  size_t payload_size,
  const RTPHeader& header,
  const PacketResult& packet_result) {
  // std::cout<<"Time:"<<ns3::Simulator::Now ().GetSeconds ()<<" OnReceivedPacketDetail!"<<std::endl;
  gym_conn_.ProduceStates(arrival_time_ms, payload_size, header, packet_result);
}
