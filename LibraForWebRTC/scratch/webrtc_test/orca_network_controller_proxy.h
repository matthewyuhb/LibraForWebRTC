#pragma once

#include <stdint.h>

#include <deque>
#include <memory>
#include <vector>

#include "absl/types/optional.h"
#include "api/network_state_predictor.h"
#include "api/rtc_event_log/rtc_event_log.h"
#include "api/transport/field_trial_based_config.h"
#include "api/transport/network_control.h"
#include "api/transport/network_types.h"
#include "api/transport/webrtc_key_value_config.h"
#include "api/units/data_rate.h"
#include "api/units/data_size.h"
#include "api/units/timestamp.h"
#include "rtc_base/constructor_magic.h"
#include "rtc_base/experiments/field_trial_parser.h"
#include "rtc_base/experiments/rate_control_settings.h"

#include "modules/congestion_controller/goog_cc/acknowledged_bitrate_estimator_interface.h"
#include "modules/congestion_controller/goog_cc/alr_detector.h"
#include "modules/congestion_controller/goog_cc/congestion_window_pushback_controller.h"
#include "modules/congestion_controller/goog_cc/delay_based_bwe.h"
#include "modules/congestion_controller/goog_cc/probe_controller.h"
#include "modules/congestion_controller/goog_cc/send_side_bandwidth_estimation.h"
#include "modules/congestion_controller/goog_cc/goog_cc_network_control.h"
#include "gym_connector.h"

namespace webrtc {

// struct GoogCcConfig {
//   std::unique_ptr<NetworkStateEstimator> network_state_estimator = nullptr;
//   std::unique_ptr<NetworkStatePredictor> network_state_predictor = nullptr;
//   bool feedback_only = false;
// };

class OrcaCcNetworkControllerProxy : public webrtc::NetworkControllerInterface
{
public:
  // WLibraCcNetworkControllerProxy();
  OrcaCcNetworkControllerProxy(webrtc::NetworkControllerConfig config,webrtc::GoogCcConfig goog_cc_config,GymConnector &conn);

  // Called when network availabilty changes.
  webrtc::NetworkControlUpdate OnNetworkAvailability(webrtc::NetworkAvailability msg) override;

  // Called when the receiving or sending endpoint changes address.
  webrtc::NetworkControlUpdate OnNetworkRouteChange(webrtc::NetworkRouteChange msg) override;

  // Called periodically with a periodicy as specified by
  // NetworkControllerFactoryInterface::GetProcessInterval.
  webrtc::NetworkControlUpdate OnProcessInterval(webrtc::ProcessInterval msg) override;

  // Called when remotely calculated bitrate is received.
  webrtc::NetworkControlUpdate OnRemoteBitrateReport(webrtc::RemoteBitrateReport msg) override;

  // Called round trip time has been calculated by protocol specific mechanisms.
  webrtc::NetworkControlUpdate OnRoundTripTimeUpdate(webrtc::RoundTripTimeUpdate msg) override;

  // Called when a packet is sent on the network.
  webrtc::NetworkControlUpdate OnSentPacket(webrtc::SentPacket sent_packet) override;

  // Called when a packet is received from the remote client.
  webrtc::NetworkControlUpdate OnReceivedPacket(webrtc::ReceivedPacket received_packet) override;

  // Called when the stream specific configuration has been updated.
  webrtc::NetworkControlUpdate OnStreamsConfig(webrtc::StreamsConfig msg) override;

  // Called when target transfer rate constraints has been changed.
  webrtc::NetworkControlUpdate OnTargetRateConstraints(webrtc::TargetRateConstraints constraints) override;

  // Called when a protocol specific calculation of packet loss has been made.
  webrtc::NetworkControlUpdate OnTransportLossReport(webrtc::TransportLossReport msg) override;

  // Called with per packet feedback regarding receive time.
  webrtc::NetworkControlUpdate OnTransportPacketsFeedback(webrtc::TransportPacketsFeedback report) override;

  // Called with network state estimate updates.
  webrtc::NetworkControlUpdate OnNetworkStateEstimate(webrtc::NetworkStateEstimate msg) override;

  webrtc::NetworkControlUpdate GetNetworkState(webrtc::Timestamp at_time) const;
private:
  webrtc::NetworkControlUpdate GetUpdate(webrtc::Timestamp at_time) const;

  GymConnector &gym_conn_;

  friend class GoogCcStatePrinter;
  std::vector<webrtc::ProbeClusterConfig> ResetConstraints(
      webrtc::TargetRateConstraints new_constraints);
  void ClampConstraints();
  void MaybeTriggerOnNetworkChanged(webrtc::NetworkControlUpdate* update,
                                    webrtc::Timestamp at_time);
  void UpdateCongestionWindowSize();
  webrtc::PacerConfig GetPacingRates(webrtc::Timestamp at_time) const;
  const webrtc::FieldTrialBasedConfig trial_based_config_;

  const webrtc::WebRtcKeyValueConfig* const key_value_config_;
  webrtc::RtcEventLog* const event_log_;
  const bool packet_feedback_only_;
  webrtc::FieldTrialFlag safe_reset_on_route_change_;
  webrtc::FieldTrialFlag safe_reset_acknowledged_rate_;
  const bool use_min_allocatable_as_lower_bound_;
  const bool ignore_probes_lower_than_network_estimate_;
  const bool limit_probes_lower_than_throughput_estimate_;
  const webrtc::RateControlSettings rate_control_settings_;
  const bool loss_based_stable_rate_;

  const std::unique_ptr<webrtc::ProbeController> probe_controller_;
  const std::unique_ptr<webrtc::CongestionWindowPushbackController>
      congestion_window_pushback_controller_;

  std::unique_ptr<webrtc::SendSideBandwidthEstimation> bandwidth_estimation_;
  std::unique_ptr<webrtc::AlrDetector> alr_detector_;
  std::unique_ptr<webrtc::ProbeBitrateEstimator> probe_bitrate_estimator_;
  std::unique_ptr<webrtc::NetworkStateEstimator> network_estimator_;
  std::unique_ptr<webrtc::NetworkStatePredictor> network_state_predictor_;
  std::unique_ptr<webrtc::DelayBasedBwe> delay_based_bwe_;
  std::unique_ptr<webrtc::AcknowledgedBitrateEstimatorInterface>
      acknowledged_bitrate_estimator_;

  absl::optional<webrtc::NetworkControllerConfig> initial_config_;

  webrtc::DataRate min_target_rate_ = webrtc::DataRate::Zero();
  webrtc::DataRate min_data_rate_ = webrtc::DataRate::Zero();
  webrtc::DataRate max_data_rate_ = webrtc::DataRate::PlusInfinity();
  absl::optional<webrtc::DataRate> starting_rate_;

  bool first_packet_sent_ = false;

  absl::optional<webrtc::NetworkStateEstimate> estimate_;

  webrtc::Timestamp next_loss_update_ = webrtc::Timestamp::MinusInfinity();
  int lost_packets_since_last_loss_update_ = 0;
  int expected_packets_since_last_loss_update_ = 0;

  std::deque<int64_t> feedback_max_rtts_;

  webrtc::DataRate last_loss_based_target_rate_;
  webrtc::DataRate last_pushback_target_rate_;
  webrtc::DataRate last_stable_target_rate_;

  absl::optional<uint8_t> last_estimated_fraction_loss_ = 0;
  webrtc::TimeDelta last_estimated_round_trip_time_ = webrtc::TimeDelta::PlusInfinity();
  webrtc::Timestamp last_packet_received_time_ = webrtc::Timestamp::MinusInfinity();

  double pacing_factor_;
  webrtc::DataRate min_total_allocated_bitrate_;
  webrtc::DataRate max_padding_rate_;
  webrtc::DataRate max_total_allocated_bitrate_;

  bool previously_in_alr_ = false;

  absl::optional<webrtc::DataSize> current_data_window_;

  RTC_DISALLOW_IMPLICIT_CONSTRUCTORS(OrcaCcNetworkControllerProxy);
};

} //namespace webrtc