#pragma once

#include "network_controller_proxy.h"
// #include "wlibra_network_controller_proxy.h"
#include "orca_network_controller_proxy.h"
#include "gym_connector.h"

#include "api/transport/network_control.h"
#include "api/transport/goog_cc_factory.h"

// class RtcEventLog;

// struct GoogCcFactoryConfig {
//   std::unique_ptr<webrtc::NetworkStateEstimatorFactory>
//       network_state_estimator_factory = nullptr;
//   webrtc::NetworkStatePredictorFactoryInterface* network_state_predictor_factory =
//       nullptr;
//   bool feedback_only = false;
// };


class NetworkControllerProxyFactory : public webrtc::NetworkControllerFactoryInterface {
public:
  NetworkControllerProxyFactory(GymConnector &conn,std::string rl_type);

  // Used to create a new network controller, requires an observer to be
  // provided to handle callbacks.
  std::unique_ptr<webrtc::NetworkControllerInterface> Create(
      webrtc::NetworkControllerConfig config) override;

  // Returns the interval by which the network controller expects
  // OnProcessInterval calls.
  virtual webrtc::TimeDelta GetProcessInterval() const override;

protected:
  // RtcEventLog* const event_log_ = nullptr;
  webrtc::GoogCcFactoryConfig factory_config_;

private:
  GymConnector &gym_conn_;
  std::string rl_type_;
};
