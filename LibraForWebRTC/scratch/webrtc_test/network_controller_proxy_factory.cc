#include "network_controller_proxy_factory.h"
#include <iostream>
using namespace webrtc;

NetworkControllerProxyFactory::NetworkControllerProxyFactory(GymConnector &conn,std::string rl_type) :
    gym_conn_(conn) {
        rl_type_=rl_type;
}

std::unique_ptr<NetworkControllerInterface> NetworkControllerProxyFactory::Create(
    NetworkControllerConfig config) {
    // if (event_log_)
    // config.event_log = event_log_;
    GoogCcConfig goog_cc_config;
    goog_cc_config.feedback_only = factory_config_.feedback_only;
    if (factory_config_.network_state_estimator_factory) {
        RTC_DCHECK(config.key_value_config);
        goog_cc_config.network_state_estimator =
            factory_config_.network_state_estimator_factory->Create(
                config.key_value_config);
    }
    if (factory_config_.network_state_predictor_factory) {
        goog_cc_config.network_state_predictor =
            factory_config_.network_state_predictor_factory
                ->CreateNetworkStatePredictor();
    }
//   return std::make_unique<GoogCcNetworkController>(config,
//                                                    std::move(goog_cc_config));
    if(rl_type_=="rl"){
        // std::cout<<"matthew: run clean-stale RL!"<<std::endl;
        return std::make_unique<NetworkControllerProxy>(gym_conn_);
    }else if(rl_type_=="orca"){
        // std::cout<<"matthew: run orca!"<<std::endl;
        return std::make_unique<webrtc::OrcaCcNetworkControllerProxy>(config,std::move(goog_cc_config),gym_conn_);
    }else{
        return std::make_unique<NetworkControllerProxy>(gym_conn_);
    }
    
}

TimeDelta NetworkControllerProxyFactory::GetProcessInterval() const {
    const int64_t kUpdateIntervalMs = 25;
    return TimeDelta::Millis(kUpdateIntervalMs);
}
