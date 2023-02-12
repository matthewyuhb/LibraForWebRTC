#include "gym_connector.h"

#include "ns3/simulator.h"

#include <boost/lexical_cast.hpp>

#include <string>

#include <vector>

#include <iostream>

using namespace webrtc;

constexpr char kBandwidthQueuePrefix[] = "/bandwidth_";
constexpr char kStatsQueuePrefix[] = "/stats_";
constexpr char kZmqTypePrefix[] = "ipc:///tmp/";
constexpr char kGymExitFlag[] = "Bye";
constexpr double kPacingFactor = 2.5f;

std::vector<std::string> Stringsplit(std::string str,const char split)
{
    std::vector<std::string> ans;
	std::istringstream iss(str);	// 输入流
	std::string token;			// 接收缓冲区
	while (getline(iss, token, split))	// 以split为分隔符
	{
		// cout << token << endl; // 输出
        ans.push_back(token);
	}
    return ans;
}

GymConnector::GymConnector(
    const std::string &gym_id,
    std::uint64_t report_interval_ms,
    BandwidthType init_bandwidth) :
    current_bandwidth_(init_bandwidth),
    report_interval_ms_(report_interval_ms),
    gym_id_(gym_id),
    // m_port_(m_port),
    zmq_sock_(zmq_ctx_, zmq::socket_type::rep),
    zmq_wait_reply_(false),
    rl_updated_(false),
    w_state_(WLibraState::Ordinary){
    // data_window_(4){
    // zmq_sock_.bind(kZmqTypePrefix + gym_id_);
    // port = zmq_sock_.bind('tcp://*', min_port=5001, max_port=10000, max_tries=100);
    // std::cout<<"[C++]GymConnector:"<<m_port_<<std::endl;
    zmq_sock_.bind(kZmqTypePrefix + gym_id_);
    // std::string connectAddr = "tcp://localhost:" + std::to_string(m_port_);
    // zmq_connect ((void*)zmq_sock_, connectAddr.c_str());
}

GymConnector::~GymConnector() {
    std::cout<<"matthew:~GymConnector()"<<std::endl;
    if (!zmq_wait_reply_){
        try{
            zmq::message_t msg;
            zmq_sock_.recv(&msg);//msg是目标带宽
        }
        catch(zmq::error_t()){
            std::cout<<"no msg from zmq python side anymore ~"<<std::endl;
        }
        zmq_wait_reply_ = true;
    }
    
    if (zmq_wait_reply_) {
        const std::string exit_flag(kGymExitFlag);
        try{
            // std::cout<<"matthew:exit_flag.c_str():"<<exit_flag.c_str()<<std::endl;
            zmq_sock_.send(exit_flag.c_str(), exit_flag.length());
        }
        catch(zmq::error_t()){
            std::cout<<"error when ~"<<std::endl;
        }
    }
    // zmq_sock_.unbind(kZmqTypePrefix + gym_id_);
}

void GymConnector::Step(std::uint64_t delay_ms) {
    zmq::message_t msg;
    // NS_LOG_UNCOND("waiting for msg for c++");
    // std::cout<<"[C++]waiting for msg from python"<<std::endl;
    zmq_sock_.recv(&msg);//msg是目标带宽
    
    BandwidthType bandwidth;
    std::string msg_str(static_cast<char *>(msg.data()), msg.size());
    std::vector<std::string> v_msgs = Stringsplit(msg_str,',');
    std::string bandwidth_str = v_msgs[0];
    std::string state_str = v_msgs[1];
    // std::string EI_sequence = v_msgs[2];
    // EI_sequence_ = 
    // std::cout<<"C++:received_bandwidth:"<<bandwidth_str<<" state:"<<state_str<<std::endl;
    // std::cout<<"matthew:current time:"<<ns3::Simulator::Now().GetMilliSeconds()<<std::endl;
    // bandwidth_str = "100000";
    try {
        bandwidth = boost::lexical_cast<BandwidthType>(bandwidth_str);
    }
    catch(const boost::bad_lexical_cast& e)
    {
        const std::string error_msg = "Wrong bandwidth " + bandwidth_str;
        zmq_sock_.send(error_msg.c_str(), error_msg.length());
        ns3::Simulator::Stop();
        return;
    }
    // webrtc::DataRate data_rate = ncu_.pacer_config->data_rate();
    // std::cout<<"matthew:get current data window:"<<ncu_.pacer_config->data_rate()<<std::endl;
    // SetBandwidth(bandwidth,WLibraState(atoi(state_str.c_str())));
    
    zmq_wait_reply_ = true;

    // std::cout<<"base_owd_:"<<base_owd_<<" delay_ms:"<<delay_ms<<std::endl;
    ns3::Simulator::Schedule(ns3::MilliSeconds(base_owd_), &GymConnector::SetBandwidth, this,bandwidth,WLibraState(atoi(state_str.c_str())));
    // std::cout<<"matthew:"<<ns3::Simulator::Now().GetMilliSeconds()<<"schedule sechedule delay:"<<base_owd_<<std::endl;
    ns3::Simulator::Schedule(ns3::MilliSeconds(delay_ms), &GymConnector::ReportStats, this);
}

void GymConnector::ReportStats() {
    auto stats = ConsumeStats();
    nlohmann::json j = stats;
    // std::cout<<"-------------Delay-------------------"<<std::endl;
    u_int64_t new_owd = 0;
    for(auto st:stats){
        // std::cout<<"send time:"<<st["send_time_ms"]<<" arrival time:"<<st["arrival_time_ms"]<<std::endl;
        // std::cout<<int(st["arrival_time_ms"])-int(st["send_time_ms"])<<std::endl;
        new_owd+=(int(st["arrival_time_ms"])-int(st["send_time_ms"]));
        base_owd_ = std::min(base_owd_,(uint64_t(st["arrival_time_ms"])-uint64_t(st["send_time_ms"])));
        // j["arrival_time_ms"] = packet_result.receive_time.ms();
    }
    if(stats.size())new_owd /=stats.size();
    if(new_owd){
        owd_ = new_owd;
    }
    const auto json_str = j.dump();
    // std::cout<<"matthew:NS3:id:"<<gym_id_<<" time:"<<ns3::Simulator::Now().GetMilliSeconds()<<std::endl;//<<" ms"<<std::endl<<"send json str:"<<json_str<<std::endl;
    try{
        zmq_sock_.send(json_str.c_str(), json_str.length());
    }
    catch(zmq::error_t()){
        ns3::Simulator::Stop();
        return;
    }
    SetRlUpdate(true); //matthew: used to control the orca manner
    zmq_wait_reply_ = false;
    // std::cout<<"base_owd delay:"<<base_owd_<<std::endl;
    uint64_t delay_ms;
    // std::cout<<"w_state_:"<<w_state_<<std::endl;
    if(report_interval_ms_>0){
        delay_ms = report_interval_ms_;
    }else{ 
        if(w_state_==WLibraState::Ordinary){
            delay_ms = base_owd_+owd_;
        }else{
            delay_ms = (base_owd_+owd_)/2;
        }
    }
    // std::cout<<"delay_ms:"<<delay_ms<<std::endl;
    
    ns3::Simulator::ScheduleNow(&GymConnector::Step, this, delay_ms);
}

// int GymConnector::GetEISequence(){
//     return EI_sequence_;
// }

void GymConnector::SetBandwidth(BandwidthType bandwidth,WLibraState state) {
    w_state_ = state;
    // std::cout<<"NS3:SettingBandwidth:state:"<<w_state_<<std::endl;
    // EI_sequence_ = EI_sequence;
    if (bandwidth == current_bandwidth_) {
        return;
    }
    {
        std::unique_lock<std::shared_timed_mutex> guard(mutex_bandiwidth_);
        current_bandwidth_ = bandwidth;
        // updated_=true;
    }
    
}

NetworkControlUpdate GymConnector::GetNetworkControlUpdate(const Timestamp& at_time) const {
    BandwidthType current_bandwidth = {0};

    {
        std::shared_lock<std::shared_timed_mutex> guard(mutex_bandiwidth_);
        current_bandwidth = current_bandwidth_;
    }

    NetworkControlUpdate update;
    DataRate target_rate = DataRate::BitsPerSec(current_bandwidth);

    update.target_rate = TargetTransferRate();
    update.target_rate->network_estimate.at_time = at_time;
    update.target_rate->network_estimate.bandwidth = target_rate;
    update.target_rate->network_estimate.loss_rate_ratio = 0;
    update.target_rate->network_estimate.round_trip_time = TimeDelta::Millis(0);
    update.target_rate->network_estimate.bwe_period = TimeDelta::Seconds(3);
    update.target_rate->at_time = at_time;
    update.target_rate->target_rate = target_rate;

    update.pacer_config = PacerConfig();
    update.pacer_config->at_time = at_time;
    update.pacer_config->time_window = TimeDelta::Seconds(1);
    update.pacer_config->data_window =  kPacingFactor * target_rate * update.pacer_config->time_window;
    update.pacer_config->pad_window = DataRate::BitsPerSec(0) * update.pacer_config->time_window;
    // std::cout<<"matthew:RL set data window:"<<update.pacer_config->data_window.bytes()<<std::endl;
    return update;
}

// void GymConnector::SetNcu(webrtc::NetworkControlUpdate ncu){
//     ncu_=ncu;
// }

// webrtc::NetworkControlUpdate GymConnector::GetNcu(){
//     return ncu_;
// }
webrtc::DataRate GymConnector::GetDataRate(){
    return target_rate_;
}

void GymConnector::SetDataRate(webrtc::DataRate data_rate){
    target_rate_ = data_rate;
}
WLibraState GymConnector::GetWState(){
    return w_state_;
}

void GymConnector::SetRlUpdate(bool rl_updated){
    rl_updated_=rl_updated;
}

bool GymConnector::GetRlUpdate(){
    return rl_updated_;
}


void GymConnector::ProduceStates(
    int64_t arrival_time_ms,
    size_t payload_size,
    const RTPHeader& header,
    const PacketResult& packet_result) {

    nlohmann::json j;
    j["send_time_ms"] = packet_result.sent_packet.send_time.ms();
    j["arrival_time_ms"] = packet_result.receive_time.ms();
    j["payload_type"] = header.payloadType;
    j["sequence_number"] = header.sequenceNumber;
    j["ssrc"] = header.ssrc;
    j["padding_length"] = header.paddingLength;
    j["header_length"] = header.headerLength;
    j["payload_size"] = payload_size;
    j["data_window"] = target_rate_.bytes_per_sec()*8;
    j["time_now"] = ns3::Simulator::Now().GetMilliSeconds();
    // std::cout<<"ns3:current time:"<<ns3::Simulator::Now().GetMilliSeconds()<<std::endl;
    // std::cout<<"matthew:target_rate_:"<<j["data_window"]<<std::endl;
    // const std::string stats = j.dump();
    {
        std::unique_lock<std::mutex> guard(mutex_stats_);
        stats_.push_back(j);
    }
}

std::list<nlohmann::json> GymConnector::ConsumeStats() {
    std::list<nlohmann::json> stats;
    {
        std::unique_lock<std::mutex> guard(mutex_stats_);
        std::swap(stats, stats_);
    }
    return stats;
}
