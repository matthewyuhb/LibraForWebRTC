#include "gym_connector.h"
#include "network_estimator_proxy_factory.h"
#include "network_controller_proxy_factory.h"
// #include "wlibra_network_controller_proxy_factory.h"
#include "trace_player.h"
#include "api/transport/goog_cc_factory.h"
#include "modules/congestion_controller/pcc/pcc_factory.h"
#include <iostream>
#include <string>
#include <deque>

#include "ns3/webrtc-defines.h"
#include "ns3/core-module.h"
#include "ns3/applications-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/traffic-control-module.h"
#include "ns3/log.h"
#include "ns3/ex-webrtc-module.h"

using namespace ns3;
using namespace std;

NS_LOG_COMPONENT_DEFINE ("Webrtc-Static");

const uint32_t TOPO_DEFAULT_BW     = 3000000;
const uint32_t TOPO_DEFAULT_PDELAY =40;
const uint32_t TOPO_DEFAULT_QDELAY =100;
const uint32_t DEFAULT_PACKET_SIZE = 1000;

static NodeContainer BuildExampleTopo (uint64_t bps,
                                       uint32_t msDelay,
                                       uint32_t msQdelay,
                                       bool enable_random_loss)
{
    NodeContainer nodes;
    nodes.Create (2);

    PointToPointHelper pointToPoint;
    pointToPoint.SetDeviceAttribute ("DataRate", DataRateValue  (DataRate (bps)));
    pointToPoint.SetChannelAttribute ("Delay", TimeValue (MilliSeconds (msDelay)));
    auto bufSize = std::max<uint32_t> (DEFAULT_PACKET_SIZE, bps * msQdelay / 8000);
    pointToPoint.SetQueue ("ns3::DropTailQueue",
                           "MaxSize", QueueSizeValue (QueueSize (QueueSizeUnit::BYTES, bufSize)));

    NetDeviceContainer devices = pointToPoint.Install (nodes);

    InternetStackHelper stack;
    stack.Install (nodes);
    Ipv4AddressHelper address;
    std::string nodeip="10.1.1.0";
    address.SetBase (nodeip.c_str(), "255.255.255.0");
    address.Assign (devices);

    // enable tc in ns3.30
    TrafficControlHelper tch;
    tch.Uninstall (devices);
    if(enable_random_loss){
        std::string errorModelType = "ns3::RateErrorModel";
        ObjectFactory factory;
        factory.SetTypeId (errorModelType);
        Ptr<ErrorModel> em = factory.Create<ErrorModel> ();
        devices.Get (1)->SetAttribute ("ReceiveErrorModel", PointerValue (em));
    }
    return nodes;
}

template<typename AppType>
Ptr<AppType> CreateApp(
  Ptr<Node> node,
  uint16_t port,
  uint64_t start_time_ms,
  uint64_t stop_time_ms,
  WebrtcSessionManager *manager) {
  Ptr<AppType> app = CreateObject<AppType>(manager);
  node->AddApplication(app);
  app->Bind(port);
  app->SetStartTime(ns3::MilliSeconds(start_time_ms));
  app->SetStopTime(ns3::MilliSeconds(stop_time_ms));
  return app;
}


string getTraceName(string trace){
  int right;
  int left;
  for(int i = trace.size()-1;i>0;i--){
    if(trace[i]=='.'){
      right = i;
    }
    if(trace[i]=='/'){
      left = i;
      break;

    }
  }
  return trace.substr(left+1,right-left-1);
}
void ConnectApp(
  Ptr<WebrtcSender> sender,
  Ptr<WebrtcReceiver> receiver,WebrtcTrace *trace=nullptr) {
  auto sender_addr =
    sender->GetNode()->GetObject<ns3::Ipv4>()->GetAddress(1, 0).GetLocal();
  auto receiver_addr =
    receiver->GetNode()->GetObject<ns3::Ipv4>()->GetAddress(1, 0).GetLocal();
  sender->ConfigurePeer(receiver_addr, receiver->GetBindPort());
  receiver->ConfigurePeer(sender_addr, sender->GetBindPort());
  if(trace){
        if(trace->LogFlag()&WebrtcTrace::E_WEBRTC_BW){
            sender->SetBwTraceFuc(MakeCallback(&WebrtcTrace::OnBW,trace));
        }
        if(trace->LogFlag()&WebrtcTrace::E_WEBRTC_RTT){
            // std::cout<<"matthew:Set owd callback:"<<std::endl;
            sender->SetRttTraceFuc(MakeCallback(&WebrtcTrace::OnRTT,trace));
        }
        // receiver->SetOwdTraceFuc(MakeCallback(&WebrtcTrace::OnRTT,trace));
    }
}

static int64_t simStopMilli=1000000;
int64_t appStartMills=1;
float appStopMills=simStopMilli-1;
uint64_t kMillisPerSecond=1000;
uint64_t kMicroPerMillis=1000;

int main(int argc, char *argv[]){

    uint64_t linkBw   = TOPO_DEFAULT_BW;
    uint32_t msDelay  = TOPO_DEFAULT_PDELAY;
    uint32_t msQDelay = TOPO_DEFAULT_QDELAY;

    double loss_rate = 0;

    std::string gym_id("gym");
    std::string mode("simu");
    std::string start_time("");
    std::string episode("0");
    std::string trace_path;
    std::uint64_t report_interval_ms = 0;
    std::uint64_t duration_time_ms = 0;
    std::uint32_t video_height = 1080;
    std::uint32_t video_width = 1920;
    std::uint32_t port_num = 5564;
    std::string congestion_control_algorithm = "gcc";
    double smoothing_coef = 0.9;
    bool do_log = true;
    CommandLine cmd;
    cmd.AddValue("episode","episode",episode);
    cmd.AddValue("loss", "loss",loss_rate);
    cmd.AddValue("start_time","start time",start_time);
    cmd.AddValue("gym_id", "gym id should be unique in global system, the default is gym", gym_id);
    cmd.AddValue("trace_path", "trace file path", trace_path);
    cmd.AddValue("report_interval_ms", "report interval (ms)", report_interval_ms);
    cmd.AddValue("duration_time_ms", "duration time (ms), the default is trace log duration", duration_time_ms);
    cmd.AddValue("video_height", "video height", video_height);
    cmd.AddValue("video_width", "video width", video_width);
    cmd.AddValue("congestion_control_algorithm", "select the congestion control algorithm:0:RL-based 1:GCC 2:PCC", congestion_control_algorithm);
    cmd.AddValue("smoothing_coef", "smoothing_coef parameter of GCC", smoothing_coef);
    cmd.AddValue("m","mode",mode);
    cmd.AddValue("port_num","port_num",port_num);
    cmd.AddValue("log","do log",do_log);
    cmd.Parse (argc, argv);
//切换simu/emu模式，但感觉没有很大变化 add by matthew 526
    TimeControllerType controller_type=TimeControllerType::SIMU_CONTROLLER;
    if (0==mode.compare("simu")){
        webrtc_register_clock();
    }else if(0==mode.compare("emu")){
        controller_type=TimeControllerType::EMU_CONTROLLER;
        GlobalValue::Bind ("SimulatorImplementationType", StringValue ("ns3::RealtimeSimulatorImpl")); 
    }else{
        return -1;
    }
//随机丢包率设置，暂时没用到
    bool enable_random_loss=false;
    // std::cout<<"c++:loss rate:"<<loss_rate<<" do log:"<<do_log<<std::endl;

    if(loss_rate>0){
        Config::SetDefault ("ns3::RateErrorModel::ErrorRate", DoubleValue (loss_rate));
        Config::SetDefault ("ns3::RateErrorModel::ErrorUnit", StringValue ("ERROR_UNIT_PACKET"));
        Config::SetDefault ("ns3::BurstErrorModel::ErrorRate", DoubleValue (loss_rate));
        // Config::SetDefault ("ns3:: BurstErrorModel::BurstSize", StringValue ("ns3::UniformRandomVariable[Min=1|Max=3]"));
        enable_random_loss=true;
    }



    NodeContainer nodes = BuildExampleTopo(linkBw, msDelay, msQDelay,enable_random_loss);

    std::unique_ptr<TracePlayer> trace_player;
    // trace_path = "rl_script/traces/"+trace_path;
    std::cout<<"c++ main:trace_path:"<<getTraceName(trace_path)<<" port_num:"<<port_num<<std::endl;
    if (trace_path.empty() && duration_time_ms == 0) {
      duration_time_ms = 5000;
    } else if (!trace_path.empty()) {
      // Set trace
      trace_player = std::make_unique<TracePlayer>(trace_path, nodes);
      if (duration_time_ms == 0) {
        // duration_time_ms = 180000;
        duration_time_ms = trace_player->GetTotalDuration();
      }
    }
    // duration_time_ms = 180000;
    // std::cout<<"c++ main: connecting gym..."<<std::endl;
    GymConnector gym_conn(gym_id, report_interval_ms,0);
    if (congestion_control_algorithm=="gcc"||congestion_control_algorithm=="pcc") {
      // gym_conn.SetBandwidth(1e6,WLibraState::Ordinary);
      // std::cout<<"run GCC/PCC!"<<std::endl;
    } else {
      gym_conn.SetBandwidth(1e6,WLibraState::Ordinary);//给一个初始速率，不然初始速率是0，会把程序卡死
      
      gym_conn.Step();
    }
    
    auto gcc_interface = std::make_shared<webrtc::GoogCcNetworkControllerFactory>(
      webrtc::GoogCcNetworkControllerFactory(smoothing_coef));
    // gcc_interface->SetSmoothingCoef(0.9);
    auto pcc_interface = std::make_shared<webrtc::PccNetworkControllerFactory>(
    webrtc::PccNetworkControllerFactory());
    
    int64_t webrtc_start_us=appStartMills*kMicroPerMillis+1;
    int64_t webrtc_stop_us=appStopMills*kMicroPerMillis;
    webrtc::TimeController* time_controller=CreateTimeController(controller_type,webrtc_start_us,webrtc_stop_us);
    

    auto cc_factory = std::make_shared<NetworkControllerProxyFactory>(gym_conn,congestion_control_algorithm);
    auto se_factory = std::make_shared<NetworkStateEstimatorProxyFactory>(gym_conn);
    auto webrtc_manager = std::make_unique<WebrtcSessionManager>(time_controller,0, duration_time_ms, cc_factory, se_factory);
    if (congestion_control_algorithm=="gcc") {
      webrtc_manager->SetCcFactory(gcc_interface);
    }else if(congestion_control_algorithm=="pcc"){
      webrtc_manager->SetCcFactory(pcc_interface);
    }
    
    webrtc_manager->SetFrameHxW(video_height,video_width);
    webrtc_manager->CreateClients();

    uint16_t sendPort=5432;
    uint16_t recvPort=5000;
    // std::cout<<"matthew: creating app..."<<std::endl;
    auto sender = CreateApp<WebrtcSender>(nodes.Get(0), sendPort, 1, duration_time_ms, webrtc_manager.get());
    auto receiver = CreateApp<WebrtcReceiver>(nodes.Get(1), recvPort, 1, duration_time_ms, webrtc_manager.get());
    WebrtcTrace trace1;
    /*测出来的带宽和单项时延*/
    // std::cout<<"matthew: connecting app..."<<std::endl;
    // std::string log = start_time+"BW_RTT_Trace";
    std::string log = congestion_control_algorithm+"_"+getTraceName(trace_path)+"_"+"BW_RTT_Trace";
    std::cout<<"matthew: log:"<<log<<std::endl;
    
    if(do_log)
      trace1.Log(log,WebrtcTrace::E_WEBRTC_BW|WebrtcTrace::E_WEBRTC_RTT);
    ConnectApp(sender, receiver,&trace1);
    // std::cout<<"matthew: start simulation..."<<std::endl;
    Simulator::Stop (MilliSeconds(duration_time_ms + 1));
    Simulator::Run ();
    Simulator::Destroy();

    // if (standalone_test_only) {
    //   // for (auto &stat : gym_conn.ConsumeStats()) {
    //   //   // std::cout << stat << std::endl;
    //   // }
    // std::cout<<"Simulation ends."<<std::endl;
    // }
    return 0;
}
