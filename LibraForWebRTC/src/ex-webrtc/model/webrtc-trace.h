#pragma once
#include <iostream>
#include <fstream>
#include <string>
namespace ns3{
class WebrtcTrace{
public:
enum WebrtcTraceEnable:uint8_t{
    E_WEBRTC_RTT=0x01,
    E_WEBRTC_BW=0x02,
    E_WEBRTC_ALL=E_WEBRTC_RTT|E_WEBRTC_BW
};
    WebrtcTrace(){};
    ~WebrtcTrace();
    void Log(std::string &s,uint8_t flag);
    void OnBW(uint32_t now,uint32_t bps);
    void OnRTT(uint32_t now,uint32_t rtt);
    uint8_t LogFlag() const {return flag_;}
private:
    void Close();
    void OpenTraceRttFile(std::string &name);
    void OpenTraceBwFile(std::string &name);
    void CloseTraceRttFile();
    void CloseTraceBwFile();
    uint8_t flag_=0;
    std::fstream m_rtt;
    std::fstream m_bw;
};
}
