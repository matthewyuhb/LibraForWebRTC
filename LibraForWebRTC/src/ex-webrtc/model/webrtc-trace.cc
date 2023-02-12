#include "webrtc-trace.h"
#include <unistd.h>
#include <memory.h>
#include <iostream>
namespace ns3{
WebrtcTrace::~WebrtcTrace(){
    Close();
}
void WebrtcTrace::Log(std::string &s,uint8_t flag){
    if(flag&E_WEBRTC_RTT){
        OpenTraceRttFile(s);
    }
    if(flag&E_WEBRTC_BW){
        OpenTraceBwFile(s);
    }
    flag_=flag;
}
void WebrtcTrace::OnRTT(uint32_t now,uint32_t rtt){
    char line [256];
    memset(line,0,256);
    if(m_rtt.is_open()){
        // std::cout<<"OnRTT"<<std::endl;

        float time=float(now)/1000;
        sprintf (line, "%f %16d",time,rtt);
        m_rtt<<line<<std::endl;
    }
}
void WebrtcTrace::OnBW(uint32_t now,uint32_t bps){
    char line [256];
    memset(line,0,256);
    if(m_bw.is_open()){
        // std::cout<<"OnBW"<<std::endl;

        float time=float(now)/1000;
        float kbps=float(bps)/1000;
        sprintf (line, "%f %16f",
                time,kbps);
        m_bw<<line<<std::endl;
    }    
}
void WebrtcTrace::OpenTraceRttFile(std::string &name){
    char buf[FILENAME_MAX];
    memset(buf,0,FILENAME_MAX);
    std::string path = std::string (getcwd(buf, FILENAME_MAX)) + "/rl_script/performance_records/runtime_data/"
            +name+"_rtt.txt";
    // printf(\n");
    // printf(path);
    std::cout<<"[C++]:OpenTraceRTT:"<<path<<std::endl;
    m_rtt.open(path.c_str(), std::fstream::out);
}
void WebrtcTrace::OpenTraceBwFile(std::string &name){
    char buf[FILENAME_MAX];
    memset(buf,0,FILENAME_MAX);
    std::string path = std::string (getcwd(buf, FILENAME_MAX)) + "/rl_script/performance_records/runtime_data/"
            +name+"_bw.txt";
    std::cout<<"[C++]:OpenTraceBwFile:"<<path<<std::endl;

    m_bw.open(path.c_str(), std::fstream::out);
}
void WebrtcTrace::CloseTraceRttFile(){
    if(m_rtt.is_open()){
        m_rtt.close();
    }
}
void WebrtcTrace::CloseTraceBwFile(){
    if(m_bw.is_open()){
        m_bw.close();
    }
}
void WebrtcTrace::Close(){
    CloseTraceRttFile();
    CloseTraceBwFile();
}
}
