#!/bin/bash
function runSet() {
    echo "runSet"
    clear
    sudo killall -9 webrtc_test
    ./waf build
    # python3  rl_script/orca.py -d "${PyConfig["test_or_train"]}" -m "${PyConfig["report_interval_ms"]} " -t "${PyConfig["train_mode"]} "
    # python3  rl_script/libra_pygcc.py -d "${PyConfig["test_or_train"]}" -m "${PyConfig["report_interval_ms"]} " -t "gcc"
    # python3  rl_script/libra_pygcc.py -d "${PyConfig["test_or_train"]}" -m "${PyConfig["report_interval_ms"]} " -t "pcc"
    # python3  rl_script/libra_pygcc.py -d "${PyConfig["test_or_train"]}" -m "${PyConfig["report_interval_ms"]} " -t "libra"
    python3  rl_script/libra_pygcc.py -d "${PyConfig["test_or_train"]}" -m "${PyConfig["report_interval_ms"]} " -t "onrl"
    # python3  rl_script/libra_pygcc.py -d "${PyConfig["test_or_train"]}" -m "${PyConfig["report_interval_ms"]} " -t "loki"

    
    
}

#####################congestion_control_algorithm
unset PyConfig; unset Ns3Config; declare -A PyConfig; declare -A Ns3Config
# congestion_control_algorithm=1 #experiment="Exp11"
trace_file="rl_script/traces/fixed_cap/track_step3.json" #trace_300k.json  _WIRED_200kbps.json
PWD=$(shell pwd)
trace_path=$PWD$trace_file
congestion_control_algorithm="pcc" #0:RL-based 1:Orca 2:PCC 3:libra 4:GCC 5:OnRL 6:Loki
report_interval_ms="0" #if the report_interval_ms is set to 0, the monitor interval is one RTT
smoothing_coef="0.1"
test_or_train="0" #0: test 1:train

# PyConfig+=(["trace_path"]=$trace_path)
PyConfig+=(["trace_path"]=$trace_path)
PyConfig+=(["test_or_train"]=$test_or_train ["report_interval_ms"]="$report_interval_ms" ["train_mode"]="$congestion_control_algorithm")
Ns3Config+=(["trace_path"]=$trace_path ["congestion_control_algorithm"]="$congestion_control_algorithm" ["smoothing_coef"]="$smoothing_coef" ["report_interval_ms"]="$report_interval_ms")

runSet

#!/bin/sh 
#============ get the file name =========== 
    # Folder_A="/home/yhb/LibraForWebRTC/LibraForWebRTC/rl_script/traces/varied_cap" 
    # COUNTER=0
    # for file_a in ${Folder_A}/*
    # do
    # temp_file=`basename $file_a` 
    # echo ----- Path ---- 
    # echo rl_script/traces/varied_cap/$temp_file
    # echo ----- Path ----

    # # echo ${Ns3Config["trace_path"]}
    # ./waf --run "webrtc_test --episode="$COUNTER" --congestion_control_algorithm="${Ns3Config["congestion_control_algorithm"]}" --trace_path="rl_script/traces/varied_cap/$temp_file" --smoothing_coef="${Ns3Config["smoothing_coef"]}" --report_interval_ms="${Ns3Config["report_interval_ms"]}""
    
    # python3  rl_script/draw.py -d "rl_script/traces/varied_cap/$temp_file" -i $COUNTER -a "${PyConfig["train_mode"]}"
    # COUNTER=$[$COUNTER+1]
    # # echo $COUNTER
    # done