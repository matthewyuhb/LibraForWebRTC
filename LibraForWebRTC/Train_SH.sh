#!/bin/bash
function runSet() {
    clear
    sudo killall -9 webrtc_test
    ./waf build
    python3  rl_script/spql_train.py -d "${PyConfig["test_or_train"]}" -m "${PyConfig["report_interval_ms"]} " -t "${PyConfig["train_mode"]} "
    # python3  rl_script/orca.py -d "${PyConfig["test_or_train"]}" -m "${PyConfig["report_interval_ms"]} " -t "${PyConfig["train_mode"]} "
}

#####################congestion_control_algorithm
unset PyConfig; declare -A PyConfig;
report_interval_ms="0" #if the report_interval_ms is set to 0, the monitor interval is one RTT
test_or_train="1" #0: test 1:train
train_mode="onrl" #"1:enable gcc like orca 0:clean-slate"
PyConfig+=(["test_or_train"]=$test_or_train ["report_interval_ms"]="$report_interval_ms" ["train_mode"]="$train_mode")
runSet