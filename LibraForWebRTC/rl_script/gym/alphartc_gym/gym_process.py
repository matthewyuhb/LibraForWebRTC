#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import os
import signal

__ROOT_PATH__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
__GYM_PROCESS_PATH__ = os.path.join(__ROOT_PATH__, "../../build/scratch/webrtc_test", "webrtc_test")


class GymProcess(object):
    def __init__(
        self,
        gym_id: str = "gym",
        trace_path: str = "",
        report_interval_ms: int = 60,
        train_mode: str = "gcc",
        duration:int = 30000,
        loss_rate:float = 0.0,
        port_num : int = 0,
        do_log: bool = True,
        start_time:str = "",
        episode:int = 0):
        process_args = [__GYM_PROCESS_PATH__]
        # process_args.append(+str(1))
        print("matthew:GymProcess  path:"+__GYM_PROCESS_PATH__)

        if(gym_id!="gym"):
            process_args.append("--gym_id="+gym_id)
        if trace_path:
            process_args.append("--trace_path="+trace_path)
        if report_interval_ms:
            process_args.append("--report_interval_ms="+str(report_interval_ms))
        # if train_mode:
        process_args.append("--congestion_control_algorithm="+str(train_mode))
        if duration:
            process_args.append("--duration_time_ms="+str(duration))
        if port_num > 0:
            # print("[Py]port_num:",port_num)
            process_args.append("--port_num="+str(port_num))
        if(do_log):
            process_args.append("--log=true")
        else:
            process_args.append("--log=false")

        if start_time!="":
            process_args.append("--start_time="+str(start_time))

        if loss_rate>0.0:
            process_args.append("--loss="+str(loss_rate))
        if episode>0:
            process_args.append("--episode="+str(episode))

        self.gym = subprocess.Popen(process_args)

    # def wait(self, timeout = None):
    #     return self.gym.wait(timeout)

    # def __del__(self):
    #     self.gym.send_signal(signal.SIGINT)
    #     self.gym.send_signal(signal.SIGKILL)
