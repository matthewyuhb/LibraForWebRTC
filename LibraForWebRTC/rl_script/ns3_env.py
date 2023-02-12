import zmq
import os
import subprocess
import json
from enum import Enum

_ZMQ_PATH = "ipc:///tmp/loki_gcc_env"
_GYM_EXIT_FLAG = "Bye"
_GYM_PROCESS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../ns3/build/scratch/webrtc_test",
    "webrtc_test")


class CCType(Enum):
    RL_BASED = "0"
    ORCA = "1"
    PCC = "2"
    WLIBRA = "3"
    GCC = "4"


class NS3Env:
    def __init__(self) -> None:
        # 初始化用到的常量参数
        self.stdout_logger = open("stdout.log", "w")
        self.stderr_logger = open("stderr.log", "w")
        # 创建ZMQ连接
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(_ZMQ_PATH)
        self.gym = None

    def reset(self, trace_path: str, cc_type: CCType, duration: int) -> None:
        '''
        根据参数启动一个环境
        trace_path: 用于训练的trace文件路径
        cc_type: 用于训练的CC类型
        duration: 训练的时长，单位为ms
        '''
        if self.gym is not None:
            self.gym.kill()
            self.stdout_logger.writelines('-'*25+' Episode End '+'-'*25+'\n')
            self.stderr_logger.writelines('-'*25+' Episode End '+'-'*25+'\n')
        # 设置启动环境的参数
        self.process_args = [
            _GYM_PROCESS_PATH,
            f"--congestion_control_algorithm={cc_type.value}",
            f"--duration_time_ms={duration}",
            f"--trace_path={trace_path}",
            f"--report_interval_ms={0}",
            f"--smoothing_coef={0.8}",
        ]
        # 启动NS3环境
        self.gym = subprocess.Popen(
            self.process_args, stdout=self.stdout_logger, stderr=self.stderr_logger)

    def send(self, msg: str):
        self.socket.send_string(msg)

    def recv(self):
        rep = self.socket.recv_string()
        if _GYM_EXIT_FLAG == rep:
            return None
        return json.loads(rep)

    def __del__(self):
        # 关闭环境
        # self.socket.send_string(_GYM_EXIT_FLAG)
        self.socket.close()
        self.context.term()
        self.stdout_logger.close()
        self.stderr_logger.close()
