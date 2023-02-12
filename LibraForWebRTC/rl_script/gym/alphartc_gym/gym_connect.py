#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import zmq
import json
import time
import sys

__ZMQ_TYPE__ = "ipc://"
__ZMQ_PATH__ = "/tmp/"
__ZMQ_PREFIX__ = __ZMQ_TYPE__ + __ZMQ_PATH__
__GYM_EXIT_FLAG__ = b"Bye"

class GymConnector(object):
    def __init__(self, gym_id = "gym",train_mode: str = 'gcc'):
        self.gym_id = gym_id
        self.zmq_ctx = zmq.Context()
        self.zmq_sock = self.zmq_ctx.socket(zmq.REQ)
        self.gym_name = __ZMQ_PREFIX__ + self.gym_id
        self.zmq_sock.connect(self.gym_name)
        self.train_mode = train_mode
        # print("binding port..{}".format(self.port))
        # try:
        #     self.zmq_sock.bind ("tcp://*:%s" % str(int(self.port)))

        # except Exception as e:
        #     print("Cannot bind to tcp://*:%s as port is already in use" % str(int(self.port)) )
        #     print("Please specify different port or use 0 to get free port" )
        #     sys.exit()
        

    # def step(self, bandwidth_bps = int):
    def step(self, msg):
        self.zmq_sock.send_string(msg)
        
        # if(self.train_mode!='gcc' and self.train_mode!='pcc'):
        
        rep = self.zmq_sock.recv()
        
        if rep == __GYM_EXIT_FLAG__:
            # print("rep:"+str(rep))
            return None
        return json.loads(rep)

    def __del__(self):
        # print("Del")

        # print(self.gym_name)
        # self.zmq_sock.disconnect(self.gym_name)
        # print("deleted");
        pass
