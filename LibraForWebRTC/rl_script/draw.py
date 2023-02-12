#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 目前实际上除了画图之外还有测试作用
# from msilib.schema import _Validation_records
from pickle import NONE
import torch
import time
import csv
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
import pandas as pd  # 保存数据用
from pandas import DataFrame as df
from rtc_env import GymEnv
from deep_rl.storage import Storage
from deep_rl.actor_critic import ActorCritic
import os
import json
from optparse import OptionParser

UNIT_M = 1000000
MAX_BANDWIDTH_MBPS = 8
MIN_BANDWIDTH_MBPS = 0.01
LOG_MAX_BANDWIDTH_MBPS = np.log(MAX_BANDWIDTH_MBPS)
LOG_MIN_BANDWIDTH_MBPS = np.log(MIN_BANDWIDTH_MBPS)


def log_to_linear(value):
    # from 0~1 to 10kbps to 8Mbps
    value = np.clip(value, 0, 1)
    log_bwe = value * (LOG_MAX_BANDWIDTH_MBPS -
                       LOG_MIN_BANDWIDTH_MBPS) + LOG_MIN_BANDWIDTH_MBPS
    return np.exp(log_bwe) * UNIT_M


def draw_state(record_action, record_state, path, test_path="normal"):
    # print("matthew:drawing state")
    
    fig = plt.figure()
    length = len(record_action)
    plt.subplot(411)
    plt.plot(range(length), record_action)
    plt.xlabel('episode')
    plt.ylabel('action')
    ylabel = ['receiving rate', 'delay', 'packet loss']
    #record_state = [t.numpy() for t in record_state]
    record_state = [t for t in record_state]
    #print(record_state)
    record_state = np.array(record_state)
    for i in range(3):
        plt.subplot(411 + i + 1)
        plt.plot(range(length), record_state[:, i])
        plt.xlabel('episode')
        plt.ylabel(ylabel[i])
    plt.tight_layout()
    plt.savefig("{}test_result_{}.jpg".format(path, test_path))
    plt.close('all')





def draw_delay(record_delay, path):
    fig = plt.figure()
    length = len(record_delay)
    print("record_delay:", record_delay)
    plt.plot(range(length), record_delay)
    plt.xlabel('episode')
    plt.ylabel('delay')
    plt.savefig("{}test_delay.jpg".format(path))
    plt.close('all')


def metric(reports, data_path, test_path="normal", threshold=200):
    #reports=[]# 接受速率，延迟，丢包 见rtc_env.py
    receiving_rate = []
    delay = []
    loss_ratio = []
    real_test_path = ""
    max_capacity1 = -1
    temp_capacity = []
    if test_path != "normal":
        real_test_path = "./test/" + test_path
        with open(real_test_path + '.json', 'r') as f:
            data = json.load(f)
            data_uplink = data["uplink"]
            data_trace_pattern = data_uplink["trace_pattern"]

            for ele in data_trace_pattern:
                # ele_dic={"duration": 60000,
                #     "capacity": 300, 表示300kbps;训练时候出来的数据单位是bps因此要进行转换
                #     "loss": 0,
                #     "jitter": 0,
                #     "time": 0.0}
                temp_capacity.append(ele["capacity"] * 1000)
                temp1_capacity = ele["capacity"] * 1000
                max_capacity1 = max(max_capacity1, temp1_capacity)
    # if max_capacity==-1:
    #     print("没有该测试文件，无法获得ground truth")
    #     assert 0

    for ele in reports:
        ele_receiving_rate = ele[0]
        ele_delay = ele[1]
        ele_loss_ratio = ele[2]
        receiving_rate.append(ele_receiving_rate)
        delay.append(ele_delay)
        loss_ratio.append(ele_loss_ratio)
    temp_capacity.sort()
    max_capacity = np.max(temp_capacity[0:len(temp_capacity) - 5])
    print("去除最大的五个的max 原来的max", max_capacity, max_capacity1)
    # delay的评价
    min_delay = min(delay)
    # base_delay = 0 if min_delay >= 0 else -min_delay
    delay_pencentile_95 = np.percentile(delay, 95)
    # mean_delay=np.mean(delay)+base_delay
    max_delay = 800  #手动设置
    delay_score = [
        (max_delay - delay_pencentile_95) / (max_delay - min_delay / 2)
        for ele_delay in delay
    ]
    avg_delay_score = max(np.mean(delay_score), 0)
    # loss的评价
    loss_list = loss_ratio
    avg_loss_rate = sum(loss_list) / len(loss_ratio)
    # receiving rate的评价
    avg_recv_rate_score = min(1, np.mean(receiving_rate) / max_capacity)

    #network_score = 50 * (1 - avg_delay_score) + 50 * (1 - avg_loss_rate)
    network_score = 100 * 0.2 * avg_delay_score + 100 * 0.2 * avg_recv_rate_score + 100 * 0.3 * (
        1 - avg_loss_rate)
    # receiving rate
    avg_receiving_rate = np.mean(receiving_rate)
    print(
        "网络得分:{},平均延迟:{},平均丢包率:{},平均接受速率:{}; network_score=100 * 0.2 * avg_delay_score +  100 * 0.2 * avg_recv_rate_score +100 * 0.3 * (1 - avg_loss_rate)"
        .format(network_score, avg_delay_score, avg_loss_rate,
                avg_receiving_rate))
    f = open(
        "{}record_metric_{}_{}.txt".format(
            data_path, test_path,
            time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())), 'w')
    f.write(
        "网络得分:{},平均延迟:{},平均丢包率:{},平均接受速率:{}; network_score=100 * 0.2 * avg_delay_score +  100 * 0.2 * avg_recv_rate_score +100 * 0.3 * (1 - avg_loss_rate)"
        .format(network_score, avg_delay_score, avg_loss_rate,
                avg_receiving_rate))
    f.close()


def draw_module(model, data_path, max_num_steps=10000):
    env = GymEnv()
    ## 获取测试集合
    test_set = env.get_test()
    print("test_set:"+str(test_set))
    for test in test_set:
        print(test)
        if env.safety == 1:
            print("强化学习决策")
        else:
            print("GCC决策")
        temp = os.path.split(test)
        test_name = os.path.splitext(temp[1])[0]

        record_reward = []
        record_state = []

        record_delay = []
        record_report = []
        record_action = []
        episode_reward = 0
        time_step = 0
        tmp = model.random_action
        model.random_action = False

        done = False
        state = torch.Tensor(env.set_test(test))
        print(state)
        while not done and time_step < max_num_steps:
            action, _, _ = model.forward(state)
            state, reward, done, _, report, action ,Time= env.draw_step(
                action)  # 如果unsafety为0，表示启用safety,也就是切换；
            state = torch.Tensor(state)
            record_state.append(state)
            # record_state.append(Time)
            record_reward.append(reward)
            record_action.append(action)
            record_delay.append(report[1])
            record_report.append(report)
            time_step += 1
        ##########################
        # 度量本次结果
        metric(reports=record_report,
               data_path=data_path,
               test_path=test_name,
               threshold=200)
        ##########################
        model.random_action = True
        draw_state(record_action,
                   record_report,
                   data_path,
                   test_path=test_name)
        print(test_name)
        output_report = np.array(record_report)
        output_report = pd.DataFrame(output_report)
        output_report.to_csv("{}record_state_{}.csv".format(
            data_path, test_name))
    # while time_step < max_num_steps:
    #     done = False
    #     state = torch.Tensor(env.reset())
    #     while not done and time_step < max_num_steps: #  and time_step < max_num_steps: 删去可以只画1000ms的
    #         action, _, _ = model.forward(state)
    #         state, reward, done, _,report,action = env.draw_step(action)
    #         state = torch.Tensor(state)
    #         record_state.append(state)
    #         record_reward.append(reward)
    #         record_action.append(action)
    #         record_delay.append(report[1])
    #         record_report.append(report)
    #         time_step += 1

    # model.random_action = True

    #draw_delay(record_delay,data_path)
    # output_report=np.array(record_report)
    # output_report=pd.DataFrame(output_report)
    # output_report.to_csv("{}record_state.csv".format(data_path))
        
def StrToFloat(s):
    try:
        return float(s)
    except:
        return 0

def draw_metric(path, metric_type, log_dir_base, episode_num = 0,trace_path=NONE,alg = 'gcc'):
    # fig = plt.figure()
    print("alg:{},trace_path:{}".format(alg,trace_path))
    record = []
    valid = [False, False]
    max_n=10.0
    max_timestamp = 0
    timestamp = 0
    data_path = path+"/rl_script/performance_records/runtime_data/"+str(alg)+"_"+str(trace_path.split('/')[-1].split('.')[0])+"_BW_RTT_Trace_"+metric_type+".txt"
    save_path = os.path.join(log_dir_base, str(episode_num)+str(metric_type)+'.jpg')
    print("save pash:{}".format(save_path))
    # save_path = path+"/rl_script/traces/"+metric_type+str(episode_num)+".jpg"
    with open(data_path, 'rt') as txtfile:
        spamreader = csv.reader((line.replace('\0','') for line in txtfile), delimiter='\t')
        next(spamreader)
        for row in spamreader:
            # print("row[0]" + str(row[0]))
            # print(row[0])
            if(len(row)==0):
                continue
            row_list=row[0].split()
            if(len(row_list)!=2):
                continue
            # print("row:"+str(row_list))
            # subflow=row_list[0]
            timestamp=StrToFloat(row_list[0])
            if(timestamp==0):
                continue
            sr=row_list[1]
            f_sr=StrToFloat(sr)
            if(f_sr>max_n):
                max_n=f_sr
            max_n=max(max_n,f_sr)
            max_timestamp = max(max_timestamp,timestamp)
            
            record.append([timestamp, f_sr])
    
    columns = ['timestamp',metric_type]

    # print(record)
    p_records = pd.DataFrame(record, columns=columns)
    v_records = p_records[["timestamp", metric_type]].values
    bw_records = []
    capacity = 0
    x_time = []
    y_bottomline = []
    if(metric_type=='bw'):#plot bw of the trace
        # loaded_json = json.load(open(path+"/"+trace_path,'r',encoding="utf-8"))
        loaded_json = json.load(open(trace_path,'r',encoding="utf-8"))
        load_dict = loaded_json['uplink']['trace_pattern']
        timestamp = 0

        for trace in load_dict:
            capacity =  float(trace['capacity'])
            bw_records.append(capacity)
            max_n=max(max_n,capacity)
            x_time.append(timestamp)
            y_bottomline.append(0)
            timestamp+=float(trace['duration'])/1000
        # print("time:"+str(timestamp)+" max:"+str(max_timestamp))
        # if(timestamp<max_timestamp):
            
        bw_records.append(capacity)
        x_time.append(max_timestamp)
        y_bottomline.append(0)

        # print(bw_records)
    # p_bw_records = pd.DataFrame(bw_records, columns=columns)
    # v_bw_records = p_bw_records[["timestamp", metric_type]].values


    # plt.clf()
    plt_metric,=plt.plot(list(v_records[:,0]), list(v_records[:,1]), 'b-',linewidth=1) 
    plt.legend([plt_metric], ['sending rate' if metric_type == 'bw' else 'rtt'], loc='upper center',ncol=2)
    print("avg"+metric_type+":"+str(np.mean(list(v_records[:,1]))))
    # plt_sendingrate,=plt.plot(list(v_bw_records[:,0]), list(v_bw_records[:,1]), 'black',linewidth=1) 
    
    ax=plt.gca()
    
    if(max_timestamp>10):
        plt.xlim(0,max_timestamp)
        x_major_locator=MultipleLocator(max_timestamp/10)
        ax.xaxis.set_major_locator(x_major_locator)

    if(max_n>10):
        plt.ylim(0,max_n*1.1)
        y_major_locator=MultipleLocator(max_n/10)
        ax.yaxis.set_major_locator(y_major_locator)

    # plt.plot(record)
    ax.plot(x_time,bw_records,color='black',linewidth=0.5)
    ax.plot(x_time,y_bottomline,color='black',linewidth=0.5)
    ax.fill_between(x_time,bw_records,y_bottomline,color='teal',alpha=0.1,interpolate=True)
    plt.savefig(save_path)
    

    

if __name__ == '__main__':
    parser = OptionParser()

    parser.add_option("-d", "--trace_path", dest="TracePath", default="rl_script/traces/trace_300k.json",
                      help="The trace path")
    parser.add_option("-i", "--episode", dest="Episode", default="0",
                      help="The episode")
    parser.add_option("-a", "--alg", dest="Algorithm", default="4",
                      help="The CC alg")
    (options, args) = parser.parse_args()
    # duration=int(options.Duration)
    trace_path = str(options.TracePath)
    episode = int(options.Episode)
    alg = str(options.Algorithm)
    print("trace_path:"+str(trace_path))
    print("episode:"+str(episode))
    print("The CC alg:"+str(alg))
    pr_base = os.path.join(os.path.abspath('.'), 'rl_script/performance_records/test')
    # 
    plt.figure(figsize=(6,9))
    plt.subplot(211)
    draw_metric(os.path.abspath('.'),'bw',pr_base,0,trace_path,alg = alg)
    plt.subplot(212)
    draw_metric(os.path.abspath('.'),'rtt',pr_base,0,alg = alg)
    save_path = os.path.join(pr_base, str(episode)+'_merge.jpg')
    plt.savefig(save_path)
    plt.close()
    # print()/home/yhb/LibraForWebRTC/LibraForWebRTC/rl_script/traces/1203_173537BW_RTT_Trace_rtt.txt