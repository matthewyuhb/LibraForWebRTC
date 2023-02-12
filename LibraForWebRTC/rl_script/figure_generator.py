#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 目前实际上除了画图之外还有测试作用
# from msilib.schema import _Validation_records
from pickle import NONE
from turtle import color
from matplotlib import style
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
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
'''
AVG_BW	AVG_RTT
447.6666667	106
180.3333333	135.3333333
122.6666667	213
	
	
AVG_BW	AVG_RTT
1169.5	135.14
500	226
177.5	115.45
	
	
AVG_BW	AVG_RTT
432	116.95
380	103.42
406	104
'''

def drawer():
    # datas = {
    #     "4G":[447.7,180.3,122.7],
    #     "5G":[1169.5,500,177.5],
    #     "WiFi":[432,380,406]
    # }
    datas = {
        "4G":[106,135.3,213],
        "5G":[135.1,226,115.5],
        "WiFi":[116.9,103.4,104]
    }
    CC = ["Libra","GCC","PCC"]
    colors = ["firebrick","royalblue","forestgreen"]

    plt.figure(figsize=(10,4))
    width = 1
    bin_width = width/5
    data_pd = pd.DataFrame(datas)
    ind = np.arange(0,len(datas))

    # 第二种方案
    for index in data_pd.index:
        day_tickets = data_pd.iloc[index]
        xs = ind-(bin_width*(2-index))
        plt.bar(xs,day_tickets,width=bin_width,edgecolor ='black',label=CC[index],color = colors[index])
        for ticket,x in zip(day_tickets,xs):
            plt.annotate(ticket,xy=(x,ticket),xytext=(x-0.08,ticket+0.2),size=12)
    # 设置图例
    plt.legend()
    plt.ylabel("Avg Delay/ms",size=12)
    # plt.title("RECORD")
    # 设置x轴的坐标
    # print("ind:"+str(ind))
    plt.xticks(ind-0.2,data_pd.columns,size=12)
    plt.yticks(size=12)
    plt.xlim
    plt.grid(linestyle='--')

    # plt.savefig("thpt.pdf")
    plt.savefig("delay.pdf")
    
    
'''
        FPS     Stall_Rate    Bitrate   QP      Vmaf
Libra:  54.47    0.1259        1.21    54.20    61.97015560000001
GCC:    54.25    0.1262        1.15    57.2     60.672491000000015
PCC:    54.58    0.1272        1.05    54.27    57.5503286
'''

'''
loss:
        FPS     Stall_Rate    Bitrate   QP      Vmaf
Libra:  44.048    26.49        1.37    53.55   47.912513299999986
GCC:    28.00     26.81        0.41    83.42   20.9728054
PCC:    17.4083   37.28        0.15    94.125  13.668482799999998

'''

'''
high jitter:
        FPS     Stall_Rate    Bitrate   QP      Vmaf
Libra:  48.730    0.63        1.04    54.8   55.0728652
GCC:    42.12     4.84        1.07    52.0   50.024041999999994
PCC:    45.95     4.62        0.75    56.8   41.6286736
'''


'''
Real path:
        FPS     Stall_Rate    Bitrate   QP      Vmaf
Libra:  56.04    0.002        1.12    53.8   55.0728652
GCC:    56.8     0.002        0.869   57.2   50.024041999999994
PCC:    26.10    0.292        0.3     81.2   41.6286736
'''
def draw_QoE_bar():

    movies = {
        "WLibra":48.73,
        "GCC":42.12,
        "PCC":45.95
    }

    plt.figure(figsize=(6,3))

    x = list(movies.keys()) # 获取x轴数据(字典的键)
    y = list(movies.values()) # 获取y轴数据(字典的值)
    

    plt.bar(x,y,width=0.3,color='firebrick',edgecolor ='black',linewidth=2)
    plt.ylim(40,50)
    plt.grid(linestyle = '--')
    
    font_size = 16
    
    for i in range(len(x)):
         plt.text(x[i],y[i],y[i],ha='center',size = font_size)

    # 绘制标题
    # plt.title("m",size=26)

    # 设置轴标签
    # plt.xlabel("CC",size=16)
    plt.xticks(size=font_size)
    plt.yticks(size=font_size)
    plt.ylabel("FPS",size=font_size)

    plt.savefig("bar_vmaf_loss.pdf")

def data_processor():
    print("data_processor")
    py_file_name = "rl_script/performance_records/raw_data.txt"
    fps = []
    stall_rate = []
    avg_sendrate = []
    qp = []
    # frame_decoded = []
    for line in open(py_file_name,'r').readlines():
        print("line:",line)
        strs_list = line.split(':')
        if strs_list[0] == "avg fps":
            fps.append(float(strs_list[1]))
        elif strs_list[0] == "stall rate":
            stall_rate.append(float(strs_list[1][:-2]))
        elif strs_list[0] == "avg_sendrate":
            avg_sendrate.append(float(strs_list[1]))
        elif strs_list[0] == "QP":
            qp.append(float(strs_list[1]))
        # elif strs_list[0] == "stall rate":
        #     stall_rate.append(float(strs_list[1]))
            
        # line = line[1:]
        # # print("line:",line)
        # line = line[:-2]
        # # print("line:",line)
        
        # list_owd_time = Derive_Value(line)
        # owd.append(list_owd_time[0])
        # owd_time.append(list_owd_time[1])
    print(np.mean(fps))
    print(np.mean(stall_rate))
    print(np.mean(avg_sendrate))
    print(np.mean(qp))

def get_correlated_dataset(n, dependency, mu, scale):
    latent = np.random.randn(n, 2)
    dependent = latent.dot(dependency)
    scaled = dependent * scale
    scaled_with_offset = scaled + mu
    # return x and y of the new, correlated dataset
    return scaled_with_offset[:, 0], scaled_with_offset[:, 1]

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
    
def data_processor_ns3(py_file_name):
    print("data_processor_ns3")
    # py_file_name = "rl_script/performance_records/ns_data.txt"
    bw = []
    delay = []
    # frame_decoded = []
    for line in open(py_file_name,'r').readlines():
        # print("line:",line)
        strs_list = line.split(':')
        if strs_list[0] == "avgbw":
            bw.append(float(strs_list[1]))
        elif strs_list[0] == "avgrtt":
            delay.append(float(strs_list[1]))
    return np.array(bw),np.array(delay)


def draw_ellipse():
    fig, ax_kwargs = plt.subplots(figsize=(6.5, 6.5))
    dependency_kwargs = [[-0.8, 0.5],
                        [-0.2, 0.5]]
    mu = 2, -3
    scale = 6, 5

    ax_kwargs.axvline(c='grey', lw=1)
    ax_kwargs.axhline(c='grey', lw=1)

    # x, y = get_correlated_dataset(500, dependency_kwargs, mu, scale)
    # print("x:"+str(type(x)))
    # print("y:"+str(y))
    # Plot the ellipse with zorder=0 in order to demonstrate
    # its transparency (caused by the use of alpha).
    libra_bw,libra_delay = data_processor_ns3("rl_script/performance_records/ns_data_libra.txt")
    gcc_bw,gcc_delay = data_processor_ns3("rl_script/performance_records/ns_data_gcc.txt")
    pcc_bw,pcc_delay = data_processor_ns3("rl_script/performance_records/ns_data_pcc.txt")
    
    confidence_ellipse(pcc_bw, pcc_delay, ax_kwargs,
                    alpha=0.8, facecolor='yellow',zorder=0,label = "PCC")
    
    confidence_ellipse(gcc_bw, gcc_delay, ax_kwargs,
                    alpha=0.8, facecolor='orange',zorder=0,label = "GCC")
    
    confidence_ellipse(libra_bw, libra_delay, ax_kwargs,
                    alpha=0.8, facecolor='lightgreen',zorder=0,label = "LIBRA")

    ax_kwargs.scatter([np.mean(pcc_bw)], [np.mean(pcc_delay)],color = 'yellow', edgecolor='black', s=100,marker="o",label = "PCC")
    ax_kwargs.scatter([np.mean(gcc_bw)], [np.mean(gcc_delay)],color = 'orange', edgecolor='black', s=100,marker="^",label = "GCC")
    ax_kwargs.scatter([np.mean(libra_bw)],np.mean(libra_delay) ,color = 'lightgreen',edgecolor='black', s=100, marker = "*",label = "LIBRA")
    
    # ax_kwargs.scatter(mu[0], mu[1], c='red', s=3)
    ax_kwargs.legend()
    ax_kwargs.grid(linestyle='--')
    plt.xlabel("Throughput/Kbps",size=16)
    plt.ylabel("Delay/ms",size=16)
    plt.xticks(size=16)
    plt.yticks(size=16)
    
    
    fig.subplots_adjust(hspace=0.25)
    plt.savefig("ellipse.pdf")
        
    # print(np.mean(fps))
    # print(np.mean(stall_rate))
    # print(np.mean(avg_sendrate))
    # print(np.mean(qp))

def reward_drawer():
    reward_record = []
    ewma_reward_record = []
    plt.figure(figsize=(8,5))
    # f=open(,"w")
    
    alpha_n = 0.15
    r = 0.0
    ewma_r = 0.0
    smoothing_coef = 0.99
    
    plt.tick_params(labelsize=15)
    plt.grid(linestyle='--')
    
    for line in open("/home/admin/LibraForWebRTC/rl_script/train_dir/reward-0721_164644.txt",'r').readlines():
        r = float(line)
        ewma_r = smoothing_coef*ewma_r + (1.0-smoothing_coef)*r
        reward_record.append(r)
        ewma_reward_record.append(ewma_r)
    reward_record = reward_record[:6000]
    ewma_reward_record = ewma_reward_record[:6000]
    plt.plot(range(len(reward_record)), reward_record,alpha = alpha_n, linewidth = 5,color = 'firebrick')
    plt.plot(range(len(ewma_reward_record)), ewma_reward_record, linewidth = 3,color = 'firebrick')
    
    reward_record = []
    ewma_reward_record = []
    r = 0.0
    ewma_r = 0.0
    smoothing_coef = 0.99
    for line in open("/home/admin/LibraForWebRTC/rl_script/train_dir/reward-0819_155703.txt",'r').readlines():
        r = float(line)
        ewma_r = smoothing_coef*ewma_r + (1.0-smoothing_coef)*r
        reward_record.append(r)
        ewma_reward_record.append(ewma_r)
    reward_record = reward_record[:6000]
    ewma_reward_record = ewma_reward_record[:6000]
    plt.plot(range(len(reward_record)), reward_record,alpha = alpha_n,linewidth = 5,color = 'orange')
    plt.plot(range(len(ewma_reward_record)), ewma_reward_record, linewidth = 3,color = 'orange')
    
    reward_record = []
    ewma_reward_record = []
    r = 0.0
    ewma_r = 0.0
    smoothing_coef = 0.99
    for line in open("/home/admin/LibraForWebRTC/rl_script/train_dir/reward-0719_114919.txt",'r').readlines():
        r = float(line)
        ewma_r = smoothing_coef*ewma_r + (1.0-smoothing_coef)*r
        reward_record.append(r)
        ewma_reward_record.append(ewma_r)
    reward_record = reward_record[:6000]
    ewma_reward_record = ewma_reward_record[:6000]
    plt.plot(range(len(reward_record)), reward_record,alpha = alpha_n,linewidth = 5,color = 'royalblue')
    plt.plot(range(len(ewma_reward_record)), ewma_reward_record, linewidth = 3,color = 'royalblue')
    
    plt.xlabel('Episode',fontsize = 15)
    plt.ylabel('Averaged episode reward',fontsize = 15)
    plt.savefig("reward.pdf")

def data_processor_vmaf():
    # with open() as f:
    count = 0
    vmaf = 0
    for line in open("rl_script/performance_records/vmaf.txt" ,'r').readlines():
        # print("line:"+str(line)+" len:"+str(len(line)))
        if(len(line)<5):
            continue
        count+=1
        st = 0
        ed = len(line)-1
        while(1):
            if line[st]>='0' and line[st]<='9':
                break
            st+=1
            
        while(1):
            if line[st]>='0' and line[st]<='9':
                break
            ed-=1
        vmaf += float(line[st:ed-1])
        if count%10==0:
            
            print("avg vmaf:"+str(vmaf/10))
            vmaf = 0



if __name__ == '__main__':
    # plt.plot.box()绘制
    # data_processor_vmaf()
    # draw_QoE_bar()
    data_processor()
    # reward_drawer()
    # drawer()
    # draw_ellipse()
    

