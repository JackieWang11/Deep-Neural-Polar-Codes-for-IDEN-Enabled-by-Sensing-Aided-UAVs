import numpy as np
import tensorflow as tf
import math

#
SNR_db=20  # dB
SNR=10 ** (SNR_db / 10)
N_dbm=-10       # 噪声功率
S_dbm=SNR_db+N_dbm
P_S=10 ** (S_dbm / 10)
P_N=10 ** (N_dbm / 10)
noise_std = np.sqrt(P_N/2)



#
tx_N=2       # 发射端天线数
rx_N=1       # 接收端天线数
tx_gain=1.   # 发射天线增益  0dB
rx_gain=1.   # 接收天线增益  0dB
#
timeslot_num=4               # 通信时隙数
tao=1                   # 1s
bandw= 1*10**8               # 带宽 100MHz
freq=4*10**9                 # 4GHz
wavelen=3*10**8/freq         # 波长
D_array= tx_N*(wavelen/2)    # 阵列孔径
theta_3dB= 0.886*(wavelen/D_array)            # 3dB带宽
###################### 感知误差 #############################
err_dis=3*10**8/(2*bandw*np.sqrt(2*SNR))      # 距离误差
err_theta=theta_3dB/(1.6*np.sqrt(2*SNR))      # 角度误差
###################### 位置信息 ##############################
isaway=1                                     # 靠近 0;远离 1
v= 0.                                        # BS速度  m/s

if isaway==1:
    ########## 目标位置信息
    x_tar_min = 0.
    x_tar_max = 5.
    y_tar_min = 0.
    y_tar_max = 5.
    h_tar = 0.
    ########## BS初始位置信息
    x_bs_min = 0#53.#0#
    x_bs_max = 5#58.#5#
    y_bs_min = 0.
    y_bs_max = 5.
    h_bs = 15.
else:
    ########## 目标位置信息
    x_tar_min = 0.
    x_tar_max = 5.
    y_tar_min = 0.
    y_tar_max = 5.
    h_tar = 0.
    ########## BS初始位置信息
    x_bs_min = 60.
    x_bs_max = 65.
    y_bs_min = 0.
    y_bs_max = 5.
    h_bs = 15.

###################### LOS/NLOS 系数 ##############################
K_R=40#17#10

###############################################
def generate_local(batch):
    # x_tar=np.reshape(np.random.randint(x_tar_min, x_tar_max, batch),[batch,1]) # batchsize
    # y_tar=np.reshape(np.random.randint(y_tar_min, y_tar_max, batch),[batch,1])
    # x_bs = np.reshape(np.random.randint(x_bs_min, x_bs_max, batch),[batch,1])
    # y_bs=np.reshape(np.random.randint(y_bs_min, y_bs_max, batch),[batch,1])
    # 每次所有 batch 都一样
    x_tar = np.tile(np.random.randint(x_tar_min, x_tar_max, 1), [batch, 1])  # batchsize
    y_tar = np.tile(np.random.randint(y_tar_min, y_tar_max, 1), [batch, 1])
    x_bs = np.tile(np.random.randint(x_bs_min, x_bs_max, 1), [batch, 1])
    y_bs = np.tile(np.random.randint(y_bs_min, y_bs_max, 1), [batch, 1])


    # 初始位置信息
    d_ini = np.sqrt((x_tar - x_bs) ** 2 + (y_tar - y_bs) ** 2 + h_bs ** 2)   # 距离
    tar_theta_ini = np.arccos(h_bs / d_ini)                                  # 目标所在角度
    # 感知位置信息
    s_tar_theta = tar_theta_ini  + err_theta    #                          # 感知角度
    s_d = d_ini + err_dis   #                                                  # 感知距离
    # 后续时隙信息(包含第0个时隙
    time_vec = np.tile(tao * np.linspace(0, timeslot_num - 1, timeslot_num),[batch,1])      # batchsize
    if isaway == 1:
        deltax_vec = v * time_vec + np.abs(x_tar - x_bs)                     # timeslot_num个时隙下的位置
    else:
        deltax_vec = np.abs(x_tar - x_bs) - v * time_vec                   #  靠近的话，一定注意参数设置，不能跑过了

    d_vec = np.sqrt(deltax_vec ** 2 + (y_tar - y_bs) ** 2 + h_bs ** 2)
    theta_vec = np.arccos(h_bs / d_vec)            # shape: [batchsize,timeslot]

    return d_vec,theta_vec,s_tar_theta,s_d


def large_ch(d,batch):
    # largescale_loss=(tx_gain*rx_gain*(wavelen**2))/((4*math.pi*d)**2)
    largescale_loss=10**(-3)*d**(-2.2)
    lar=np.tile(np.reshape(largescale_loss,[batch,timeslot_num,1]),[1,1,tx_N])
    h = np.sqrt(lar)
    return h

def small_ch(batch):
    H = np.random.normal(0.0, 1.0,[batch,timeslot_num,tx_N,2]) / np.sqrt(2)
    H_complex =H[:,:,:, 0]+1j*H[:,:,:, 1]
    return H_complex

def a_theta(theta,batch):
    sin_theta=np.sin(np.radians(theta))
    tx_n=np.linspace(0,tx_N-1,tx_N)
    theta_a= np.zeros([batch, timeslot_num,tx_N],dtype=complex)
    for i in range(tx_N):
        theta2 = np.exp(-1j * math.pi * tx_n[i] * sin_theta)
        theta_a[:, :, i]= theta2

    return theta_a

def channel(theta,d,batch):
    # H=large_ch(d,batch)* a_theta(theta,batch)#*small_ch(batch)
    H=large_ch(d,batch)*(np.sqrt(K_R/(1+K_R))*a_theta(theta,batch)+np.sqrt(1/(1+K_R))*small_ch(batch))
    return H

def sens_chan(theta,d,batch):
    sin_theta = np.sin(np.radians(theta))
    tx_n = np.linspace(0, tx_N - 1, tx_N)
    theta_a = np.zeros([batch, 1, tx_N], dtype=complex)
    for i in range(tx_N):
        theta2 = np.exp(-1j * math.pi * tx_n[i] * sin_theta)
        theta_a[:, :, i] = theta2


    # largescale_loss = (tx_gain * rx_gain * (wavelen ** 2)) / ((4 * math.pi * d) ** 2)
    largescale_loss = 10 ** (-3) * d ** (-2.2)
    lar = np.tile(np.reshape(largescale_loss, [batch, 1, 1]), [1, 1, tx_N])
    h = np.sqrt(lar) * theta_a
    h2=np.tile(h,[1,timeslot_num,1])

    return h2

################################################
batch=3
d_vec,theta_vec,s_tar_theta,s_d=generate_local(batch)
# print(_)
print(d_vec)
print(s_d)
# tt=a_theta(theta_vec,batch)
# print(tt)
q=np.sqrt(K_R/(1+K_R))*large_ch(d_vec,batch)
ss=np.sqrt(1/(1+K_R))*small_ch(batch)
nnnn=channel(theta_vec,d_vec,batch)
print('\n')
nn=sens_chan(s_tar_theta,s_d,batch)
# # T0 = np.zeros((batch, tx_N, timeslot_num), dtype=complex)
# # mm=T0+m
print('real(lar:',q)
print('real(small:',ss)
print('real(+small:',nnnn)
print('sens:',nn)