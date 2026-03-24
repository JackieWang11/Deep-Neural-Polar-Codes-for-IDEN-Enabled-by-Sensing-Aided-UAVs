# BP是按blocknum形式来的，block=1

import numpy as np
import pickle
from function import *
from sensing import *

######

c_step=500
c_batchsize=500
rho=0.5

cLV1 = np.float32(np.ones((n_,N_,1)))
cRV1 = np.float32(np.ones((n_,N_,1)))
cLV2 = np.float32(np.ones((n_,N_,1)))
cRV2 = np.float32(np.ones((n_,N_,1)))
cLV3 = np.float32(np.ones((n_,N_,1)))
cRV3 = np.float32(np.ones((n_,N_,1)))
cLV4 = np.float32(np.ones((n_,N_,1)))
cRV4 = np.float32(np.ones((n_,N_,1)))

cnet_dict1 = {}
cnet_dict2 = {}
cnet_dict3 = {}
cnet_dict4 = {}
# ###############  get bit information ############## 放 function了
# indices = np.loadtxt('FrozenBitreal/'+str(N)+'.txt',dtype=int)
# FZlookup = np.zeros((N))
# FZlookup[indices[:K]] = 1            # 1代表信息位

# indices = np.loadtxt('GA_128/'+str(SNR_db)+'.txt',dtype=int)
# FZlookup = np.zeros((N))
# FZlookup[indices[:K]] = 1            # 1代表信息位 0为冻结位
###################################################
############################ 传能 #############################
"Open the EH model"
file_name = '../Sys_params.pickle'
with open(file_name, 'rb') as f:
    EH_Model = pickle.load(f)

def EH(Rx,EH_Model):
    W1, b1 = EH_Model['W1'], EH_Model['b1']
    W2, b2 = EH_Model['W2'], EH_Model['b2']
    W3, b3 = EH_Model['W3'], EH_Model['b3']
    W4, b4 = EH_Model['W4'], EH_Model['b4']
    W5, b5 = EH_Model['W5'], EH_Model['b5']

    Rx_new = tf.transpose(tf.reshape(tf.transpose(Rx), [-1, 2]))
    P_in = 4.342944819032518 * tf.log(tf.reduce_sum(Rx_new ** 2, axis=0, keepdims=True))

    Z1 = tf.matmul(W1, P_in) + b1
    A1 = tf.nn.tanh(Z1)
    Z2 = tf.matmul(W2, A1) + b2
    A2 = tf.nn.tanh(Z2)
    Z3 = tf.matmul(W3, A2) + b3
    A3 = tf.nn.tanh(Z3)
    Z4 = tf.matmul(W4, A3) + b4
    A4 = tf.nn.tanh(Z4)
    Z5 = tf.matmul(W5, A4) + b5
    del_power = tf.reduce_mean(tf.nn.tanh(Z5))

    return del_power

def power_splitter(y):
    out1=np.sqrt(rho)*y       # communication
    out2=np.sqrt(1-rho)*y          # EH
    return out1,out2

def c_encoder():
    u=np.random.choice(2, K*c_batchsize)  # 生成信息比特
    # u = np.zeros([c_batchsize * K])
    u2=np.reshape(u, (c_batchsize, K))

    x = np.tile(FZlookup.copy(), (c_batchsize))       # len(x)=N
    x[x == 1] = u  # -1's will get replaced by message bits below
    x = np.reshape(x, (-1, N))
    x = x[:, bitreversedindices]  # bit reversal
    x = np.mod(np.matmul(x, get_F_kron_n(n)), 2)  # encode
    return x,u2

def c_bin2dec(binary):
    dec=[]
    for i in range(0,len(binary)):
        arr=binary[i]
        result=0
        for j in range(len(arr)):
            # 利用for循环及切片从右至左依次取出，然后再用内置方法求2的次方
            result += int(arr[-(j + 1)]) * pow(2, j)
        dec.append(result)
    return dec

def c_get_fropos2(bachsize,temp_fro0):

    temp_fro = -1 * (temp_fro0 - 1)

    # ttt=temp_fro
    for i in range(0, int(np.log2(blocknum))):
        # 每一个stage都有一个矩阵
        pi = int(N / (2 ** i))
        diag = np.zeros([bachsize, 1], dtype=np.float32)
        for j in range(0, 2 ** (i)):  # 用矩阵吧 #组建矩阵
            in_ = temp_fro[:, j * pi:(j + 1) * pi]
            in_later = in_[:, int(pi / 2):pi]
            diag = np.concatenate(
                [diag, np.reshape(in_later, [bachsize, -1]), np.ones((bachsize, int(pi / 2)), dtype=np.float32)],
                axis=1)

        temp_fro = np.multiply(temp_fro, diag[:, 1:])
    temp_fro2=(temp_fro - 1)*(-1)
    temp_fro2 = temp_fro2.astype('float32')#tf.cast(temp_fro - 1, dtype=tf.float32)*(-1)

    return temp_fro2


def c_fFunction(a,b):
    c = np.sign(a)*np.sign(b)*np.minimum(np.abs(a),np.abs(b))
    return c


def c_initial(x,net_dict,temp_fro2,N,n):
    for i in range(n + 1):
        for j in range(N):
            net_dict["L_{0}{1}{2}".format(i, j, 0)] = np.zeros((c_batchsize))
            net_dict["R_{0}{1}{2}".format(i, j, 0)] = np.zeros((c_batchsize))

    for j in range(N):
        net_dict["L_{0}{1}{2}".format(n, j, 0)] = np.ones((1)) * x[:, j]
        mius_FZlookup = (-1*temp_fro2[:, j]+1) * inf_num
        net_dict["R_{0}{1}{2}".format(0, j, 0)] = mius_FZlookup
    return net_dict

def c_bp_algorithm(bp_iter_num,net_dict,RV,LV,n,N):
    # bp algorithm
    for j in range(bp_iter_num):
        itr = 0
        for i in range(n-1, -1, -1):  # i决定的是第几层
            for block in range(0, int(N / 2 ** (i + 1))):
                for ps in range(2 ** i):
                    phi=block*2**(i+1)+ps
                    net_dict["L_{0}{1}{2}".format(i, phi, 0)] =LV[i, phi,itr]*c_fFunction(net_dict["L_{0}{1}{2}".format(i+1,phi,0)],net_dict["L_{0}{1}{2}".format(i+1,phi+2**i,0)]+net_dict["R_{0}{1}{2}".format(i,phi+2**i,0)])
                    net_dict["L_{0}{1}{2}".format(i, phi+2**i, 0)] =LV[i, phi,itr]*c_fFunction(net_dict["L_{0}{1}{2}".format(i+1,phi,0)],net_dict["R_{0}{1}{2}".format(i,phi,0)])+net_dict["L_{0}{1}{2}".format(i+1,phi+2**i,0)]

        for i in range(0, n):  # i决定的是第几层
            for block in range(0, int(N / 2 ** (i + 1))):
                for ps in range(2 ** i):
                    phi = block * 2 ** (i + 1) + ps
                    net_dict["R_{0}{1}{2}".format(i+1, phi, 0)] =RV[i, phi,itr]*c_fFunction(net_dict["L_{0}{1}{2}".format(i+1,phi+2**i,0)]+net_dict["R_{0}{1}{2}".format(i,phi+2**i,0)],net_dict["R_{0}{1}{2}".format(i,phi,0)])
                    net_dict["R_{0}{1}{2}".format(i + 1, phi+2**i, 0)] =RV[i, phi+2**i,itr]*c_fFunction(net_dict["R_{0}{1}{2}".format(i,phi,0)],net_dict["L_{0}{1}{2}".format(i+1,phi,0)])+net_dict["R_{0}{1}{2}".format(i,phi+2**i,0)]

    temp_arr = np.zeros([c_batchsize, 1])
    for i in range(N):
        temp_llr=np.reshape(net_dict["L_{0}{1}{2}".format(0, i, 0)],[c_batchsize,1])
        temp_arr=np.concatenate([temp_arr,temp_llr],axis=1)

    llr = np.reshape(temp_arr[:, 1:], [c_batchsize * N, ])
    llr_output = np.reshape(llr, [c_batchsize,N ]) * -1  #

    return llr_output



def total_BP(input,bp_iter_num,temp_fro2,cnet_dict1,cnet_dict2,cnet_dict3,cnet_dict4):

    bp_input = input[:, bitreversedindices]
    input__= np.split(bp_input, blocknum, 1)
    FZlookup__ = np.split(temp_fro2, blocknum, 1)

    cnet_dict1= c_initial(input__[0],cnet_dict1,FZlookup__[0],N_,n_)
    # cnet_dict2 = c_initial(input__[1], cnet_dict2, FZlookup__[1], N_, n_)
    # cnet_dict3 = c_initial(input__[2], cnet_dict3, FZlookup__[2], N_, n_)
    # cnet_dict4 = c_initial(input__[3], cnet_dict4, FZlookup__[3], N_, n_)

    out1 = c_bp_algorithm(bp_iter_num,cnet_dict1,cRV1,cLV1,n_,N_)
    # out2 = c_bp_algorithm(bp_iter_num, cnet_dict2, cRV2, cLV2, n_,N_)
    # out3 = c_bp_algorithm(bp_iter_num, cnet_dict3, cRV3, cLV3, n_, N_)
    # out4 = c_bp_algorithm(bp_iter_num, cnet_dict4, cRV4, cLV4, n_, N_)

    out=out1
    # out=np.concatenate((out1,out2),axis=1) #
    # out = np.concatenate((out1, out2,out3,out4), axis=1)

    return out





def sim(noise_std):
    ber=0
    Pdel_sum = 0

    N0=(2*noise_std**2)

    for it in range(0, c_step):
        x_qam, y_qam = c_encoder()  # 二进制比特
        x_enc1 = np.sqrt(1 / 2) * (2 * x_qam - 1) * np.sqrt(P_S)
        x_enc2 = np.reshape(x_enc1, [c_batchsize, timeslot_num, int(N / (timeslot_num * k)), 2])
        Z_complex = x_enc2[:, :, :, 0]+1j*x_enc2[:, :, :, 1]
        z1, z2 = power_splitter(Z_complex)
        # _, z2 = power_splitter(x_enc1)

        ################### Channel ###########################
        d_vec, theta_vec, s_tar_theta, s_d = generate_local(c_batchsize)
        H_ch = channel(theta_vec, d_vec, c_batchsize)
        H_ch2 = np.tile(np.reshape(H_ch, [c_batchsize, timeslot_num, tx_N, 1]),
                        [1, 1, 1, int(N / (timeslot_num * k * tx_N))])
        H_ch3 = np.reshape(H_ch2, [c_batchsize, timeslot_num, tx_N * int(N / (timeslot_num * k * tx_N))])
        s_H_ch = sens_chan(s_tar_theta, s_d, c_batchsize)
        s_H_ch2 = np.tile(np.reshape(s_H_ch, [c_batchsize, timeslot_num, tx_N, 1]),
                          [1, 1, 1, int(N / (timeslot_num * k * tx_N))])
        s_H_ch3 = np.reshape(s_H_ch2, [c_batchsize, timeslot_num, tx_N * int(N / (timeslot_num * k * tx_N))])

        ###############################3

        Noise = np.random.normal(0,noise_std,[c_batchsize, timeslot_num, int(N / (timeslot_num * k)), 2])
        Noise_com = Noise[:, :, :, 0]+1j* Noise[:, :, :, 1]

        y1 = z1/ s_H_ch3 * H_ch3 + Noise_com#
        y2 = z2/ s_H_ch3 * H_ch3  + Noise_com  #

        real1 = np.reshape(np.real(y1), [c_batchsize, -1, 1])
        imag1 = np.reshape(np.imag(y1), [c_batchsize, -1, 1])
        out_C1 = np.reshape(np.concatenate((real1, imag1), axis=2), [c_batchsize, N])
        #
        real2 = np.reshape(np.real(y2), [c_batchsize, -1, 1])
        imag2 = np.reshape(np.imag(y2), [c_batchsize, -1, 1])
        out_C2 = np.reshape(np.concatenate((real2, imag2), axis=2), [c_batchsize, N])

        # out_C2 = z2.astype('float32')
        out_C2 = out_C2.astype('float32')
        P_del = EH(out_C2, EH_Model)

        ###################### ## #############################

        # demodulation
        demod_bits= -4 * np.sqrt(1 / 2) * out_C1 / N0

        # decoding
        FZlookup2=np.tile(np.reshape(FZlookup,[1,N]),[c_batchsize,1])
        FZlookup3=c_get_fropos2(c_batchsize,FZlookup2)
        y_out = total_BP(demod_bits, bp_iter_num, FZlookup3, cnet_dict1, cnet_dict2,cnet_dict3,cnet_dict4)

        outhat = np.zeros((c_batchsize, N))  # (batch_size_mod, k)
        outhat[y_out >= 0] = 1

        uhat = after_harddecision00(outhat)
        uhat_info = uhat[FZlookup2 == 1]
        uhat_info=np.reshape(uhat_info,[c_batchsize,K])
        ber = ber + sum(sum(uhat_info != y_qam))#sum(uhat_info != y_qam)
        Pdel_sum = Pdel_sum + P_del

    ber = ber/(c_batchsize*K*c_step)
    Pdel_sum = Pdel_sum/c_step

    return ber,Pdel_sum

qam_ber,pdel = sim(noise_std)

if isaway == 0:
    print('靠近')
else:
    print('远离')

print('SNR:',SNR_db)
print('码字：','(',N,',',K,')')
print('ber=',qam_ber)
array = tf.Session().run(pdel)
print("pdel= ",array)
print('blocknum=',blocknum)
print('iter=',bp_iter_num)