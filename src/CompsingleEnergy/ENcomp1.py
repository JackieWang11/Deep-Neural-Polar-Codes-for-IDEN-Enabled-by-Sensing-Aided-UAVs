import numpy as np
import pickle
import tensorflow as tf
# from function import *
from scipy.linalg import *
# from sensing import *

c_step=1
c_batchsize=200

M = 4                      # M-QAM
k = int(np.log2(M))
N=128
K=64
n=int(np.log2(N))


blocknum=1
bp_iter_num=2
inf_num=1000
timeslot=1

# N_total=N*timeslot_num
# K_total=K*timeslot_num

N_ = int(N / blocknum)
n_ = int(np.log2(N / blocknum))

cLV1 = np.float32(np.ones((n_,N_,1)))
cRV1 = np.float32(np.ones((n_,N_,1)))
cLV2 = np.float32(np.ones((n_,N_,1)))
cRV2 = np.float32(np.ones((n_,N_,1)))

cnet_dict1 = {}
cnet_dict2 = {}

# ###############  get bit information ############## 放 function了
indices = np.loadtxt('../PW/'+str(N)+'.txt',dtype=int)
FZlookup = np.zeros((N))
FZlookup[indices[:K]] = 1            # 1代表信息位
###################################################

########### bitreverse ###########
bitreversedindices = np.zeros((N),dtype=int)
for i in range(N):
    b = '{:0{width}b}'.format(i, width=n)
    bitreversedindices[i] = int(b[::-1], 2)
################################################
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

def power_splitter(y,rho):
    out1=np.sqrt(rho)*y       # communication
    out2=np.sqrt(1-rho)*y          # EH
    return out1,out2

def get_F_kron_n(n):
    n=int(n)
    G = np.array([[1, 0], [1, 1]])
    GKron = G
    for j in range(0, n - 1):
        GKron = np.kron(GKron, G)

    return GKron

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
                [diag, np.reshape(in_later, shape=[bachsize, -1]), np.ones((bachsize, int(pi / 2)), dtype=np.float32)],
                axis=1)

        temp_fro = np.multiply(temp_fro, diag[:, 1:])
    temp_fro2=(temp_fro - 1)*(-1)
    temp_fro2 = temp_fro2.astype('float32')#tf.cast(temp_fro - 1, dtype=tf.float32)*(-1)

    return temp_fro2


def generate_S_matr00(subblock_num):
    # 生成单个
    mat_scale=int(N/subblock_num)
    identity_mat=np.identity(int(mat_scale/2))
    zero_mat=np.zeros((int(mat_scale/2),int(mat_scale/2)))
    singel_mat=np.block([
    [identity_mat,zero_mat],
    [identity_mat, identity_mat ]])
    # 组装
    S_mat = [[0]]
    for i in range(subblock_num):
        S_mat=block_diag(S_mat, singel_mat)

    return S_mat[1:,1:]

def after_harddecision00(input):
    stage = int(np.log2(blocknum))
    total_mat=np.identity(N)
    for i in range(n - stage + 1, n + 1):
        subblock_num = int(N / (2 ** i))
        total_mat=np.matmul(total_mat,generate_S_matr00(subblock_num))

    out=np.mod(np.matmul(input,total_mat),2)
    return out


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



def total_BP(input,bp_iter_num,temp_fro2,cnet_dict1,cnet_dict2):

    bp_input = input[:, bitreversedindices]
    input__= np.split(bp_input, blocknum, 1)
    FZlookup__ = np.split(temp_fro2, blocknum, 1)

    cnet_dict1= c_initial(input__[0],cnet_dict1,FZlookup__[0],N_,n_)
    # net_dict2 = initial(input__[1], cnet_dict2, FZlookup__[1], N_, n_)

    out1 = c_bp_algorithm(bp_iter_num,cnet_dict1,cRV1,cLV1,n_,N_)
    # out2 = bp_algorithm(bp_iter_num, cnet_dict2, cRV2, cLV2, n_,N_)
    out=out1
    # out=np.concatenate((out1,out2),axis=1) #

    return out





def sim1(P_S,rho,noise_std):
    ber=0
    Pdel_sum=0

    N0=(2*noise_std**2)

    for it in range(0, c_step):#1
        x_qam, y_qam = c_encoder()  # 二进制比特

        x_enc1 = np.sqrt(1 / 2) * (2 * x_qam - 1)*np.sqrt(P_S)
        x_enc2 = np.reshape(x_enc1, [c_batchsize, timeslot, int(N / (timeslot * k)), 2])
        Z_complex = x_enc2[:, :, :, 0]+1j*x_enc2[:, :, :, 1]
        z1, z2 = power_splitter(Z_complex,rho)

        ################### Channel ###########################
        # d_vec, theta_vec, s_tar_theta, s_d = generate_local(c_batchsize)
        # H_ch = channel(theta_vec, d_vec, c_batchsize)
        # H_ch2 = np.tile(np.reshape(H_ch, [c_batchsize, timeslot_num, tx_N, 1]),
        #                 [1, 1, 1, int(N / (timeslot_num * k * tx_N))])
        # H_ch3 = np.reshape(H_ch2, [c_batchsize, timeslot_num, tx_N * int(N / (timeslot_num * k * tx_N))])
        # s_H_ch = sens_chan(s_tar_theta, s_d, c_batchsize)
        # s_H_ch2 = np.tile(np.reshape(s_H_ch, [c_batchsize, timeslot_num, tx_N, 1]),
        #                   [1, 1, 1, int(N / (timeslot_num * k * tx_N))])
        # s_H_ch3 = np.reshape(s_H_ch2, [c_batchsize, timeslot_num, tx_N * int(N / (timeslot_num * k * tx_N))])

        #
        # np.save('ParamOfComp/H/t1', H_ch3)
        # np.save('ParamOfComp/s_H/t1', H_ch3)

        H_ch = np.load('../ParamOfSingle/H/t1.npy')
        s_H_ch = np.load('../ParamOfSingle/s_H/t1.npy')

        H_ch3=np.reshape(H_ch[:, 1, :], [c_batchsize, 1, -1])
        s_H_ch3 = np.reshape(s_H_ch[:, 1, :], [c_batchsize, 1, -1])
        # print('\n0:H:',H_ch3)
        # print('\n0:sH:', s_H_ch3)
        # print('\n0', H_ch3 / s_H_ch3)
        ############################

        Noise = np.random.normal(0,noise_std,[c_batchsize, timeslot, int(N / (timeslot * k)), 2])
        Noise_com = Noise[:, :, :, 0]+1j* Noise[:, :, :, 1]

        y1 = z1 / s_H_ch3 * H_ch3 + Noise_com  #
        y2 = z2 / s_H_ch3 * H_ch3 + Noise_com  #

        real1 = np.reshape(np.real(y1), [c_batchsize, -1, 1])
        imag1 = np.reshape(np.imag(y1), [c_batchsize, -1, 1])
        out_C1 = np.reshape(np.concatenate((real1, imag1), axis=2), [c_batchsize, N])

        real2 = np.reshape(np.real(y2), [c_batchsize, -1, 1])
        imag2 = np.reshape(np.imag(y2), [c_batchsize, -1, 1])
        out_C2 = np.reshape(np.concatenate((real2, imag2), axis=2), [c_batchsize, N])

        out_C2 = out_C2.astype('float32')
        P_del = EH(out_C2, EH_Model)

        ###################### ## #############################

        # demodulation
        demod_bits= -4 * np.sqrt(1 / 2) * out_C1 / N0

        # decoding
        FZlookup2=np.tile(np.reshape(FZlookup,[1,N]),[c_batchsize,1])
        FZlookup3=c_get_fropos2(c_batchsize,FZlookup2)
        y_out = total_BP(demod_bits, bp_iter_num, FZlookup3, cnet_dict1, cnet_dict2)

        outhat = np.zeros((c_batchsize, N))  # (batch_size_mod, k)
        outhat[y_out >= 0] = 1

        uhat = after_harddecision00(outhat)
        uhat_info = uhat[FZlookup2 == 1]
        uhat_info=np.reshape(uhat_info,[c_batchsize,K])
        ber = ber + sum(sum(uhat_info != y_qam))#sum(uhat_info != y_qam)
        Pdel_sum = Pdel_sum + P_del

    ber = ber/(c_batchsize*K*c_step)
    Pdel_sum = Pdel_sum / c_step

    return ber,Pdel_sum

# qam_ber = sim3()
# print(qam_ber)