# for Fig1 close

import numpy as np
import math
import tensorflow as tf
import pickle
from function import *
from sensing import *

epoch=1
n_steps= 150          #100
batch_size=200
learning_rate=0.005
############################
x = tf.placeholder(tf.float32, shape=[N * batch_size])
y = tf.placeholder(tf.float32, shape=[None,N*batch_size])
H=tf.placeholder(tf.complex64,shape=[None,timeslot_num,None])
s_H=tf.placeholder(tf.complex64,shape=[None,timeslot_num,None])
Kron=tf.placeholder(tf.float32, shape=[N, N])
noisestd=tf.placeholder(tf.float32)
zero_mat = tf.placeholder(tf.float32)
lr=tf.placeholder(tf.float32)
istest=tf.placeholder(tf.float32)

# for PW normalization
p_range=np.max(weight2)-np.min(weight2)
pw=(weight2-np.min(weight2))/p_range

ffro=np.tile(np.reshape(pw,(1,N)),[batch_size,1])
# ffro=np.tile(np.reshape(weight2,(1,N)),[batch_size,1])
rho=tf.constant(1.)
initializer = tf.contrib.layers.xavier_initializer()
llr_froz_vec=tf.Variable((np.float32(ffro)-0.5))#tf.Variable((np.float32(ffro))*0.1)#tf.Variable((np.float32(ffro)-0.5))#  远离：tf.Variable(initializer([batch_size, N])*10) tf.Variable(np.float32(pa)*0.1)  靠近：tf.Variable((np.float32(ffro)-0.5))

###########
# BP
LV1 = tf.Variable(np.float32(np.ones((n_,N_,1))))#tf.Variable(np.float32(np.ones((n_,N_,1))))
RV1 = tf.Variable(np.float32(np.ones((n_,N_,1))))
LV2 = tf.Variable(np.float32(np.ones((n_,N_,1))))
RV2 = tf.Variable(np.float32(np.ones((n_,N_,1))))

########################## 测试 push bits 的 可删
total_len = tf.constant([batch_size * N])
_, index = tf.nn.top_k(llr_froz_vec, K)
aad = tf.cast(tf.reshape(tf.linspace(0.,N*(batch_size-1),batch_size),shape=[batch_size,1]),dtype=tf.int32)
pos_index=tf.reshape(tf.add(index, aad),[batch_size*K,1])
pos_index = tf.sort(pos_index, axis=0, direction="ASCENDING")
posss=tf.reshape(tf.scatter_nd(pos_index, tf.constant([1.] * (batch_size * K)), total_len), [batch_size, N])

############## 求batchsize_mod
extranum=0
total_bits = batch_size * N
if int(total_bits / k) != (total_bits / k):
    extranum = int(np.ceil((batch_size * N) / k) * k - (batch_size * N))
    total_bits=total_bits+extranum

batch_size_mod = int(total_bits / k)

############################# 函数 ##################################

############################ 传能 #############################
"Open the EH model"
file_name = 'Sys_params.pickle'
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
    out1=tf.sqrt(rho)*y       # communication
    out2=tf.sqrt(1-rho)*y          # EH
    return out1,out2

############################ 通信 #############################
def BNN(input_op):
    p = tf.clip_by_value((input_op + 1.0) / 2, 0, 1)  # Hard sigmoid

    forward_path = tf.cast(tf.greater(p, tf.random_uniform(tf.shape(p))), tf.float32)

    backward_path = tf.clip_by_value((input_op), -1.0, 1.0)

    return p,backward_path + tf.stop_gradient(forward_path - backward_path)


def fun1(pos):
    pos1 = tf.reshape(pos, [batch_size * N, ])
    pos_index = tf.cast(tf.where(tf.equal(pos1, 1)), dtype=tf.int32)
    return pos_index,pos

def fun2(total_len):
    _, index = tf.nn.top_k(llr_froz_vec, K)
    aad = tf.cast(tf.reshape(tf.linspace(0.,N*(batch_size-1),batch_size),shape=[batch_size,1]),dtype=tf.int32)
    pos_index=tf.reshape(tf.add(index, aad),[batch_size*K,1])
    pos_index = tf.sort(pos_index, axis=0, direction="ASCENDING")

    pos=tf.reshape(tf.scatter_nd(pos_index, tf.constant([1.] * (batch_size * K)), total_len), [batch_size, N])
    return pos_index,pos

def polar_encoder(x,Kr,isTest):
    total_len = tf.constant([batch_size * N])

    pro,pos=BNN(llr_froz_vec)
    pos_index,pos1=tf.cond(tf.equal(isTest,0.),lambda: fun1(pos),lambda:fun2(total_len))
    ###################################################
    # x=uG
    valid_len = tf.size(pos_index)
    valid_x=x[0:valid_len]
    u=tf.scatter_nd(pos_index,valid_x,total_len)
    u=tf.reshape(u, (batch_size, N))  # tf.reshape(u[0,1:], (batch_size, N))

    remain = x[valid_len:]
    remain=(2 * remain - 1)* inf_num       # 为loss考虑

    # bit reverse
    u0=tf.matmul(u, one_matre)
    X=tf.mod(tf.matmul(u0, Kr), 2)

    return X,pos1,remain,pro,u,valid_len,pos_index

############################### BP #####################################

def initial(x,net_dict,temp_fro2,N,n):
    for i in range(n + 1):
        for j in range(N):
            net_dict["L_{0}{1}{2}".format(i, j, 0)] = tf.zeros((batch_size))
            net_dict["R_{0}{1}{2}".format(i, j, 0)] = tf.zeros((batch_size))

    for j in range(N):
        net_dict["L_{0}{1}{2}".format(n, j, 0)] = tf.ones((1)) * x[:, j]
        mius_FZlookup = (-1*temp_fro2[:, j]+1) * inf_num
        net_dict["R_{0}{1}{2}".format(0, j, 0)] = mius_FZlookup
    return net_dict

def bp_algorithm(bp_iter_num,net_dict,RV,LV,n,N):
    # bp algorithm
    for j in range(bp_iter_num):
        itr = 0
        # itr = j
        for i in range(n-1, -1, -1):  # i决定的是第几层
            for block in range(0, int(N / 2 ** (i + 1))):
                for ps in range(2 ** i):
                    phi=block*2**(i+1)+ps
                    net_dict["L_{0}{1}{2}".format(i, phi, 0)] =LV[i, phi,itr]*fFunction(net_dict["L_{0}{1}{2}".format(i+1,phi,0)],net_dict["L_{0}{1}{2}".format(i+1,phi+2**i,0)]+net_dict["R_{0}{1}{2}".format(i,phi+2**i,0)])
                    net_dict["L_{0}{1}{2}".format(i, phi+2**i, 0)] =LV[i, phi,itr]*fFunction(net_dict["L_{0}{1}{2}".format(i+1,phi,0)],net_dict["R_{0}{1}{2}".format(i,phi,0)])+net_dict["L_{0}{1}{2}".format(i+1,phi+2**i,0)]

        for i in range(0, n):  # i决定的是第几层
            for block in range(0, int(N / 2 ** (i + 1))):
                for ps in range(2 ** i):
                    phi = block * 2 ** (i + 1) + ps
                    net_dict["R_{0}{1}{2}".format(i+1, phi, 0)] =RV[i, phi,itr]*fFunction(net_dict["L_{0}{1}{2}".format(i+1,phi+2**i,0)]+net_dict["R_{0}{1}{2}".format(i,phi+2**i,0)],net_dict["R_{0}{1}{2}".format(i,phi,0)])
                    net_dict["R_{0}{1}{2}".format(i + 1, phi+2**i, 0)] =RV[i, phi+2**i,itr]*fFunction(net_dict["R_{0}{1}{2}".format(i,phi,0)],net_dict["L_{0}{1}{2}".format(i+1,phi,0)])+net_dict["R_{0}{1}{2}".format(i,phi+2**i,0)]

    temp_arr = tf.zeros([batch_size, 1])
    for i in range(N):
        temp_llr=tf.reshape(net_dict["L_{0}{1}{2}".format(0, i, 0)],[batch_size,1])
        temp_arr=tf.concat([temp_arr,temp_llr],axis=1)

    llr = tf.reshape(temp_arr[:, 1:], [batch_size * N, ])
    llr_output = tf.reshape(llr, [batch_size,N ]) * -1  #

    return llr_output

def total_BP(input,bp_iter_num,temp_fro2,net_dict1,net_dict2):

    bp_input = tf.matmul(input, one_matre)
    input__= tf.split(bp_input, blocknum, 1)
    FZlookup__ = tf.split(temp_fro2, blocknum, 1)

    net_dict1 = initial(input__[0],net_dict1,FZlookup__[0],N_,n_)
    # net_dict2 = initial(input__[1], net_dict2, FZlookup__[1], N_, n_)

    out1 = bp_algorithm(bp_iter_num,net_dict1,RV1,LV1,n_,N_)
    # out2 = bp_algorithm(bp_iter_num, net_dict2, RV2, LV2, n_,N_)
    out=tf.concat((out1),axis=1) # ,out2

    return out



def generate_output(ori,Kr,zero,noise,isTest,H):

    ################ Polar Encoder ################
    encoded, fro_pos, rem,prob,u,va_len,posindex = polar_encoder(ori,Kr,isTest)
    temp_fro2 = get_fropos2(batch_size, fro_pos)
    encoded1 = tf.reshape(encoded, [batch_size, N])
    encoded2=tf.reshape(encoded1,[batch_size,timeslot_num, int(N/timeslot_num)])

    # encoded = tf.concat([encoded, zero], axis=0)
    # 可以补零，但此处不用补，暂时不写
    ################ Autoencoder ################
    # 4QAM
    x_enc1 = tf.sqrt(1 / 2) * (2 * encoded2 - 1) * tf.sqrt(P_S)
    x_enc2 = tf.reshape(x_enc1, [batch_size,timeslot_num, int(N/(timeslot_num*k)),2])
    z1, z2 = power_splitter(x_enc2)

    Z1_complex = tf.complex(z1[:,:,:,0], z1[:,:,:,1])
    Z2_complex = tf.complex(z2[:, :, :, 0], z2[:, :, :, 1])

    # 可删  (跳过信道)
    # encoded3=tf.sqrt(1 / 2) * (2 * encoded1 - 1)

    # # channel
    # Noise1 = tf.random.normal([batch_size, 1, int(N / (timeslot_num * k)), 2], mean=0.0, stddev=0.5)  # 测试
    # Noise2 = tf.random.normal([batch_size, 1, int(N / (timeslot_num * k)), 2], mean=0.0, stddev=1.5)
    # Noise3 = tf.random.normal([batch_size, 1, int(N / (timeslot_num * k)), 2], mean=0.0, stddev=8.)
    # Noise4 = tf.random.normal([batch_size, 1, int(N / (timeslot_num * k)), 2], mean=0.0, stddev=12.)
    # Noise = tf.concat([Noise1, Noise2, Noise3,Noise4], axis=1)

    Noise = tf.random.normal([batch_size,timeslot_num,int(N/(timeslot_num*k)),2], mean=0.0, stddev=noisestd)
    Noise_com = tf.complex(Noise[:, :, :, 0],Noise[:, :, :, 1])

    yin = Z1_complex/ s_H * H + Noise_com  #
    yen= Z2_complex/s_H*H+Noise_com #
    #
    real = tf.reshape(tf.real(yin),[batch_size,-1,1])
    imag = tf.reshape(tf.imag(yin),[batch_size,-1,1])
    out_C = tf.reshape(tf.concat((real, imag), axis=2),[batch_size,N])
    #
    real_en = tf.reshape(tf.real(yen), [batch_size, -1, 1])
    imag_en = tf.reshape(tf.imag(yen), [batch_size, -1, 1])
    out_C_en = tf.reshape(tf.concat((real_en, imag_en), axis=2), [batch_size, N])

    #
    P_del = EH( out_C_en, EH_Model)
    mod_out = -4 * tf.sqrt(1 / 2) * out_C / (2 * noisestd ** 2)
    ############### BP decoder #################
    y_out = total_BP(mod_out, bp_iter_num, temp_fro2, net_dict1,net_dict2)

    y_out_label = after_harddecision(u)

    return y_out,y_out_label,prob,fro_pos,va_len,P_del,(Z1_complex/s_H*H),posindex# 从y1开始可删

y_output,y_out_label,Prob,Fro,kk,P_d,Z,Z2=generate_output(x,Kron,zero_mat,noisestd,istest,H)
loss=1*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_out_label,logits=y_output))+\
      1*tf.reduce_mean(tf.square(Rate-tf.reduce_mean(Prob,axis=1)))#+0.1*tf.reduce_mean(Rate-tf.reduce_mean(Prob,axis=1))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)


saver = tf.train.Saver()
#############################################################
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # saver.restore(sess, 'D:/workspace/cuijingwen/stage3_2/ParamofFig3/close/256/2/save_net.ckpt')
    # saver.restore(sess, 'D:/workspace/cuijingwen/stage3_2/ParamofFig3/away/128/test/save_net.ckpt')
    ##################### train
    print('initial:  ', sess.run(posss[0:5, :]))
    print(sess.run(llr_froz_vec[0:5, :]))

    ##################################
    zzz=0
    is_test=0
    for j in range(epoch):
        train_ber = 0
        train_loss = 0
        llen=0

        for i in range(n_steps):
            X_train = np.random.choice(2, N * batch_size)
            # X_train=np.ones([N * batch_size])
            X_train0 = np.reshape(X_train, [batch_size, N])

            ######################
            d_vec,theta_vec,s_tar_theta,s_d=generate_local(batch_size)
            H_ch = channel(theta_vec,d_vec,batch_size)
            H_ch2=np.tile(np.reshape(H_ch,[batch_size,timeslot_num,tx_N,1]),[1,1,1,int(N/(timeslot_num*k*tx_N))])
            H_ch3=np.reshape(H_ch2,[batch_size,timeslot_num,tx_N*int(N/(timeslot_num*k*tx_N))])
            s_H_ch = sens_chan(s_tar_theta, s_d, batch_size)
            s_H_ch2 = np.tile(np.reshape(s_H_ch, [batch_size, timeslot_num, tx_N, 1]),[1, 1, 1, int(N / (timeslot_num * k * tx_N))])
            s_H_ch3 = np.reshape(s_H_ch2, [batch_size, timeslot_num, tx_N * int(N / (timeslot_num * k * tx_N))])


            b0_mat = [0] * extranum
            ######################## #######
            y_pred, lab, ffro, K_, P_del,_loss,_= sess.run(fetches=[y_output, y_out_label, Fro, kk,P_d, loss,optimizer],
                         feed_dict={x: X_train, Kron: get_F_kron_n(n), H:H_ch3, s_H:s_H_ch3,noisestd: noise_std, zero_mat: b0_mat,
                                    lr: learning_rate, istest: is_test})

            # zzz=ZZ[0:5,:,:]
            # print('!',Y1)
            # print(Z2_complex)
            # print(Y1!=Z2_complex)

            train_loss = _loss + train_loss
            y_pred1 = np.reshape(y_pred, (batch_size, N))
            uhat = np.zeros((batch_size, N))  # (batch_size_mod, k)
            uhat[y_pred1[:, :] >= 0] = 1
            uhat = uhat.astype(np.int64)
            uhat2 = after_harddecision00(uhat)
            uhat_info = uhat2[ffro == 1]
            llen = llen + len(uhat_info)

            train_ber = train_ber + sum(uhat_info != X_train[0:len(uhat_info)])
            print('\tEpoch:{0:d}'.format(j),
                  '\tsnr:{0:.7f}'.format(SNR_db),
                  '\tLoss:{0:.7f}'.format(_loss),
                  '\tBER:{0:.7f}'.format(
                      sum(uhat_info != X_train[0:len(uhat_info)]) / len(uhat_info)),  # (batch_size * K)
                  '\tK:{0:d}'.format(K_),
                  '\titer:{0:d}'.format(bp_iter_num),
                  '\tpdel:{0:.7f}'.format(P_del),
                  )
        print('\titer:{0:d}'.format(bp_iter_num))
        print('\tepo_Loss:{0:.7f}'.format(train_loss / n_steps))
        print('\tepo_BER:{0:.7f}'.format(train_ber / (llen)))
    # # # #
    # # #
    # save_path = saver.save(sess, 'D:/cuijingwen/PycharmProjects/stage3_2/Paramoffig1/512/kao/0.25_2/save_net.ckpt')
    # save_path = saver.save(sess, 'D:/workspace/cuijingwen/stage3_2/ParamofFig3/away/128/test/save_net.ckpt')
    # print("Model saved in file: ", save_path)  # % save_path

    # ##################### test
    print('!!!!!!!!!!!!!!!!!!!!!!!! test !!!!!!!!!!!!!!!!!!!!!!!!!!\n')
    is_test = 1

    test_ber = 0
    test_loss = 0
    llen = 0
    test_Pdel=0
    pos_pro = [0] * N

    zz=0
    zzz=0
    # th=0
    # sth=0

    for i in range(n_steps):
        X_test = np.random.choice(2, N * batch_size)
        # X_train=np.ones([N * batch_size])
        X_test0 = np.reshape(X_test, [batch_size, N])

        ######################
        d_vec, theta_vec, s_tar_theta, s_d = generate_local(batch_size)
        H_ch = channel(theta_vec, d_vec, batch_size)
        H_ch2 = np.tile(np.reshape(H_ch, [batch_size, timeslot_num, tx_N, 1]),
                        [1, 1, 1, int(N / (timeslot_num * k * tx_N))])
        H_ch3 = np.reshape(H_ch2, [batch_size, timeslot_num, tx_N * int(N / (timeslot_num * k * tx_N))])
        s_H_ch = sens_chan(s_tar_theta, s_d, batch_size)
        s_H_ch2 = np.tile(np.reshape(s_H_ch, [batch_size, timeslot_num, tx_N, 1]),
                          [1, 1, 1, int(N / (timeslot_num * k * tx_N))])
        s_H_ch3 = np.reshape(s_H_ch2, [batch_size, timeslot_num, tx_N * int(N / (timeslot_num * k * tx_N))])


        b0_mat = [0] * extranum
        ######################## #######
        y_pred, lab, ffro,ZZ,ZZZ,K_,P_del,POSSS,RELIA = sess.run(fetches=[y_output, y_out_label, Fro,Z,Z2,kk,P_d,posss,llr_froz_vec],
                                           feed_dict={x: X_test, Kron: get_F_kron_n(n), H:H_ch3, s_H:s_H_ch3,
                                                      noisestd: noise_std, zero_mat: b0_mat,
                                                      lr: learning_rate, istest: is_test})

        # zz = ZZ[0:5, :, :]
        # zzz = ZZZ[0:5, :, :]

        y_pred1 = np.reshape(y_pred, (batch_size, N))
        uhat = np.zeros((batch_size, N))  # (batch_size_mod, k)
        uhat[y_pred1[:, :] >= 0] = 1
        uhat = uhat.astype(np.int64)
        uhat2 = after_harddecision00(uhat)
        uhat_info = uhat2[ffro == 1]
        llen = llen + len(uhat_info)

        test_ber = test_ber + sum(uhat_info != X_test[0:len(uhat_info)])
        test_Pdel = test_Pdel + P_del
        pos_pro = pos_pro + np.sum(POSSS, axis=0)

    pos_pro2 = pos_pro / (batch_size * n_steps)
    test_Pdel=test_Pdel/n_steps

    if isaway==0:
        print('靠近')
    else:
        print('远离')

    print('码字：', '(', N, ',', K, ')')
    print('SNR:', SNR_db)
    print('iter:{0:d}'.format(bp_iter_num))
    print('test_BER:{0:.7f}'.format(test_ber / (llen)))
    print('Test Pdel:  ', test_Pdel)
    print(K_)
    print('final:  ', sess.run(posss[0:5, :]))



    ####################### save reliability #######################
    chan_rel=normalization(RELIA,batch_size)
    rr=np.sum(chan_rel, axis=0)/batch_size
    # print(np.shape(rr))
    np.savetxt('reliabilityyuan2/256/test/'+str(N)+'s'+str(SNR_db)+'e'+str(epoch)+'step150_f.txt', rr)


    ##############################################
    # print('x', zz)
    # print('h/h', zzz)
    # print('分割\n')
    # print('h',th[0:5,:, :])
    # print('sh',sth[0:5,:, :])
    # print('\n')
    # print('np',th[0:5,:, :]/sth[0:5,:, :])

    # plot
    xva = np.linspace(0, N - 1, N)
    plt.scatter(xva, rr, s=30, label='probability')
    plt.xlabel('bit index')
    plt.ylabel('Reliability')
    # plt.gca().set_ylim(0, 1)
    plt.gca().set_xlim(0, N)

    plt.grid(True, which="both")
    # plt.legend(['probability'],loc='upper right', ncol=1)
    plt.show()

sess.close()

