# 单时隙对比的总函数

from sensing import *
# from function import *
from CompsingleEnergy.ENcomp0 import sim0
from CompsingleEnergy.ENcomp1 import sim1
from CompsingleEnergy.ENcomp2 import sim2
from CompsingleEnergy.ENcomp3 import sim3

N=128*timeslot_num
M=4
k = int(np.log2(M))
c_batchsize=200
rho=0.1

step=800
ber=0
pdel=0

for i in range(0,step):
    d_vec, theta_vec, s_tar_theta, s_d = generate_local(c_batchsize)
    H_ch = channel(theta_vec, d_vec, c_batchsize)
    H_ch2 = np.tile(np.reshape(H_ch, [c_batchsize, timeslot_num, tx_N, 1]),
                    [1, 1, 1, int(N / (timeslot_num * k * tx_N))])
    H_ch3 = np.reshape(H_ch2, [c_batchsize, timeslot_num, tx_N * int(N / (timeslot_num * k * tx_N))])
    s_H_ch = sens_chan(s_tar_theta, s_d, c_batchsize)
    s_H_ch2 = np.tile(np.reshape(s_H_ch, [c_batchsize, timeslot_num, tx_N, 1]),
                      [1, 1, 1, int(N / (timeslot_num * k * tx_N))])
    s_H_ch3 = np.reshape(s_H_ch2, [c_batchsize, timeslot_num, tx_N * int(N / (timeslot_num * k * tx_N))])

    np.save('../ParamOfSingle/H/t1', H_ch3)
    np.save('../ParamOfSingle/s_H/t1', s_H_ch3)

    ber0,P1 = sim0(P_S,rho,noise_std)
    ber1,P2 = sim1(P_S,rho,noise_std)
    ber2,P3 = sim2(P_S,rho,noise_std)
    ber3,P4 = sim3(P_S,rho,noise_std)
    ber = ber+(ber0 + ber1 + ber2 + ber3) / 4  #
    pdel=pdel+(P1+P2+P3+P4)

ber=ber/step
pdel=pdel/step
pd = tf.Session().run(pdel)
p1 = tf.Session().run(P1)
p2 = tf.Session().run(P2)
p3 = tf.Session().run(P3)
p4 = tf.Session().run(P4)


if isaway == 0:
    print('靠近')
else:
    print('远离')

print('SNR:',SNR_db)
print('\n')
print('\ttest_BER:{0:.7f}'.format(ber))
print('\n各时隙BER：')
print('\ttest_BER:{0:.7f}'.format(ber0))
print('\ttest_BER:{0:.7f}'.format(ber1))
print('\ttest_BER:{0:.7f}'.format(ber2))
print('\ttest_BER:{0:.7f}'.format(ber3))
print('\n')
print('\ttest_P:{0:.7f}'.format(pd))
print('\t均值能量:{0:.7f}'.format(pd/4))
print('\n各时隙能量：')
print('\ttest_P:{0:.7f}'.format(p1))
print('\ttest_P:{0:.7f}'.format(p2))
print('\ttest_P:{0:.7f}'.format(p3))
print('\ttest_P:{0:.7f}'.format(p4))


# print('\nH:',H_ch3[:,0,:])
# print('\nsH:',s_H_ch3[:,0,:])
# zz=H_ch3/s_H_ch3
# print('\n相除结果：')
# print(zz)#np.abs(zz)


# H_ch322 = np.load('ParamOfComp/H/t1.npy')
# H_ch322=np.reshape(H_ch322[:,0,:],[c_batchsize,1,-1])
# print(H_ch3[:,0,:])
# print(H_ch322)
# print(np.shape(H_ch3))
# print(np.shape(H_ch3[:,0,:]))
# print(np.shape(H_ch322))



