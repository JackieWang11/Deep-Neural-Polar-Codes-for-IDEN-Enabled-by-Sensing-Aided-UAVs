from function import *
from sensing import *

from WithEnSingleBNN.B0 import train0,test0
from WithEnSingleBNN.B1 import train1,test1
from WithEnSingleBNN.B2 import train2,test2
from WithEnSingleBNN.B3 import train3,test3

p=0.08

# print('------------ Train slot1 -----------------')
# train0(p)
# print('------------ Train slot2 -----------------')
# train1(p)
# print('------------ Train slot3 -----------------')
# train2(p)
# print('------------ Train slot4 -----------------')
# train3(p)
# print('------------ Finished! ------------ ')


# step=5
# ber=0
# pdel=0
# BER0=0
# BER1=0
# BER2=0
# BER3=0
#
# c_batchsize=200

# for i in range(0,step):
#     d_vec, theta_vec, s_tar_theta, s_d = generate_local(c_batchsize)
#     H_ch = channel(theta_vec, d_vec, c_batchsize)
#     H_ch2 = np.tile(np.reshape(H_ch, [c_batchsize, timeslot_num, tx_N, 1]),
#                     [1, 1, 1, int(N*4 / (timeslot_num * k * tx_N))])
#     H_ch3 = np.reshape(H_ch2, [c_batchsize, timeslot_num, tx_N * int(N*4 / (timeslot_num * k * tx_N))])
#     s_H_ch = sens_chan(s_tar_theta, s_d, c_batchsize)
#     s_H_ch2 = np.tile(np.reshape(s_H_ch, [c_batchsize, timeslot_num, tx_N, 1]),
#                       [1, 1, 1, int(N*4 / (timeslot_num * k * tx_N))])
#     s_H_ch3 = np.reshape(s_H_ch2, [c_batchsize, timeslot_num, tx_N * int(N*4 / (timeslot_num * k * tx_N))])
#
#     np.save('D:/cuijingwen/PycharmProjects/stage3/ParamOfSingleBNN/h/H/t1', H_ch3)
#     np.save('D:/cuijingwen/PycharmProjects/stage3/ParamOfSingleBNN/h/s_H/t1', s_H_ch3)
#
#
#     # print('------------ Test slot1 -----------------')
#     ber0,pdel0=test0(p)
#     # print('------------ Test slot2 -----------------')
#     ber1,pdel1=test1(p)
#     # print('------------ Test slot3 -----------------')
#     ber2,pdel2=test2(p)
#     # print('------------ Test slot4 -----------------')
#     ber3,pdel3=test3(p)
#
#     BER0 = BER0+ber0
#     BER1 = BER1 + ber1
#     BER2 = BER2 + ber2
#     BER3 = BER3 + ber3
# #
#     ber = ber+(ber0 + ber1 + ber2 + ber3) / 4  #
#     pdel=pdel+(pdel0 + pdel1+ pdel2 + pdel3 ) / 4    #
#
# ber=ber/step
# pdel=pdel/step
# print('------------ Test slot1 -----------------')
# print('ber0=',BER0/step)
# print('------------ Test slot2 -----------------')
# print('ber1=',BER1/step)
# print('------------ Test slot3 -----------------')
# print('ber2=',BER2/step)
# print('------------ Test slot4 -----------------')
# print('ber3=',BER3/step)

###############################################
print('------------ Test slot1 -----------------')
ber0,pdel0=test0(p)
print('------------ Test slot2 -----------------')
ber1,pdel1=test1(p)
print('------------ Test slot3 -----------------')
ber2,pdel2=test2(p)
print('------------ Test slot4 -----------------')
ber3,pdel3=test3(p)

ber = (ber0 + ber1 + ber2 + ber3) / 4  #
pdel=(pdel0 + pdel1+ pdel2 + pdel3 ) / 4

print('------------ Test slot1 -----------------')
print('ber0=',ber0)
print('------------ Test slot2 -----------------')
print('ber1=',ber1)
print('------------ Test slot3 -----------------')
print('ber2=',ber2)
print('------------ Test slot4 -----------------')
print('ber3=',ber3)
##############################################

print('------------ Total -----------------')
if isaway == 0:
    print('靠近')
else:
    print('远离')
print('SNR:',SNR_db)
print('总码字：','(',N*4,',',K*4,')')
print('ber=',ber)
print('pdel=',pdel)