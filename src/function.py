#
import numpy as np
import tensorflow as tf
from scipy.linalg import *
import matplotlib.pyplot as plt

# np.random.seed(42)
# tf.random.set_random_seed(42)

M = 4                      # M-QAM
k = int(np.log2(M))
# N=  512#128#     64                # 码长   # 至少16 bit
# K=  256#32#  32                   # 信息比特数
N=  256
K=  256
n=int(np.log2(N))
Rate=K/N
P_tar=0.084
lam=0                      # 能量需求
######################## Partition BP 参数 ###########
inf_num=1000
# net_dict = {}
bp_iter_num=2
blocknum=1
N_ = int(N / blocknum)
n_ = int(np.log2(N / blocknum))
##################################
net_dict1 = {}
net_dict2 = {}
net_dict3 = {}
net_dict4 = {}
###############  get bit information ##############
# PW
weight = np.loadtxt('../PW/weight'+str(N)+'.txt',dtype=float)   #
weight2=weight[::-1]
#
indices = np.loadtxt('../PW/'+str(N)+'.txt',dtype=int)
# indices = np.loadtxt('D:/workspace/cuijingwen/stage3_2/FrozenBitreal/'+str(N)+'.txt',dtype=int)
FZlookup = np.zeros((N))
FZlookup[indices[:K]] = 1            # 1代表信息位
######################## standard constellation ########################
# 4QAM
sym=np.array([-1,-1,-1,1,1,-1,1,1])
sym2=np.reshape(sym,[M,2,1])
sym_nor=sym2/np.sqrt(2*np.mean(np.square(sym2)))

########### bitreverse ###########
bitreversedindices = np.zeros((N),dtype=int)
for i in range(N):
    b = '{:0{width}b}'.format(i, width=n)
    bitreversedindices[i] = int(b[::-1], 2)

one_mat=np.eye(N)
one_matre=tf.constant(one_mat[:,bitreversedindices],dtype=tf.float32)

########################## function ##########################

def normalization(data,batch):
    _range = np.max(data, axis=1) - np.min(data, axis=1)
    _range = np.reshape(_range, [batch, 1])
    minm = np.reshape(np.min(data, axis=1), [batch, 1])
    rr = (data - minm) / _range
    return rr

def get_F_kron_n(n):
    n=int(n)
    G = np.array([[1, 0], [1, 1]])
    GKron = G
    for j in range(0, n - 1):
        GKron = np.kron(GKron, G)

    return GKron


############ BP #############
def fFunction(a,b):
    c = tf.sign(a)*tf.sign(b)*tf.minimum(tf.abs(a),tf.abs(b))
    return c

def buling(batch_size):  #补零
    extranum = 0
    total_bits = batch_size * N
    if int(total_bits / k) != (total_bits / k):
        extranum = int(np.ceil((batch_size * N) / k) * k - (batch_size * N))
        total_bits = total_bits + extranum

    batch_size_mod = int(total_bits / k)
    return batch_size_mod

####################### stage3
def generate_S_matr(subblock_num):
    # 生成单个
    mat_scale=int(N/subblock_num)
    identity_mat=tf.cast(tf.diag(tf.ones([mat_scale/2])),tf.float32)#tf.identity(int(mat_scale/2))
    zero_mat=tf.zeros((int(mat_scale/2),int(mat_scale/2)))
    upp=tf.concat([identity_mat,zero_mat],axis=1)
    down=tf.concat([identity_mat,identity_mat],axis=1)
    singel_mat =tf.concat([upp,down],axis=0)
    # 组装
    S_mat = tf.linalg.LinearOperatorFullMatrix(tf.zeros([1,1]))#[[0]]
    for i in range(subblock_num):
        S_mat=tf.linalg.LinearOperatorBlockDiag([S_mat, tf.linalg.LinearOperatorFullMatrix(singel_mat)])

    return S_mat.to_dense()[1:,1:]

def after_harddecision(input):
    stage = int(np.log2(blocknum))
    total_mat=tf.cast(tf.diag(tf.ones([N])),tf.float32)
    for i in range(n, n - stage ,-1): # (n - stage + 1, n + 1)
        subblock_num = int(N / (2 ** i))
        total_mat=tf.matmul(total_mat,generate_S_matr(subblock_num))

    out=tf.mod(tf.matmul(input,total_mat),2)
    return out

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


def get_fropos2(bachsize,temp_fro0):

    # main2    # 0.999算info
    # temp_fro00 = tf.sign(-1*temp_fro0 )
    # temp_fro = 1+ temp_fro00

    # main        0.999也算frozen
    # temp_fro00 = tf.sign(temp_fro0 - 1.)
    # temp_fro = -1 * temp_fro00
    temp_fro = -1 * (temp_fro0 - 1)


    # ttt=temp_fro
    for i in range(0, int(np.log2(blocknum))):
        # 每一个stage都有一个矩阵
        pi = int(N / (2 ** i))
        diag = tf.zeros([bachsize, 1], dtype=tf.float32)
        for j in range(0, 2 ** (i)):  # 用矩阵吧 #组建矩阵
            in_ = temp_fro[:, j * pi:(j + 1) * pi]
            in_later = in_[:, int(pi / 2):pi]
            diag = tf.concat(
                [diag, tf.reshape(in_later, shape=[bachsize, -1]), tf.ones((bachsize, int(pi / 2)), dtype=tf.float32)],
                axis=1)

        temp_fro = tf.multiply(temp_fro, diag[:, 1:])
    temp_fro2 = tf.cast(temp_fro - 1, dtype=tf.float32)*(-1)

    return temp_fro2


