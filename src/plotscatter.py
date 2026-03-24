import numpy as np
import pandas as pd
import heapq
import matplotlib.pyplot as plt
N=256
SNR_db=20
epoch=0
# y=np.loadtxt('D:/workspace/cuijingwen/stage3_2/reliabilityyuan2/64/'+str(N)+'s'+str(SNR_db)+'e'+str(epoch)+'step150.txt')
y=np.loadtxt('D:/workspace/cuijingwen/stage3_2/newnormwayforscatter/away/3_bar.txt')
print(np.mean(y))
#
xva = np.linspace(0, N - 1, N)
plt.scatter(xva, y, s=30, label='probability')
plt.xlabel('bit index')
plt.ylabel('Reliability')
plt.gca().set_xlim(0, N)

plt.grid(True, which="both")
plt.show()

######## 查看信息比特所在区间
# 原 PW
weight = np.loadtxt('D:/workspace/cuijingwen/stage3_2/PW/weight'+str(N)+'.txt',dtype=float)
y2=weight[::-1]
# score = pd.Series(y2)    # # score=heapq.nlargest(int(N/2),y2)

# NN
score = pd.Series(y)
score2 =score.sort_values(ascending=False)
rr=[-1,int(N/4)-1,int(N/2)-1,int(3*N/4)-1,N-1]
se1 = pd.cut(score2[0:int(N/2)].index, bins=rr)
print(se1.value_counts())


# 废
# score=heapq.nlargest(int(3*N/4), range(len(y)), y.__getitem__)
# ran=[-1,int(N/2)-1] # 远离      靠近[int(3*N/4),N-1]
# score = pd.Series(score)
# se1 = pd.cut(score, ran) # 统计0-1,1-2依次类推各个区间的数值数量
# print(se1.value_counts())
