import sys
sys.path.append('.')
from params import par_list
from ol_com_log import OnlineCompressedLogistic
import numpy as np 
import matplotlib.pyplot as plt

A = np.load('./data/A.npy')
y = np.load('./data/y.npy')
ans = np.inf
desired = (0,0)

for p in par_list:
    print('**************START**********************')
    m = OnlineCompressedLogistic(p)
    m.optimize(A,y)
    (max_regret, trans_bit_history) = (m.avg_m_regret,m.avg_trans_bit)
    time = np.arange(1,max_regret.shape[0]+1)
    s_regret = max_regret/time
    print("loss:",s_regret[-1])
    print('*****************END********************\n')

    if p.mu>0:
        off = np.load("./data/strong_convex.npy")
    else:
        off = np.load("./data/convex.npy")
    if (s_regret[-1]< ans):
        desired = (p.eps,p.nu)
        ans = s_regret[-1]
    plt.subplot(121)
    plt.plot(s_regret-off,label = 'eps={:.4f},b={:.4f}'.format(p.eps,p.nu))
    plt.subplot(122)
    plt.loglog(trans_bit_history,s_regret-off,label = 'eps={:.4f},b={:.4f}'.format(p.eps,p.nu))
    plt.legend(prop={'size':8})
    plt.xlabel('Bits',fontdict={'size':18})
    plt.ylabel('E[Reg(t)] / t',fontdict={'size':18})
    plt.xticks(size=14)
    plt.yticks(size=14)
        #plt.gcf().set_facecolor(np.ones(3))
    plt.grid(True)
print("best choice:{}".format(desired))
plt.show()



