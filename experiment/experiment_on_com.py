import sys
sys.path.append('..')
from experiment.param_on_com import par_list_com
from ol_com_log import OnlineCompressedLogistic
import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42

A = np.load('../data/A.npy')
y = np.load('../data/y.npy')
Figsize=(5,5)
Linewidth=5.0
LabelFontdict=14
AxisFontdict=14

res = []

for p in par_list_com:
    
    m = OnlineCompressedLogistic(p)
    m.optimize(A,y,rounds=10)
  
    time = np.arange(1,m.avg_m_regret.shape[0]+1)
    s_regret = m.avg_m_regret/time
    if p.mu>0:
        off = np.load("../data/strong_convex.npy")
    else:
        off = np.load("../data/convex.npy")
    np.save('../result/SC_{}_SR.npy'.format(p.kwargs['label']), s_regret)
    np.save('../result/SC_{}_Bits.npy'.format(p.kwargs['label']), m.avg_trans_bit)

    plt.figure(1, figsize=Figsize)
    plt.semilogy(s_regret - off, **p.kwargs)
    plt.legend(loc=1, prop={'size': LabelFontdict})
    plt.xlabel('Time horizon', fontdict={'size': AxisFontdict})
    plt.ylabel('SR(T)', fontdict={'size': AxisFontdict})
    plt.xticks(size=AxisFontdict)
    plt.yticks(size=AxisFontdict)
    plt.gcf().set_facecolor(np.ones(3))
    plt.grid(True)
    plt.subplots_adjust(left=0.15)
    plt.savefig('../figures/SC_CompType_Time.pdf')

    plt.figure(2, figsize=Figsize)
    plt.loglog(m.avg_trans_bit, s_regret - off, **p.kwargs)
    plt.legend(loc=1, prop={'size': LabelFontdict})
    plt.xlabel('Transmitted bits', fontdict={'size': AxisFontdict})
    plt.ylabel('SR(T)', fontdict={'size': AxisFontdict})
    plt.xticks(size=AxisFontdict)
    plt.yticks(size=AxisFontdict)
    plt.gcf().set_facecolor(np.ones(3))
    plt.grid(True)
    plt.subplots_adjust(left=0.15)
    plt.savefig('../figures/SC_CompType_Bits.pdf')
plt.show()