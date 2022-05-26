import sys
sys.path.append('..')
from experiment.param_convex import par_list_convex,p_ams
from ol_com_log import OnlineCompressedLogistic
from baseline import AdaOnlineCompressedLogistic
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

for p in par_list_convex  :
    
    m = OnlineCompressedLogistic(p)
    m.optimize(A,y,rounds=10)
    max_regret =  m.avg_m_regret
    time = np.arange(1,max_regret.shape[0]+1)
    s_regret = max_regret/time
    off = np.load("../data/convex.npy")

    plt.figure(1, figsize=Figsize)
    plt.plot(s_regret - off, **p.kwargs)
    plt.legend(loc=1, prop={'size': LabelFontdict})
    plt.figure(2, figsize=Figsize)
    plt.loglog(m.avg_trans_bit, s_regret - off, **p.kwargs)
    plt.legend(loc=1, prop={'size': LabelFontdict})
    np.save('../result/C_{}_SR.npy'.format(p.kwargs['label']), s_regret)
    np.save('../result/C_{}_Bits.npy'.format(p.kwargs['label']), m.avg_trans_bit)

m = AdaOnlineCompressedLogistic(p_ams)
m.optimize(A,y,rounds=10)
(max_regret, trans_bit_history) = (m.avg_m_regret,m.avg_trans_bit)
time = np.arange(1,max_regret.shape[0]+1)
s_regret = max_regret/time
np.save('../result/C_ECDAMSGrad_SR.npy', s_regret)
np.save('../result/C_ECDAMSGrad_Bits.npy', m.avg_trans_bit)

plt.figure(1,figsize=Figsize)
plt.plot(s_regret-off, c='#d62425', ls = '-.', label ='ECD-AMSGrad',linewidth = Linewidth)
plt.legend(loc=1,prop={'size':LabelFontdict})
plt.xlabel('Time horizon',fontdict={'size':AxisFontdict})
plt.ylabel('SR(T)',fontdict={'size':AxisFontdict})
plt.ylim(0,50)
plt.xticks(size=AxisFontdict)
plt.yticks(size=AxisFontdict)
plt.gcf().set_facecolor(np.ones(3))
plt.grid(True)
#plt.tight_layout()
plt.subplots_adjust(left=0.15)
plt.savefig('../figures/Four_C_Time.pdf')

plt.figure(2,figsize=Figsize)
plt.loglog(trans_bit_history,s_regret-off,c='#d62425', ls = '-.',label ='ECD-AMSGrad',linewidth = Linewidth)
plt.legend(loc=1,prop={'size':LabelFontdict})
plt.xlabel('Transmitted bits',fontdict={'size':AxisFontdict})
plt.ylabel('SR(T)',fontdict={'size':AxisFontdict})
plt.xticks(size=AxisFontdict)
plt.yticks(size=AxisFontdict)
plt.gcf().set_facecolor(np.ones(3))
plt.grid(True)
#plt.tight_layout()
plt.subplots_adjust(left=0.15)
plt.savefig('../figures/Four_C_Bits.pdf')

plt.show()