import sys
sys.path.append('..')

from experiment.param_on_number_sc import meta_par_list_graph
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

for meta in meta_par_list_graph:
    i = 0
    res = np.zeros((2,len(meta)))
    for p in meta:
        m = OnlineCompressedLogistic(p)
        m.optimize(A,y,rounds=10)
        max_regret =  m.avg_m_regret
        off = np.load("../data/strong_convex.npy")
        res[0,i] = p.agent_num
        res[1,i] = (max_regret[-1] / (p.epoch)  - (off[-1] * 786 / p.epoch) ) / p.agent_num
        i+=1
        np.save('../result/SC_{}_res.npy'.format(p.kwargs['label']), res)
    plt.figure(1, figsize=Figsize)
    plt.plot(res[0],res[1], **p.kwargs)
    plt.legend(loc=1, prop={'size': LabelFontdict})

plt.xlabel('Node number', fontdict={'size': AxisFontdict})
plt.ylabel('AR(T)', fontdict={'size': AxisFontdict})
plt.xticks(size=AxisFontdict)
plt.yticks(size=AxisFontdict)
plt.ylim((0,5))
plt.gcf().set_facecolor(np.ones(3))
plt.grid(True)
plt.subplots_adjust(left=0.15)
plt.savefig('../figures/SC_Number_Time.pdf')
plt.show()
    

