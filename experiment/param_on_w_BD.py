import sys
from parameter import Parameters
import numpy as np
BATCH_SIZE = 10
STRONG_CONV = 0
AGENT_NUM = 9
par_list_w =[]


for (w,gamma,kwargs) in [(0.05,0.09,{'c': '#ff7f0e', 'ls': ':', 'linewidth': 5}),(0.1,0.1,{'c': '#d62425', 'ls': '-.', 'linewidth': 5}),(0.5,0.32,{'c': '#249c24', 'ls': '--', 'linewidth': 5}),(1,1,{'c': '#1772b2', 'ls' : '-' , 'linewidth': 5})]:
    p = Parameters( mu = STRONG_CONV, 
                 compressor_type = 'top',
                 w = w,
                 topology = 'random',
                 agent_num = AGENT_NUM,
                 edge_num = 18,
                 feedback_type= 'one-bandit',
                 label='w={}'.format(w),
                 L = 10,
                 R=10,
                 r=9,
                 C=10,
                 distribution_type= 'label',
                 batch_size= BATCH_SIZE,
                 tune= True,
                 gamma = gamma,
                 kwargs = kwargs,
                 nu=0.01,
                 eps=1
        )
    par_list_w.append(p)
