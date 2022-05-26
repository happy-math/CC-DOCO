import sys
from parameter import Parameters
import numpy as np
BATCH_SIZE = 10
STRONG_CONV = 1
AGENT_NUM = 9
par_list_w =[]


for (w,gamma,kwargs) in [(0.05,0.09,{'c': '#ff7f0e', 'ls': ':', 'linewidth': 5}),(0.1,0.1,{'c': '#d62425', 'ls': '-.', 'linewidth': 5}),(0.5,0.32,{'c': '#249c24', 'ls': '--', 'linewidth': 5})]:
    p = Parameters( mu = STRONG_CONV, 
                 compressor_type = 'top',
                 w = w,
                 topology = 'random',
                 agent_num = AGENT_NUM,
                 edge_num = 18,
                 feedback_type= 'full',
                 label='w={}'.format(w),
                 D = 10,
                 L = 10,
                 distribution_type= 'label',
                 batch_size= BATCH_SIZE,
                 tune= True,
                 gamma = gamma,
                 kwargs = kwargs
        )
    par_list_w.append(p)

p_no_com = Parameters( mu = STRONG_CONV ,
                 compressor_type = 'no',
                 topology = 'random',
                 agent_num = AGENT_NUM,
                 edge_num = 18,
                 feedback_type= 'full',
                 label='Exact comm.',
                 D = 10,
                 L = 10,
                 distribution_type= 'label',
                 tune = True,
                 batch_size= BATCH_SIZE,
                 gamma = 1,
                 kwargs = {'c': '#1772b2', 'ls' : '-' , 'linewidth': 5}
    )
par_list_w.append(p_no_com)