import sys
from parameter import Parameters
import numpy as np
BATCH_SIZE = 10
STRONG_CONV = 1
AGENT_NUM = 9
par_list_com =[]
    
for (type,gamma,label,kwargs) in [('gossip',0.09,'RGossip_p',{'c': '#d62425', 'ls': '-.', 'linewidth': 5}),('rand',0.09,'Rand_k',{'c': '#ff7f0e', 'ls': ':', 'linewidth': 5}),('top',0.28,'Top_k',{'c': '#249c24', 'ls': '--', 'linewidth': 5}),('quant',0.26,'QSGD_s',{'c': '#1772b2', 'ls' : '-' , 'linewidth': 5})]:
    p = Parameters( mu = STRONG_CONV, 
                 compressor_type = type,
                 w = 0.30383243470068705,
                 s = 2,
                 topology = 'random',
                 agent_num = AGENT_NUM,
                 edge_num = 18,
                 feedback_type= 'full',
                 label=label,
                 D = 10,
                 L = 10,
                 distribution_type= 'label',
                 batch_size= BATCH_SIZE,
                 tune= True,
                 gamma = gamma,
                 kwargs=kwargs
        )
    par_list_com.append(p)

