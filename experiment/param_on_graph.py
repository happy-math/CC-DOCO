import sys
from parameter import Parameters
import numpy as np
BATCH_SIZE = 10
STRONG_CONV = 1
#AGENT_NUM = 9
meta_par_list_graph =[]
gamma = {}
for agent_num in [9]:
    par_list_graph = []
    for (type,label,kwargs) in [('ring','Ring',{'c': '#d62425', 'ls': '-.', 'linewidth': 5}),('random','G(N,2N)',{'c': '#249c24', 'ls': '--', 'linewidth': 5}),('full','Full',{'c': '#1772b2', 'ls' : '-' , 'linewidth': 5})]:
        p = Parameters( mu = STRONG_CONV, 
                 compressor_type = 'top',
                 w = 0.05,
                 topology = type,
                 agent_num = agent_num,
                 edge_num = 2 * agent_num,
                 feedback_type= 'full',
                 label=label,
                 D = 10,
                 L = 10,
                 distribution_type= 'label',
                 batch_size= BATCH_SIZE,
                 tune= True,
                 gamma = 0.09,
                 kwargs=kwargs
        )
        par_list_graph.append(p)
    meta_par_list_graph.append(par_list_graph)

