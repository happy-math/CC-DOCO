import sys
from parameter import Parameters
import numpy as np
BATCH_SIZE = 10
STRONG_CONV = 1
AGENT_NUM = 9
eps ={}
eps['full'] = None
eps['one-bandit'] = 0.5
eps['two-bandit'] = 0.01
nu = {}
nu['full'] = 1
nu['one-bandit'] = 0.05
nu['two-bandit'] = 0.5

gamma = {}
meta_par_list_graph = []
for (type,label,kwargs) in [('full','DC-DOGD',{'c': '#1772b2', 'ls' : '-' , 'linewidth': 5}),('one-bandit','DC-DOBD',{'c': '#d62425', 'ls': '-.', 'linewidth': 5}),('two-bandit','DC-DO2BD',{'c': '#249c24', 'ls': '--', 'linewidth': 5})]:
    par_list_graph =[]
    for agent_num in np.arange(10,55,5):
        p = Parameters( mu = STRONG_CONV, 
                    compressor_type = 'top',
                    w = 0.1,
                    topology = 'full',
                    agent_num = agent_num,
                    edge_num = int( 0.5 * (agent_num)*(agent_num-1) ),
                    feedback_type= type,
                    label=label,
                    D = 10,
                    L = 10,
                    R = 10,
                    r = 9,
                    C = 10,
                    distribution_type= 'label',
                    batch_size= BATCH_SIZE,
                    tune= True,
                    gamma = 0.1,
                    eps = eps[type],
                    nu=nu[type],
                    kwargs=kwargs
            )
        par_list_graph.append(p)
    meta_par_list_graph.append(par_list_graph)

