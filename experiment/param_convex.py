import sys
sys.path.append('..')
from parameter import Parameters
import numpy as np
BATCH_SIZE = 10
STRONG_CONV = 0
AGENT_NUM = 9
par_list_convex =[]

'''
p_no_com = Parameters( mu = STRONG_CONV , 
                 compressor_type = 'no',
                 topology = 'random',
                 agent_num = AGENT_NUM,
                 edge_num = 18,
                 feedback_type= 'full',
                 D = 10,
                 L = 10,
                 distribution_type= 'label',
                 tune = True,
                 batch_size= BATCH_SIZE,
                 gamma = 1
       )
par_list_convex.append(p_no_com)
'''

p_full = Parameters( mu = STRONG_CONV, 
                compressor_type = 'quant',
                s = 2,
                topology = 'random',
                agent_num = AGENT_NUM,
                edge_num = 18,
                feedback_type= 'full',
                label='DC-DOGD',
                D = 10,
                L = 10,
                distribution_type= 'label',
                batch_size= BATCH_SIZE,
                tune= True,
                gamma = 0.26,
                nu = 0.1
        )
par_list_convex.append(p_full)

p_one = Parameters( mu = STRONG_CONV, 
                compressor_type = 'quant',
                s = 2,
                topology = 'random',
                agent_num = AGENT_NUM,
                edge_num = 18,
                feedback_type= 'one-bandit',
                label='DC-DOBD',
                L = 10,
                C = 10,
                R = 10,
                r = 9,
                distribution_type= 'label',
                batch_size= BATCH_SIZE,
                tune= True,
                gamma = 0.26,
                eps = 0.5,
                nu = 0.01
        )
par_list_convex.append(p_one)

p_two = Parameters( mu = STRONG_CONV, 
                compressor_type = 'quant',
                s = 2,
                topology = 'random',
                agent_num = AGENT_NUM,
                edge_num = 18,
                feedback_type= 'two-bandit',
                label='DC-DO2BD',
                L = 10,
                R = 10,
                r = 9,
                distribution_type= 'label',
                batch_size= BATCH_SIZE,
                tune= True,
                gamma = 0.26,
                eps=0.05,
                nu = 0.1
        )
par_list_convex.append(p_two)

p_ams = Parameters( mu = STRONG_CONV ,
                compressor_type = 'quant',
                s = 2,
                topology = 'random',
                agent_num = 9,
                edge_num = 18,
                feedback_type= 'full',
                D = 10,
                L = 10,
                distribution_type= 'label',
                tune = True,
                batch_size= BATCH_SIZE,
                nu = 1
    )