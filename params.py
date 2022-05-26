from parameter import Parameters
import numpy as np
BATCH_SIZE = 10
STRONG_CONV = 1
AGENT_NUM = 9
#meta_par_list=[]
b = np.array([0.5])#eps
a = np.array([0.04,0.05,0.06,0.07])#nu#b
#par_eps = a + 10
#par_gamma = b
par_list = []

for eps in b:
    for nu in a:
        p = Parameters( mu = STRONG_CONV, 
            compressor_type = 'quant',
            s = 2,
            topology = 'random',
            agent_num = AGENT_NUM,
            edge_num = 18,
            feedback_type= 'one-bandit',
            label='DC-DOBD',
            D=10,
            R = 10,
            r = 9,
            C = 10,
            L = 10,
            distribution_type= 'label',
            batch_size= BATCH_SIZE,
            tune= True,
            gamma = 0.26,
            nu = nu,
            eps = eps
            )

        par_list.append(p)

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
p = Parameters( mu = STRONG_CONV , 
                compressor_type = 'quant',
                w = 1,
                s = 2,
                topology = 'random',
                agent_num = 9,
                edge_num = 18,
                feedback_type= 'full',
                D = 10,
                R = 200,
                C = 10,
                r = 2,
                L = 10,
                distribution_type= 'label',
                tune = True,
                batch_size= BATCH_SIZE,
                gamma = 1
    )
#par_list.append(p)
