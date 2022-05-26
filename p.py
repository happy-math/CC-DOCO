from parameter import Parameters
BATCH_SIZE = 10
STRONG_CONV = 1
AGENT_NUM = 9

p = Parameters( mu = STRONG_CONV , 
                 compressor_type = 'no',
                 topology = 'random',
                 agent_num = AGENT_NUM,
                 edge_num = 18,
                 feedback_type= 'full',
                 D = 20,
                 L = 10,
                 distribution_type= 'label',
                 tune = True,
                 batch_size= BATCH_SIZE,
                 gamma = 1
    )