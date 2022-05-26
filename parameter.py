
class Parameters:
    def __init__(self,
                 mu = 0,
                 compressor_type = None,
                 w = None,
                 s = None,
                 topology = None,
                 agent_num = None,
                 edge_num = None,
                 feedback_type = None,
                 label=None,
                 D=None,
                 L=None,
                 C=None,
                 R=None,
                 r=None,
                 batch_size = 1,
                 distribution_type = None,
                 tune = False,
                 gamma = None,
                 eps = None,
                 nu = 1,
                 c = None,
                 kwargs = None
                 ):

        assert mu >= 0
        self.mu = mu

        assert compressor_type in ['no','rand','top','gossip','quant']
        if compressor_type == 'quant':
            assert s
        else:
            if compressor_type != 'no':
                assert w
        self.compressor_type = compressor_type
        self.s = s
        self.w = w

        assert topology in ['ring','random','full']
        assert agent_num > 0
        if topology == 'random':
            assert edge_num > 0
            assert isinstance(edge_num ,int)
            
        self.topology = topology
        self.agent_num = int(agent_num)
        self.edge_num = edge_num
        self.delta = None
        self.beta = None

        assert feedback_type in ['full','one-bandit','two-bandit']
        if kwargs ==None:
            if feedback_type == 'full':
                assert D>0
                assert L>0
                self.kwargs = {'c': '#1772b2', 'ls' : '-' , 'linewidth': 5}
            elif feedback_type == 'two-bandit':
                assert L>0
                assert R>0
                assert r>0
                if not D:
                    D = 2 * R
                self.kwargs = {'c': '#249c24', 'ls': '--', 'linewidth': 5}
            elif feedback_type == 'one-bandit':
                assert R>0
                assert r>0
                assert C>0
                if not D:
                    D = 2 * R
                self.kwargs = {'c': '#ff7f0e', 'ls': ':', 'linewidth': 5}
        else: self.kwargs = kwargs

        assert batch_size > 0
        assert isinstance(batch_size,int)
        self.batch_size = batch_size

        self.feedback_type = feedback_type
        self.D = D
        self.L = L
        self.C = C
        self.R = R
        self.r = r

        assert distribution_type in ['no','random','label']
        self.distribution_type = distribution_type

        self.tune = tune
        self.gamma = gamma
        self.eps = eps
        self.c = c
        self.nu = nu
        if label!= None:
            self.kwargs['label'] = label

        self.epoch = None

