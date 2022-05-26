
from matplotlib.pyplot import pause
from parameter import Parameters
from logistic_base import LogisticBase
from compressor import *
import networkx as nx
import numpy as np

def choose_on_shpere(d):
    vec = np.random.randn(d) 
    return vec/np.linalg.norm(vec)


class  AdaOnlineCompressedLogistic(LogisticBase):
    def __init__(self, params: Parameters, do_prints = False):
        p = params
        super().__init__(p)
        self.compressor = self.__choose_compressor(p.compressor_type,p.w,p.s)
        self.W= self.__create_matrix(p.agent_num, p.edge_num, p.topology)
        self.do_prints = do_prints
        
    def __choose_compressor(self,type,w,s):            
        if type == 'no':
            self.params.w = 1
            return RandGossip(1)
        if type == 'rand':
            return RandSparse(w)
        if type == 'top':
            return TopSparse(w)
        if type == 'gossip':
            return RandGossip(w)
        if type == 'quant':
            return RandQuant(s)

    def __create_matrix(self,n,m,topology):
        
        if n==1:
            self.degree = np.ones(n)
            return np.eye(1)
        if topology == 'ring':
            if n == 2:
                W = np.array(( (0,1),(1,0)   ))
                self.degree = np.zeros(n)
            else:
                value = 0.5
                W = np.zeros( (n, n) )
                np.fill_diagonal(W[1:,:], value )
                np.fill_diagonal(W[:, 1:], value )
                W[0, n - 1] = value
                W[n - 1, 0] = value
                self.degree = np.zeros(n) * 2
        elif topology == 'random':
            G = nx.gnm_random_graph(n,m,seed=42)
            assert nx.is_connected(G)
            W = nx.to_numpy_array(G)
            self.degree = np.zeros(n)
            for i in nx.nodes(G):
                sum = 0
                self.degree[i] = nx.degree(G,i)
                for j in iter(nx.neighbors(G,i)):
                    weight = 1/( max(nx.degree(G,i) , nx.degree(G,j) )  )
                    W[i][j] = weight
                    sum += weight
                W[i][i] = 1 - sum
        elif topology == 'full':
            W = np.ones((n,n)) / n
            self.degree = np.ones(n) * n
        return W
                    
        
    
    def stepsize(self,t):
        p = self.params
        return p.eta / (t) ** 0.5

    def grad_oracle(self,x,A,y):
        p = self.params
        d = A.shape[1]
        if p.feedback_type == 'full':
            g = self.grad(x,A,y)
            x_now = x
        if p.feedback_type == 'one-bandit':
            u = choose_on_shpere(d)  
            g = d / p.eps * self.loss(x+p.eps * u,A,y) * u
            x_now = x+p.eps * u
        if p.feedback_type == 'two-bandit':
            u = choose_on_shpere(d)
            g =  d / (2 * p.eps) * ( self.loss(x+ p.eps * u,A,y) - self.loss(x- p.eps *u,A,y)  ) * u
            x_now =  [x+p.eps * u, x-p.eps * u]
        assert not np.isnan(g).any()
        return x_now,g

    def print_params(self):
        p = self.params
        print('optimization start.')
        print('Graph topology:{}'.format(p.topology))
        print('agent number: {},\nlearning rounds:{}'.format(p.agent_num,p.epoch) )
        print("------------Parameters-------------")
        print(" |gamma   |{:>20.5f}|\n |c       |{:>20.5f}|\n |eps     |{:>20.5f}|\n |eta     |{:>20.5f}|\n |delta   |{:>20.5f}|\n |beta    |{:>20.5f}|".format(p.gamma,p.c,p.eps,p.eta,p.delta,p.beta))
        print("-----------------------------------")

    def optimize(self,A,y,rounds=1):
        p = self.params
        bs = p.batch_size
        n = p.agent_num
        p.eta = p.nu
        num , d = A.shape

        if p.compressor_type == 'quant':
            self.compressor.update_w(d)

        p.epoch = int (num / (bs * n))
        num_sample = p.epoch * bs * n
        T = p.epoch

        A = A[:,:num_sample]
        y = y[:num_sample]

        index = np.arange(num)
        if p.distribution_type == 'random':
            np.random.seed(42)
            np.random.shuffle(index)
            np.random.seed()
        if p.distribution_type == 'label':
            index = np.argsort(y)
        indice = np.zeros((p.agent_num,p.epoch,p.batch_size),dtype='int64')
        start = 0
        for i in range(p.agent_num):
            for j in range(p.epoch):
                indice[i,j] = index[start:start+p.batch_size]
                start += p.batch_size
        if self.do_prints:
            self.print_params()
        self.avg_trans_bit = np.zeros(T)
        self.avg_m_regret = np.zeros(T)
        for _ in range(rounds):
            self.compressor.total_bit = int(0)
            (m_regret , bits) = self.train(A,y,indice)
            self.avg_m_regret += m_regret
            self.avg_trans_bit += bits
        self.avg_m_regret /= rounds
        self.avg_trans_bit /= rounds
        self.is_complete = True

    def train(self,A,y,indice):
        # parameters
        p = self.params
        n = p.agent_num
        num , d = A.shape
        T = p.epoch
        beta_1 = 0.9
        beta_2 = 0.999
        #initialize
        self.v = np.zeros((d,n))
        self.vh = np.zeros((d,n))
        self.w_half = np.zeros((d,n))
        self.wh = np.zeros((d,n))
        self.w_t_p_1 = np.zeros((d,n))
        self.w_t = np.ones((d,n))
        self.z = np.zeros((d,n))
        self.m = np.zeros((d,n))
        self.g = np.zeros((d,n))
        loss_history = np.zeros((T,n))
        trans_bit_history = np.zeros(T)
        self.offline_order = []
        for t in range(T):
            index_for_current_epoch = indice[ :, t].flatten()
            self.offline_order += list(index_for_current_epoch)
            a_t = A[  index_for_current_epoch  ]
            y_t = y[  index_for_current_epoch  ]
            for i in range(n):
                
                # take a batch
                a_i = A[ indice[ i , t ] ]
                y_i = y[ indice[ i , t ] ]
                
                #loss
                w_i = self.w_t[:,i]
                loss_history[t,i] = self.loss(w_i,a_t,y_t)
                #gradient
                (w_i_now , g_i) = self.grad_oracle(w_i,a_i,y_i)
                self.g[:,i] = g_i
                #momentum
                self.m[:,i] *= beta_1
                self.m[:,i] += (1-beta_1) * g_i
                self.v[:,i] *= beta_2
                self.v[:,i] += (1-beta_2) * g_i ** 2
                self.vh[:,i] = np.maximum(self.vh[:,i],self.v[:,i])
            #update
            alpha_t = self.stepsize(t+1)
            self.w_half = self.wh @ self.W
            self.w_t_p_1 = self.w_half - alpha_t * (self.m / (np.sqrt( self.vh ) ) )
            self.z = (1-(t+1) / 2) * self.w_t + ((t+1)/2) * self.w_t_p_1
            for i in range(n):
                C_z_i = self.compressor.compress(self.z[:,i],self.degree[i])
                wh_i =  self.wh[:,i] 
                self.wh[:,i] =  (1 - 2 /(t+1)  ) * wh_i  + 2/(t+1) * C_z_i
            self.w_t = self.w_t_p_1
            
            trans_bit_history[t] = self.compressor.total_bit
        
        m_regret = self.max_regret(loss_history)
        return ( m_regret, trans_bit_history)

    def max_regret(self,loss_history):
        # assert self.is_complete
        sum = np.zeros(loss_history.shape)
        sum[0] = loss_history[0]
        for i in range(1,self.params.epoch):
            sum[i] = sum[i-1] + loss_history[i]
        max = np.max(sum,axis=1)
        return max

    def proj(self,x,r):
        if  np.linalg.norm(x) > r:
            return x / np.linalg.norm(x) * r
        else:
            return x



