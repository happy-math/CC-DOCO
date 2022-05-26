from parameter import Parameters
from logistic_base import LogisticBase
from compressor import *
import networkx as nx
import numpy as np

def choose_on_shpere(d):
    vec = np.random.randn(d) 
    return vec/np.linalg.norm(vec)

class  OnlineCompressedLogistic(LogisticBase):
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
                self.degree = np.ones(n)
            else:
                value = 0.5
                W = np.zeros( (n, n) )
                np.fill_diagonal(W[1:,:], value )
                np.fill_diagonal(W[:, 1:], value )
                W[0, n - 1] = value
                W[n - 1, 0] = value
                self.degree = np.ones(n) * 2
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

    def __calculate_config(self,d):
        par = self.params
        if par.compressor_type == 'quant':
            self.compressor.update_w(d)
        W = self.W
        
        if par.agent_num == 1:
            delta = 1
        else:
            delta = 1 - np.partition( np.abs(np.linalg.eigvalsh(W)), -2)[-2]
        beta = np.linalg.norm(np.eye(par.agent_num)- W,ord=2)
        assert delta > 1e-16 and delta <=1
        assert beta >= 0 and beta <=2
        
        w = self.compressor.w
        par.beta = beta
        par.delta = delta

        if par.gamma == None:
            gamma = 3 * (delta ** 3) * (w ** 2) * (w + 1) / (
                        48 * (delta ** 2 + 18 * delta * (beta ** 2) + 36 * (beta ** 2)) * (beta ** 2) * (w + 2) * (
                            1 - w) + 4 * (delta ** 2) * (beta + 1) * beta * (w + 2) * (1 - w) * w + 6 * (
                                    delta ** 3) * w)
            par.gamma = gamma
        else:
            gamma = par.gamma

        H = 4 * 3 ** 0.5 * (par.agent_num ** 0.5 + 2 * 3 **0.5 / (gamma * delta) + 1 ) * (1 + 1/(gamma * delta ) + 1/ w)
        (R,D,C,T,L,r,mu) = (par.R, par.D, par.C, par.epoch,par.L,par.r,par.mu)

        if mu>0:
            c = 16 / (3 * gamma * delta)
        else:
            c = 8 / (3 * gamma * delta)

        if par.c == None:
            par.c = c
        else:
            c=par.c

        if par.feedback_type == 'full':
            eps = 0
            if par.tune == True:
                eta = 1
            else:
                if mu > 0:
                    eta = 1/mu
                else:
                    eta = D/L
            
        if par.feedback_type == 'one-bandit':
            if par.tune == True:
                eps = par.eps
                eta = 1
            else:
                if mu > 0:
                    eps = ( (H*d ** 2 * C **2 * np.log(T+c) )/( (L+C/r) *mu * T ) )**(1/3)
                    eta = 1/mu
                else:
                    eps = ((1+4*H) *R*d*C / (2 * ( L + C/r)))**0.5 * (T+c) ** 0.25 / T ** 0.5
                    eta = (2 * R * eps) / (d * C)
                   
        if par.feedback_type == 'two-bandit':
            if par.tune == True:
                eps = par.eps
                eta = 1
            else:
                if mu > 0:
                    eps = np.log(T) / T
                    eta = 1/mu
                else:
                    eps = 1  / T** 0.5
                    eta = (2 * R) / (d * L)

        par.eps = eps
        par.eta =  eta * par.nu #b

    def stepsize(self,t):
        p = self.params
        if p.mu > 0:
            return p.eta / (t+p.c)
        else:
            return p.eta / (t+p.c) ** 0.5

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
        num , d = A.shape

        p.epoch = int (num / (bs * n))
        num_sample = p.epoch * bs * n
        T = p.epoch

        A = A[:,:num_sample]
        y = y[:num_sample]

        self.__calculate_config(d)

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

        #initialize
        self.x = np.zeros((d,n))
        self.xh = np.zeros((d,n))
        self.s = np.zeros((d,n))
        self.q = np.zeros((d,n))
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
                
                # projection
                if p.feedback_type == 'full':
                    self.x[:,i] = self.proj( self.x[:,i] , p.D/2 )
                else:
                    self.x[:,i] = self.proj( self.x[:,i] , (1 - p.eps / p.r) )
                #loss
                x_i = self.x[:,i]
                xh_i = self.xh[:,i]
                
                #compression
                self.q[:,i] = self.compressor.compress(x_i - xh_i,self.degree[i])

                #gradient
                (x_i_now , self.g[:,i]) = self.grad_oracle(x_i,a_i,y_i)
                if p.feedback_type == 'two-bandit':
                    x_1 = x_i_now[0]
                    x_2 = x_i_now[1]
                    loss_history[t,i] = ( self.loss(x_1,a_t,y_t) + self.loss(x_2,a_t,y_t) ) / 2
                else:
                    loss_history[t,i] = self.loss(x_i_now,a_t,y_t)

            #update
            self.xh += self.q
            self.s += self.q @ self.W
            eta_t = self.stepsize(t+1)
            self.x += ( p.gamma * (self.s-self.xh) - eta_t * self.g )
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