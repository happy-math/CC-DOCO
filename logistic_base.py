import numpy as np
from parameter import Parameters
from scipy.special import expit
class LogisticBase:
    def __init__(self, params: Parameters) -> None:
        #x列为一个agent的状态
        self.x = None 
        self.xh = None
        self.s = None
        self.g = None
        self.q = None
        self.loss_history = None
        self.params = params
        self.is_complete = False
    
    def loss(self,x,A,y):
        #A行为一列数据
        arr_logit = np.vectorize(LogisticBase.logit)
        p = self.params
        num_sample = A.shape[0]
        loss = np.sum(arr_logit(-y * (A @ x))) 
        # print(np.exp(-y * (A @ x)))
        if p.mu > 0:
            loss += p.mu * ( num_sample / (p.batch_size) ) /2 * np.linalg.norm(x) ** 2 
        # print(loss)
        assert not np.isnan(loss)
        return loss
    
    def grad(self,x,A,y):
        #A行为一列数据
        arr_sigmoid = np.vectorize(LogisticBase.sigmoid)
        p = self.params
        num_sample = A.shape[0]
        # reshape for broadcast
        grad = - y.reshape(-1,1) * A  * (arr_sigmoid ( -y * (A @ x) ).reshape(-1,1))
        grad =  grad.T
        if grad.ndim == 2: 
            if grad.shape[1] != 1:
                grad = grad.sum(axis = 1)
            grad = grad.flatten()
        if p.mu:
            grad += p.mu * ( num_sample / (p.batch_size) ) * x
        # assert np.linalg.norm(grad)>1e-5
        
        return grad
    
    @staticmethod
    def logit(x):
        if x<= 30:
            return np.log(1+np.exp(x))
        else:
            return x

    def sigmoid(x):
        if  x>= 50:
            return 1
        else:
            return expit(x)

