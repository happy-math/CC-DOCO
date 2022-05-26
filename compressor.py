MBIT = 64

from cmath import sqrt
import numpy as np
class Compressor:
    def __init__(self,w):
        self.w = w
        self.total_bit = 0

class RandSparse(Compressor):
    def compress(self , vec , times):
        d = vec.shape[0]
        k = int( self.w * d )
        ind = np.random.randint(0,d,k)

        a = np.zeros(d)
        a[ind] = 1

        self.total_bit += times * (MBIT * k + np.log2(d))        
        return vec*a

class TopSparse(Compressor):
    def compress(self , vec , times):
        d = vec.shape[0]
        k = int( self.w * d )
        ind = np.argpartition(np.abs(vec), -k)[-k:]


        a = np.zeros(d)
        a[ind] = 1

        self.total_bit += times * (MBIT * k + np.log2(d))        
        return vec*a

class RandGossip(Compressor):
    def compress(self,vec, times):
        d = vec.shape[0]
        p = np.random.random()
        if p <= self.w:
            self.total_bit += times * MBIT * d
            return vec
        else:
            a = np.zeros(d)
            return a

class RandQuant():   
    def __init__(self,s) -> None:
        self.s = s
        self.total_bit = 0

    def update_w(self,d):
        s = self.s
        sigma = 1 + min(d/s ** 2 , d ** 0.5/s)
        self.w = 1 / sigma

    def compress(self,vec, times):
        d = vec.shape[0]
        s = self.s
        sigma = 1 + min(d/s ** 2 , d ** 0.5/s)
        sgn = np.sign(vec)
        norm = np.linalg.norm(vec)
        if norm < 1e-15:
            return np.zeros(d)
        nor_vec = np.abs(vec) / norm
        xi = np.random.random(d)
        quant_vec = np.trunc( (nor_vec * self.s) + xi ) 
        self.total_bit += times * (MBIT + np.ceil(np.log2(self.s * 2 + 1)) * d )
        return norm * sgn * quant_vec /(self.s * sigma)