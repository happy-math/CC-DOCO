from p import p
from ol_com_log import OnlineCompressedLogistic
import numpy as np

A = np.load('./data/A.npy')
y = np.load('./data/y.npy')


m = OnlineCompressedLogistic(p)
m.optimize(A,y)
np.save('./data/order.npy',m.offline_order)


