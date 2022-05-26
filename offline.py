import numpy as np
from sklearn.linear_model import LogisticRegression
from  params import BATCH_SIZE,STRONG_CONV,AGENT_NUM 



def loss(t, x, A, y, mu):
    baseline_loss = ( 1 / t ) * np.sum (np.log( 1 + np.exp( -y * (A @ x) ) ))
    baseline_loss += (mu / 2) * np.linalg.norm(x) ** 2
    return baseline_loss

A = np.load('./data/A.npy')
y = np.load('./data/y.npy')
indice = np.load('./data/order.npy')
batch_size = BATCH_SIZE
n = AGENT_NUM 
num_sample = A.shape[0]

epoch = int (num_sample / (batch_size * n))
num_sample = epoch * batch_size * n
off_loss = np.zeros( epoch )
mu = STRONG_CONV

for t in range(off_loss.shape[0]):
    print(t)
    clf = LogisticRegression(tol=1e-4,
                             penalty='l2',
                             max_iter=1e3, 
                             C = mu / (n * (t+1)) )
    clf.fit(A[indice[: (t+1) * n * batch_size] ], y[ indice[: (t+1) * n * batch_size] ])
    x_off = clf.coef_.T.flatten()
    off_loss[t] = loss( t+1,
                        x_off, 
                        A[ indice[: (t+1) * n * batch_size]],
                        y[ indice[: (t+1) * n * batch_size]],
                        n)

np.save('./data/strong_convex.npy',off_loss)

clf = LogisticRegression(tol=1e-4, penalty='none', max_iter=1e3)
for t in range(off_loss.shape[0]):
    print(t)
    clf.fit(A[indice[ : (t+1) * n * batch_size]], y[indice[: (t+1) * n * batch_size]])
    x_off = clf.coef_.T.flatten()
    off_loss[t] = loss(t+1 ,x_off, A[indice[: (t+1) * n * batch_size]], y[indice[: (t+1) * n * batch_size]], 0)

np.save('./data/convex.npy',off_loss)