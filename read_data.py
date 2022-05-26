
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

data = pd.read_csv('./data/diabetes.csv')
data = data.to_numpy()

A = np.copy(data[:,1:])
scaler = StandardScaler()
scaler.fit(A)
nor_A = np.copy(scaler.transform(A))  
y = np.copy(data[:,0])
y[ y<=0.5 ] = -1

np.save('./data/A.npy',nor_A)
np.save('./data/y.npy',y)