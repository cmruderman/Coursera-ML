import scipy.optimize as opt 
from scipy import linalg, sparse, optimize #for linalg
import matplotlib.pyplot as plt #for the graph
import numpy #for lin alg operations
import pandas as pd #for reading in file



data = pd.read_csv('ex2data1.txt', header=None, delimiter=','); #data is a 97x2 matrix

data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data.shape[1];  
X = data.iloc[:,0:cols-1];  
y = data.iloc[:,cols-1:cols];