from __future__ import division
import scipy.io #Used to load the OCTAVE *.mat files
from scipy import linalg, sparse #for linalg
import matplotlib.pyplot as plt #for the graph
import numpy as np #for lin alg operations
from scipy.special import expit #vectorized sigmoid function
import itertools

data_file = 'ex5data1.mat'
mat = scipy.io.loadmat(data_file)
X, Xtest, Xval = mat['X'], mat['Xtest'], mat['Xval']
y, ytest, yval = mat['y'], mat['ytest'], mat['yval']
theta=np.matrix([[1],[1]])
m=X.shape[0]

p=8;

def featureNormalize(matrix):
	mu=np.mean(matrix, axis=0)
	newMat=np.subtract(matrix,mu)
	std=np.std(newMat,ddof=1, axis=0) #ddof=1 makes sure std hat is unbiased estimator
	matrix=np.divide(newMat,std)
	return matrix;

def polyFeatures(X,p):  
	X_poly=np.mat(np.zeros((X.shape[0], p)))
	for i in range(p):
		X_poly[:,i]=np.power(X,i)
	return X_poly

X_poly = polyFeatures(X, p);
X_poly=featureNormalize(X_poly);
X_poly= np.hstack((np.ones((m,1)), X_poly))
print(X_poly)

