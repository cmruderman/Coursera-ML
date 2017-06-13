from __future__ import division
from scipy.io import loadmat
from scipy import linalg, sparse #for linalg
import matplotlib.pyplot as plt #for the graph
import numpy as np #for lin alg operations
from scipy.special import expit #vectorized sigmoid function



datafile = 'ex3data1.mat'
mat = loadmat( datafile )
y=np.mat([[1], [0], [1], [0], [1]])


X=np.mat([[1, .1, .6, 1.1], [1, .2,.7,1.2], [1, .3, .8, 1.3], [1,.4,.9,1.4], [1, .5, 1, 1.5]])
theta_t=np.mat([[-2], [-1], [1], [2]])

m = y.shape[0]

J=0


def sigmoid(z):  
    return 1 / (1 + np.exp(-z));

argument=(np.dot(theta_t.T, X.T)).T
g = sigmoid(argument)

def gradientFunc(theta, X, y, mylambda):
	temp=theta
	temp[0]=0
	grad=(1/m)*(np.dot(X.T,g-y))+(mylambda/m)*temp
	return grad

def costFunc(theta , X, y, mylambda):
    m = X.shape[0]
    summation = sum(np.dot((-1*y).T,np.log(g))-np.dot((1-y).T, np.log(1-g)))
    summation=np.unwrap(summation)[0][0]
    third_term=sum(np.square(theta[1:,:]))
    J=((1./m) * summation)+(mylambda/(2*m))*np.unwrap(third_term)[0][0]
    #(mylambda/(2*m)) is not working
    return J

cost=costFunc(theta_t, X, y, 3);
grad=gradientFunc(theta_t, X, y, 3);

print("Cost at initial theta (zeros): " +  str(cost));
print("Expected cost: 2.534819\n");
print("Gradient at initial theta (zeros): \n");
print(grad);
print("Expected gradients (approx): 0.146561\n -0.548558\n 0.724722\n 1.398003\n");




