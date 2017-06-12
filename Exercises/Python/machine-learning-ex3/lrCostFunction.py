from scipy.io import loadmat
from scipy import linalg, sparse #for linalg
import matplotlib.pyplot as plt #for the graph
import numpy as np #for lin alg operations
from scipy.special import expit #vectorized sigmoid function



datafile = 'ex3data1.mat'
mat = loadmat( datafile )
y =mat['y']

X=np.mat([[1, .1, .6, 1.1], [1, .2,.7,1.2], [1, .3, .8, 1.3], [1,.4,.9,1.4], [1, .5, 1, 1.5]])
theta_t=np.mat([[-2], [-1], [1], [2]])

m = y.shape[0]

J=0
theta = np.asmatrix(np.zeros((theta_t.shape[0], 1))); #column of 0s

def sigmoid(z):  
    return 1 / (1 + np.exp(-z));

def gradientFunc(theta_matrix, X_matrix):
    return expit(np.dot(X_matrix,theta_matrix))

def costFunc(theta_matrix ,X_matrix,y_matrix,mylambda = 0.):
    m = X_matrix.shape[0]
    gradient = gradientFunc(theta_matrix,X_matrix) 
    term1 = np.log(gradient).dot(-y_matrix.T) 
    term2 = np.log(1.0-gradient).dot( 1 - y_matrix.T ) 
    left_hand = (term1 - term2) / m 
    right_hand = theta_matrix.T.dot( theta_matrix ) * mylambda / (2*m) 
    return left_hand + right_hand 

cost=costFunc(theta, X, y);
grad=gradientFunc(theta, X);

print("Cost at initial theta (zeros): " +  str(cost));
print("Expected cost: 2.534819\n");
print("Gradient at initial theta (zeros): \n");
print(grad);
print("Expected gradients (approx): 0.146561\n -0.548558\n 0.724722\n 1.398003\n");




