from scipy.io import loadmat
from scipy import linalg, sparse #for linalg
import matplotlib.pyplot as plt #for the graph
import numpy as np #for lin alg operations

mat = loadmat('ex3weights.mat')
data = loadmat('ex3data1.mat')

theta1 = mat['Theta1']
theta2= mat['Theta2']

y = data['y']
X = np.hstack((np.ones((data['X'].shape[0],1)), data['X']));

def sigmoid(z):  
    return 1 / (1 + np.exp(-z));

def predict(theta_1, theta_2, X_matrix):
    z2 = theta_1.dot(X_matrix.T)
    a2 = np.hstack((np.ones((data['X'].shape[0],1)), sigmoid(z2).T))
    z3 = a2.dot(theta_2.T)
    a3 = sigmoid(z3)
    return(np.argmax(a3, 1)+1) 
    #add one because python uses zero based indexing

pred = predict(theta1, theta2, X)
accuracy=format(np.mean(pred == y.ravel())*100);
print('Training Set Accuracy: '+  str(accuracy) +'%');