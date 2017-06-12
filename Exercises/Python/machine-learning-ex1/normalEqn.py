from scipy import linalg, sparse #for linalg
import matplotlib.pyplot as plt #for the graph
import numpy as np #for lin alg operations
import pandas as pd #for reading in file

data = np.asmatrix(pd.read_csv('ex1data2.txt', header=None, delimiter=',')); #data is a 97x2 matrix
X = data[:,[0,1]]
y = data[:, 2];
m=data.shape[0]; #gets the number of training ex
ones = np.ones((m, 1)); #column of 1s
X=np.hstack((ones,X)); #concatenate 1s to represent X0

def normalEqn(matrix_A, matrix_B):
	A_transpose=matrix_A.T;
	temp = np.linalg.pinv(np.dot(matrix_A, A_transpose));
	theta = np.dot(np.dot(A_transpose, temp), matrix_B);
	np.set_printoptions(precision=2)
	return theta;

theta = normalEqn(X, y);

print('Theta computed from the normal equations:');
print(theta);

price = np.dot(np.mat([1, 1650, 3]), theta);

print("Predicted price " + str(price));