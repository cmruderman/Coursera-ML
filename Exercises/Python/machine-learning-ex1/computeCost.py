from scipy import linalg, sparse #for linalg
import matplotlib.pyplot as plt #for the graph
import numpy #for lin alg operations
import pandas as pd #for reading in file


def computeCost(matrix, theta_value):
	m=matrix.shape[0]; #gets the number of training ex
	X=numpy.asmatrix(matrix.values[:,0]).T; #zero indxed columns
	y=numpy.asmatrix(matrix.values[:,1]).T; #real values
	ones = numpy.ones((m, 1)); #column of 1s
	X=numpy.hstack((ones,X)); #concatenate 1s to represent X0
	predictions=numpy.dot(X, theta_value);
	squaredError=numpy.square((predictions-y));
	sumValue = squaredError.sum(axis=0).item(0); #sum() returns a matrix, so get the item and store it
	return (1./(2*m))*sumValue;


data = pd.read_csv('ex1data1.txt', header=None, delimiter=','); #data is a 97x2 matrix
theta = numpy.zeros((2, 1)); #fitting parameters
print(computeCost(data, theta));
print("Expected cost value (approx) 32.07\n");

newTheta=numpy.mat([[-1], [2]]);
print(computeCost(data, newTheta));
print("Expected cost value (approx) 54.24\n");
