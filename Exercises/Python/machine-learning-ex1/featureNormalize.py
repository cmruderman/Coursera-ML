from scipy import linalg, sparse #for linalg
import matplotlib.pyplot as plt #for the graph
import numpy #for lin alg operations
import pandas as pd #for reading in file


def featureNormalize(matrix):
	mu=numpy.mean(matrix, axis=0);
	newMat=numpy.subtract(matrix,mu);
	std=numpy.std(newMat,ddof=1, axis=0); #ddof=1 makes sure std hat is unbiased estimator
	matrix=numpy.divide(newMat,std);
	return matrix;

def printMatrix(mat1, mat2):
	for r in range(0,10):
		print("\n");
		print "x="+ '['+ str(mat1.item(r,0)), str(mat1.item(r,1)) + "], " + "y=" + str(mat2.item(r));

data = numpy.asmatrix(pd.read_csv('ex1data2.txt', header=None, delimiter=',')); #data is a 97x2 matrix
X = data[:,[0,1]]
y = data[:, 2];

print('First 10 examples from the dataset: \n');
printMatrix(X,y);
print(featureNormalize(X));


