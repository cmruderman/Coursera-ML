from scipy import linalg, sparse
import numpy as np



# data = np.loadtxt("ex1data1.txt", skiprows=0);

def eye(dimension):
	return np.mat(np.identity(5));

np.savetxt('warmUpExercise.txt', eye(5), delimiter=' ')