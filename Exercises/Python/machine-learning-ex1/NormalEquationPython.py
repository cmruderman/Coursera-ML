from scipy import linalg, sparse
import numpy as np


X = np.mat([[1,2104,5,1,45], [1,1416,3,2,40], [1,1534,3,2,30], [1,852,2,1,36]])

Y=np.mat([[460], [232], [315], [178]])

def thetaFunc(matrix_A, matrix_B):
	A_transpose=matrix_A.T
	temp = np.dot(matrix_A, A_transpose).I
	theta = np.dot(np.dot(A_transpose, temp), matrix_B)
	np.set_printoptions(precision=2)
	return theta;

# X_transpose=X.T

# A = np.dot(X, X_transpose).I

# theta = np.dot(np.dot(X_transpose, A), Y)

#np.savetxt('test.txt', thetaFunc(X, Y), delimiter=',')
print(thetaFunc(X, Y));
