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

plt.plot(X, y,  'g^', label='Admitted')
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()

def linearRegCostFunction(theta, X, y,lambda_value): 
	m=y.shape[0]
	J=0
	grad = np.mat(np.zeros((theta.shape[0], theta.shape[1])))
	h=np.dot(X,theta)
	subtract_value=np.subtract(h,y)
	g=np.square((np.subtract(h,y)))
	J=((1/(2*m))*np.array(sum(g))[0][0])+(lambda_value/(2*m))*sum(np.square(theta[1:theta.shape[0],:]))
	J=np.array(J)[0][0]
	temp=theta
	temp[0]=0
	grad=(1/m)*(np.dot(X.T, np.subtract(h, y)))+(lambda_value/m)*temp
	ret=dict()
	ret['Cost']=J
	ret['Gradient']=grad;
	return ret

lst = linearRegCostFunction(theta, np.hstack((np.ones((m,1)), X)), y, 1)
J=lst['Cost']
grad=lst['Gradient']

print('Cost at theta=[1;1]: ' + str(J) + '\n(this value should be about 303.993192)')


