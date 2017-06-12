from scipy import linalg, sparse #for linalg
import matplotlib.pyplot as plt #for the graph
import numpy #for lin alg operations
import pandas as pd #for reading in file


def gradientDescent(X, y, theta, alpha, num_iters):
	for i in range(0, num_iters):
		hypothesis=numpy.dot(X, theta);
		error=numpy.subtract(hypothesis,y);
		delta = (1./m)*numpy.dot(X.T, error);
		theta = theta - alpha*delta;
	return theta


data = pd.read_csv('ex1data1.txt', header=None, delimiter=','); #data is a 97x2 matrix
t = numpy.zeros((2, 1)); #fitting parameters
m=data.shape[0]; #gets the number of training ex
X=numpy.asmatrix(data.values[:,0]).T; #zero indxed columns
y=numpy.asmatrix(data.values[:,1]).T; #real values
ones = numpy.ones((m, 1)); #column of 1s
X=numpy.hstack((ones,X)); #concatenate 1s to represent X0

theta = gradientDescent(X, y, t, .01, 1500)

plt.title("Plotting");
plt.ylabel("Profit in $10,000s");
plt.xlabel("Population of City in 10000s");
plt.plot(data.values[:,0], data.values[:,1], 'go');
plt.axis([4, 25, -5, 25]);
plt.plot(X[:,1],numpy.dot(X, theta));

plt.show();

print("Theta found by gradient descent:\n");
print(str(theta.item(0))+ " "+ str(theta.item(1)));
print("Expected theta values (approx)\n");
print("-3.6303\n 1.1664\n\n");


