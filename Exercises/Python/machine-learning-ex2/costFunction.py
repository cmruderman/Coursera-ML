import scipy.optimize as opt 
from scipy import linalg, sparse, optimize #for linalg
import matplotlib.pyplot as plt #for the graph
import numpy #for lin alg operations
import pandas as pd #for reading in file



data = pd.read_csv('ex2data1.txt', header=None, delimiter=','); #data is a 97x2 matrix

data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data.shape[1];  
X = data.iloc[:,0:cols-1];  
y = data.iloc[:,cols-1:cols];

# convert to numpy arrays and initalize the parameter array theta
X = numpy.array(X.values);
y = numpy.array(y.values);
theta = numpy.zeros(3);


def sigmoid(z):  
    return 1 / (1 + numpy.exp(-z));

def costFunc(theta_matrix, matrix_X, matrix_Y):  
    theta_matrix = numpy.matrix(theta_matrix);
    matrix_X = numpy.asmatrix(matrix_X);
    matrix_Y = numpy.asmatrix(matrix_Y);
    X = numpy.multiply(-matrix_Y, numpy.log(sigmoid(matrix_X * theta_matrix.T)));
    y = numpy.multiply((1 - matrix_Y), numpy.log(1 - sigmoid(matrix_X * theta_matrix.T)));
    return numpy.sum(X - y) / (len(matrix_X));

def gradientFunc(theta, X, y):  
    theta = numpy.matrix(theta)
    X = numpy.matrix(X)
    y = numpy.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = numpy.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = numpy.multiply(error, X[:,i])
        grad[i] = numpy.sum(term) / len(X)

    return grad;

cost=costFunc(theta, X, y);
grad=gradientFunc(theta, X, y);

print("Cost at initial theta (zeros): " +  str(cost));
print("Expected cost (approx): 0.693\n");
print("Gradient at initial theta (zeros): \n");
print(grad);
print("Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n");




result = opt.fmin_tnc(func=costFunc, x0=theta, fprime=gradientFunc, args=(X, y))  

def predict(theta, X):  
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]

theta_min = numpy.matrix(result[0])  
predictions = predict(theta_min, X)  
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]  
accuracy = (sum(map(int, correct)) % len(correct))  
print 'accuracy = {0}%'.format(accuracy) 






