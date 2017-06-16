import scipy.io #Used to load the OCTAVE *.mat files
from scipy import linalg, sparse #for linalg
import matplotlib.pyplot as plt #for the graph
import numpy as np #for lin alg operations
from scipy.special import expit #vectorized sigmoid function
import itertools

input_layer_size  = 400  #20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10   

data_file = 'ex4data1.mat'
mat = scipy.io.loadmat(data_file)
X, y = mat['X'], mat['y']

m=X.shape[0]

weight_file = 'ex4weights.mat'
mat = scipy.io.loadmat( weight_file )
Theta1, Theta2 = mat['Theta1'], mat['Theta2']

myThetas = [ Theta1, Theta2 ]

def sigmoid(z):  
    return 1 / (1 + np.exp(-z));

def flattenParams(thetas_list):
    flattened_list = [ mytheta.flatten() for mytheta in thetas_list ]
    combined = list(itertools.chain.from_iterable(flattened_list))
    assert len(combined) == (input_layer_size+1)*hidden_layer_size + \
                            (hidden_layer_size+1)*num_labels
    return np.array(combined).reshape((len(combined),1))

nn_params= flattenParams(myThetas)

def sigmoidGradient(z):  
	g=np.zeros((z.shape[0], z.shape[1]))
	g=sigmoid(z)*(1-sigmoid(z))
   	return g

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambdah):
	Theta1 = np.reshape(nn_params[0:(hidden_layer_size*(input_layer_size+1)),0], (hidden_layer_size, (input_layer_size + 1)), order='F')
	Theta2 = np.reshape(nn_params[((hidden_layer_size*(input_layer_size+1))):nn_params.shape[0],0], (num_labels, (hidden_layer_size + 1)), order='F')

	m=X.shape[0]
	J=0
	Theta1_grad = np.zeros((Theta1.shape[0], Theta1.shape[1]))
	Theta2_grad = np.zeros((Theta2.shape[0], Theta2.shape[1]))

	a1 = np.hstack((np.ones((m,1)), X))
	z2=np.dot(a1,Theta1.T) #calculate z2
	a2=sigmoid(z2); #calculate g(z2)
	a2 = np.hstack((np.ones((m,1)), a2))

	z3=np.dot(a2,Theta2.T) #calculate z3 
	h_theta=a3=sigmoid(z3) #calculate g(z3), h_theta is 5000x10

	y_vector=np.zeros((m, num_labels))

	for i in range(m):
		if(y[i,0]==10): 
			y_vector[i, 0]=1 #row vectors that represent the correct output,y_vector is 5000x10
		else:
			y_vector[i, y[i,0]]=1 #row vectors that represent the correct output,y_vector is 5000x10

	left_term=sum((-1*y_vector)*np.log(h_theta))
	right_term=sum((1-y_vector)*np.log(1-h_theta))

	J=(1/m)*(sum(left_term-right_term))

	Theta1_term=sum(np.square(Theta1[:,1:Theta1.shape[1]]))
	Theta2_term=sum(np.square(Theta2[:,1:Theta2.shape[1]]))
	regularization=(sum(Theta1_term)+sum(Theta2_term))*(lambdah/(2*m));
	J=J+regularization;

	tridelt_1=0;
	tridelt_2=0;

	delta_3=a3-y_vector; #yk:{0,1} 

	z2=np.hstack((np.ones((m,1)), z2))
	delta_2=np.dot(delta_3,Theta2)*sigmoidGradient(z2)
	delta_2=delta_2[:,1:delta_2.shape[1]];

	tridelt_1=tridelt_1+np.dot((delta_2.T),(a1)) # Same size as Theta1_grad (25x401)
	tridelt_2=tridelt_2+np.dot((delta_3.T),(a2)) # Same size as Theta2_grad (10x26)

	Theta1_grad = tridelt_1 / m #for j=0, Dij=(1/m)*delta_ij
	Theta2_grad = tridelt_2 / m #for j=0, Dij=(1/m)*delta_ij

	Theta1_grad[:, 1:Theta1_grad.shape[1]] = Theta1_grad[:, 1:Theta1_grad.shape[1]] + (lambdah/m) * Theta1[:, 1:Theta1.shape[1]]
	Theta2_grad[:, 1:Theta2_grad.shape[1]] = Theta2_grad[:, 1:Theta2_grad.shape[1]] + (lambdah/m) * Theta2[:, 1:Theta2.shape[1]]

	grad_list = [ Theta1_grad, Theta2_grad ]

	grad = flattenParams(grad_list)

	ret=dict()
	ret['Cost']=J
	ret['Gradient']=grad;
	return ret


lst = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, 0)
J=lst['Cost']
grad=lst['Gradient']

print('Cost at parameters (loaded from ex4weights) (this value should be about 0.287629) :' + str(J))

lst = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, 1)
J=lst['Cost']
grad=lst['Gradient']

print('Cost at parameters (loaded from ex4weights) (this value should be about 0.383770) :' + str(J))



