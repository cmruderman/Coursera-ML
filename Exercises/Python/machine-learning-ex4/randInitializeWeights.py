import scipy.io #Used to load the OCTAVE *.mat files
from scipy import linalg, sparse #for linalg
import numpy as np #for lin alg operations
from numpy import random as rand
import itertools

input_layer_size  = 400
hidden_layer_size = 25
num_labels = 10

def randInitializeWeights(L_in, L_out):  
	W=np.zeros((L_out, 1+L_in))
	epsilon_init=.12;
	return np.mat(np.random.rand(L_out, 1+L_in)*2*epsilon_init-epsilon_init);

def flattenParams(thetas_list):
    flattened_list = [ mytheta.flatten() for mytheta in thetas_list ]
    combined = list(itertools.chain.from_iterable(flattened_list))
    #assert len(combined) == (input_layer_size+1)*hidden_layer_size + \
                            #(hidden_layer_size+1)*num_labels
    return np.array(combined).reshape((len(combined),1))

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

theta_list = [ initial_Theta1, initial_Theta2 ]

initial_nn_params = flattenParams(theta_list)

lst  = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, 3);

debug_J=lst['Cost']
grad=lst['Gradient']

print('Cost at (fixed) debugging paramters (this value should be about 0.576051)' + str(debug_J))


