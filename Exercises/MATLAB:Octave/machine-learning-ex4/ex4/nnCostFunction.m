function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

a1 = [ones(m, 1) X]; %include bias unit

z2=a1*Theta1'; %calculate z2
a2=sigmoid(z2); %calculate g(z2)
a2=[ones(m, 1) a2]; %include bias unit a0(2)

z3=a2*Theta2'; %calculate z3
h_theta=a3=sigmoid(z3); %calculate g(z3)
						%h_theta is 5000x10

for w=1:m,
	y_vector(w, y(w))=1; %row vectors that represent the correct output,y_vector is 5000x10
end;



left_term=sum((-1*y_vector).* log(h_theta));
right_term=sum((1-y_vector).*log(1-h_theta));

J=(1/m)*(sum(left_term-right_term));

Theta1_term=sum(Theta1(:,2:end).^2);
Theta2_term=sum(Theta2(:,2:end).^2);
regularization=(sum(Theta1_term)+sum(Theta2_term))*(lambda/(2*m));
J=J+regularization;

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).
tridelt_1=0;
tridelt_2=0;

delta_3=a3-y_vector; %yk:{0,1} 

z2=[ones(m, 1) z2]; %include bias unit a0(2)
delta_2=(delta_3*Theta2).*sigmoidGradient(z2);
delta_2=delta_2(:,2:end);

tridelt_1=tridelt_1+(delta_2')*(a1); % Same size as Theta1_grad (25x401)
tridelt_2=tridelt_2+(delta_3')*(a2); % Same size as Theta2_grad (10x26)


Theta1_grad = tridelt_1 / m; %for j=0, Dij=(1/m)*delta_ij
Theta2_grad = tridelt_2 / m; %for j=0, Dij=(1/m)*delta_ij

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda/m) * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda/m) * Theta2(:, 2:end);

%for j>=1, Dij=(1/m)*delta_ij+(lambda/m)(theta_ij)



% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
