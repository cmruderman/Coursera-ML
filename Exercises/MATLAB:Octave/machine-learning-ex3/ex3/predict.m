function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

X = [ones(m, 1) X]; %include bias unit

z2=X*Theta1'; %calculate z2
a2=sigmoid(z2); %calculate g(z2)
hidden_layer=[ones(m, 1) a2]; %include bias unit a0(2)

z3=hidden_layer*Theta2'; %calculate z3
a3=sigmoid(z3); %calculate g(z3)

[pval, p] = max(a3, [], 2); 
  %max returns the max value (pval) and the index of the max value (p)






% =========================================================================


end
