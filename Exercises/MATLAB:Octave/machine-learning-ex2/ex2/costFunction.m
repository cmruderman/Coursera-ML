function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

argument=0;
g=0;
sumation=0;
subtraction = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

argument=((theta')*X')';
g=(1./(1+(exp(-1*argument))));

sumation=sum(((-1*y)'*log(g))-((1-y)'*log(1-g)));
J=(1/m)*(sumation);

for iter = 1:m,
	for feature=1:length(theta),
		subtraction(feature)=subtraction(feature)+((g(iter)-y(iter))'*X(iter,feature));
	end;
end;

grad=(1/m)*subtraction;




% =============================================================

end
