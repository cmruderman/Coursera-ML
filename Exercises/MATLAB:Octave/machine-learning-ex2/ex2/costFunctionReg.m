function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
argument=0;
g=0;
subtraction = zeros(size(y));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

argument=((theta')*X')';   %theta transpose X
g=(1./(1+(exp(-1*argument))));

sumation=sum(((-1*y)'*log(g))-((1-y)'*log(1-g)));
J=((1/m)*(sumation))+(lambda/(2*m))*sum((theta(2:end,:).^2));


%%%%%%% theta 0 %%%%%%%%%%%
for feature=1:m,
	subtraction(feature)=subtraction(feature)+((g(feature)-y(feature))'*X(feature,1));
end;

grad(1)=(1/m)*sum(subtraction);
%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%% theta >=1 %%%%%%%%%%%

% for i=1:m,
	for j=2:length(theta),
		grad(j)=((1/m)*sum((g-y)'*X(:,j)))+((lambda/m)*theta(j))+grad(j);
	end;
% end; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%

% =============================================================

end
