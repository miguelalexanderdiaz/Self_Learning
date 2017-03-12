function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
sig_z=sigmoid(sum(theta'.*X,2));
J= 1/m*sum(-y.*log(sig_z)-(1-y).*log(1-sig_z));



reg_J=lambda/(2*m)*sum(theta(2:length(theta)).^2);

J+=reg_J;


reg_grad=(lambda/m).*theta';
reg_grad(1)=0; %don't regularize theta_0

grad = (1/m)*sum((sig_z-y).*X)+reg_grad;



% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
