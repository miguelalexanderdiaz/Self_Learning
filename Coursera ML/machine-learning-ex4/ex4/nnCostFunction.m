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
J = 0;     
% Add ones to the X data matrix
X = [ones(m, 1) X];   
%y=y(randi ([1,length(y)], 20, 1));

y_k=eye(num_labels)(y,:); %convert y to one hot vector
a_2=sigmoid(Theta1*X')'; %calculate input dot w_1
a_2 = [ones(m, 1) a_2];  %append the bias
a_3=sigmoid(Theta2*a_2')'; %calculate hidden layer activations dot w_2
J+=1/m.*sum(sum(-y_k.*log(a_3)-(1-y_k).*(log(1-a_3)),2)); %compute the cost function


t1_aux=Theta1; %create copies of theta1,theta2
t2_aux=Theta2; 
t1_aux(:,1)=0; %remove biases
t2_aux(:,1)=0;

J+=lambda/(2*m)*((sum(sum(t1_aux.^2,2)))+sum(sum(t2_aux.^2,2))); %


d_3=a_3-y_k; %difference in output layer
d_2=(d_3*Theta2(:,2:end)).*sigmoidGradient((X*Theta1'));%difference in hidden layer



Theta1(:,1)=0; %remove biases
Theta2(:,1)=0;

Theta1_grad = 1/m.* (d_2'*X)+ lambda/m.*Theta1;

Theta2_grad = 1/m.*( d_3'*a_2)+lambda/m.*Theta2;




% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
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
