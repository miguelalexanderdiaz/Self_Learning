function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);
X = [ones(m, 1) X]; % Add ones to the X data matrix

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

a_1=sigmoid(Theta1*X')';

size(X)
a_1 = [ones(size(a_1,1), 1) a_1]; %append the bias for the next layer
size(a_1)
a_2=sigmoid(Theta2*a_1')';
size(a_2)

for i=1:m
  maxim=max(a_2,[],2); %get the maximum probability for each example
  p(i)=find(a_2(i,:)==maxim(i));
endfor





% =========================================================================


end
