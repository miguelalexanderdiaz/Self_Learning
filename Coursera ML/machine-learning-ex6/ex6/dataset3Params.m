function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_vals=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_vals=C_vals;

c_opt=0;
s_opt=0;
err_opt=-1;


for i=1:columns(C_vals)
  for j=1:columns(sigma_vals)
  
    model= svmTrain(X, y, C_vals(i), @(x1, x2) gaussianKernel(x1, x2, sigma_vals(j)));
    pred=svmPredict(model,Xval);
    err=sum((yval-pred).^2)/(2*columns(pred));
 
    if (or(err_opt==-1 , err < err_opt))
      err_opt=err;
      C=C_vals(i);
      sigma=sigma_vals(j);
      display([err_opt, C, sigma]);
    endif
    
  endfor
endfor



% =========================================================================

end
