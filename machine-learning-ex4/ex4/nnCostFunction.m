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


summation = 0;
Delta1 = 0;
Delta2 = 0;

for i = 1:m
  % Part 1 | Forward prop
  a1 = X(i, :)(:);
  a1_with_bias = [ 1; a1 ];
  z2 = Theta1 * a1_with_bias;
  a2 = sigmoid(z2);
  a2_with_bias = [ 1; a2 ];
  z3 = Theta2 * a2_with_bias;
  h = sigmoid(z3);
  
  % compute unregularized J
  yi = zeros(num_labels, 1);
  yi(y(i)) = 1;  % resize y into vector
  summation = summation + sum(-yi .* log(h) - (1-yi) .* log(1-h));
  
  % Part 2 | Back prop
  d3 = h - yi;  % note that d3 = h - yi !!!
  d2 = (Theta2(:, 2:end)' * d3) .* sigmoidGradient(z2);  % !!!  also:sigmoidGradient(z2)
  Delta1 += d2 * a1_with_bias';   % size = 25 x 400
  Delta2 += d3 * a2_with_bias';
  % ¦Ä2 or d2 is tricky. It uses the (:,2:end) columns of Theta2. 
  % d2 is the product of d3 and Theta2(no bias), then element-wise scaled by sigmoid gradient of z2. 
  % The size is (m x r) ? (r x h) --> (m x h). The size is the same as z2, as must be.
  
  % Part 3 | Regularization
  Theta1NoBias = Theta1(:, 2:end); % !!!
  Theta2NoBias = Theta2(:, 2:end); % !!!
  reg_J = lambda/(2*m) * ( sum(sum(Theta1NoBias.^2, 2)) + sum(sum(Theta2NoBias.^2), 2) );
  reg_grad1 = (lambda/m) * Theta1NoBias;
  reg_grad2 = (lambda/m) * Theta2NoBias;
  
endfor

Theta1_grad = (1/m) * Delta1;
Theta2_grad = (1/m) * Delta2;

% Implement regularizaiton
J = summation / m + reg_J;
Theta1_grad(:, 2:end) += reg_grad1;
Theta2_grad(:, 2:end) += reg_grad2;












% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
