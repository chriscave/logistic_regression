function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

m = length(y); 
J_history = zeros(num_iters, 1);

for iter = 1:num_iters;  

h = X * theta;
error = sigmoid(h) - y;
gradient = X' * error;
theta_change = (alpha / m) * gradient;
theta = theta - theta_change;






    J_history(iter) = computeCost(X, y, theta);

end

end