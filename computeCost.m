function J = computeCost(X, y, theta)
m = length(y);
J = 0;
h = sigmoid(X * theta);
J = 1/m * (-y' * log(h)- (1-y)' * log(1-h));


end