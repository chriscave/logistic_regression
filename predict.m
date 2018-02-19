function [p,prob] = predict(theta, X,y)
m = size(X, 1);
prob = 0;

p = zeros(size(X, 1), 1);

A = sigmoid(X * theta);
p = round(A);
dif = abs(p-y);
sum_dif = sum(dif(:) == 1);
prob = 1 - (sum_dif / length(y));





end
