function g = sigmoid(z)
g = zeros(size(z));
A = exp(-z);
g = 1 ./ (1 + A);

end
