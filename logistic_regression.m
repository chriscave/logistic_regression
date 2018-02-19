data = load('iris.data');
bin_data = data(1:100,:);
X = bin_data(:,[1:4]);
y = bin_data(:, 5);

[m, n] = size(X);
X = [ones(m, 1) X];

initial_theta = zeros(n + 1, 1);
num_iters = 1500;
alpha = 0.01;


[theta, J_history] = gradientDescent(X, y, initial_theta, alpha, num_iters);

[p,prob] = predict(theta, X,y);
prob