function [T, train_loss, test_loss, num_nodes_nn, y_new_plot, w1, w2, b1, b2, mu_0, sigma_0] = initialize_parameters(data, n,d)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
%% initialize variables
T =500;
train_loss = zeros(T,1);
test_loss = zeros(T,1);
num_nodes_nn = fix(n);
y_new_plot = zeros(T,1);

w1 = randn(1,num_nodes_nn);
w2 = 1e-1*randn(num_nodes_nn,n);
b1 = 0;
b2 = 1e-1*randn(num_nodes_nn,1);

pair_dist = zeros(n*n,1);

for i=1:n
    for j=1:n
        if i==j
            continue;
        end
        pair_dist((i-1)*n+j,:) = log(norm(data(i,2:d+1)-data(j,2:d+1)));
    end
end
pair_dist_ordering = sort(pair_dist);
mu_0 = pair_dist_ordering(fix((n*n-n)/2));% hyper-parameter for sampling w
%sigma_0 = 3*var(pair_dist_ordering);%hyper-parameter for sampling w
sigma_0 = 1;

end

