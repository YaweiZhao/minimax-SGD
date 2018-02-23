%ridge regression
clear; close all;
rng('default');
%load data: computer_hardware
data = load('./heart.mat');
data = data.data;
data = data(1:2:100,:);

[n,d] = size(data);
label = data(:,1);
n_test = 10;
test_label = label(1:n_test,:);
test_data = [data(1:n_test,2:d) ones(n_test,1)];
%training_data = transpose(mapstd(training_data'));
training_label = label(n_test+1:n,:);
training_data = [data(n_test+1:n,2:d) ones(n - n_test,1)];% add 1-offset
[n_training,d] = size(training_data);
%initialize parameters
eta = 1e-1;%learning rate
T = 500000;%total number of iterations
x = zeros(d,1);%the initial parameter
loss = zeros(T,1);


loss_init = 1/n_training*sum(log(1+exp(-1*training_label .* (training_data*x))));
stoc_nabla_x = zeros(d,1);
b = fix(0.1*n_training);%mini-batch
for t=1:T
    stoc_nabla_x = zeros(d,1);
    for j=1:b
        i = randi(n_training);
        stoc_nabla_x_temp = -(transpose(training_data(i,:))*training_label(i,:))/(1+exp(training_label(i,:)*training_data(i,:)*x));
        stoc_nabla_x = stoc_nabla_x + stoc_nabla_x_temp;
    end
    stoc_nabla_x = 1/b*stoc_nabla_x;
    x = x - eta/sqrt(t)*stoc_nabla_x;
    %evaluate the loss
    loss(t,1) = 1/n_training*sum(log(1+exp(-1*training_label .* (training_data*x))));
end
train_likelyhold = -1*loss(T,1);
test_likelyhold = -mean(log(1+exp(-1*test_label .* (test_data*x))));












