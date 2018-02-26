function [data, label, training_data, test_data, training_label, test_label, n, d, n_train, n_test ] = prepare_data( )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
%load data
data = load('./heart.mat');
data = data.data;
data = data(1:50,:);
[n,d] = size(data);
n_test = fix(n/5);%used to test our algo.
label = data(1:n,1);
test_label = data(1:n_test,1);
training_label = label(n_test+1:n,:);
%label(label==2) = -1;% all the labels are +1 or -1.
test_data = data(1:n_test,2:d);
training_data = data(n_test+1:n,2:d);
%training_data = transpose(mapstd(training_data'));
%training_data = [training_data ones(n,1)];% add 1-offset
[n_train,d] = size(training_data);
end

