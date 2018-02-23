clear; close all;
rng('default');
%load data
data = load('./heart.mat');
data = data.data;
data = data(1:2:100,:);
[n,d] = size(data);
n_test = fix(n/5);%used to test our algo.
label = data(1:n,1);
test_label = data(1:n_test,1);
training_label = label(n_test+1:n,:);
test_data = data(1:n_test,2:d);
training_data = data(n_test+1:n,2:d);% ALL data is used to train
%training_data = transpose(mapstd(training_data'));
%training_data = [training_data ones(n,1)];% add 1-offset
[n,d] = size(training_data);

meanfunc = @meanConst; hyp.mean = 0;
covfunc = @covSEard;   
%hyp.cov = log(ones(1,d+1));%%%NOTICE
u_cov = load('./u_save.mat');
hyp.cov = log([transpose(u_cov.u_save) 1]);
likfunc = @likLogistic;
hyp = minimize(hyp, @gp, -800, @infLaplace, meanfunc, covfunc, likfunc, training_data, training_label);
[a b c d lp] = gp(hyp, @infLaplace, meanfunc, covfunc, likfunc, training_data, training_label, test_data, test_label);

%hyp = minimize(hyp, @gp, -900, @infEP, meanfunc, covfunc, likfunc, training_data, training_label);
%[a b c d lp] = gp(hyp, @infEP, meanfunc, covfunc, likfunc, training_data, training_label, test_data, test_label);


%hyp = minimize(hyp, @gp, -800, @infVB, meanfunc, covfunc, likfunc, training_data, training_label);
%[a b c d lp] = gp(hyp, @infVB, meanfunc, covfunc, likfunc, training_data, training_label, test_data, test_label);

%% plot the test likelyhold
plot(1:n_test, lp);
