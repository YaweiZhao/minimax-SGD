clear; close all;
rng('default');
%load data
data = load('../heart.mat');
data = data.data;
data = data(1:2:100,:);
label = data(:,1);
%data = transpose(mapstd(data'));
[n,d] = size(data);
n_test = fix(n/5);%used to test our algo.
test_label = label(1:n_test,:);
training_label = label(n_test+1:n,:);
test_data = data(1:n_test,2:d);
training_data = data(n_test+1:n,2:d);% ALL data is used to train
[n_train,d] = size(training_data);

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

meanfunc = @meanConst; hyp.mean = 0;
covfunc = @covSEard;   
hyp.cov = log([sqrt(exp(mu_0*ones(1,d))) 1]);%%%NOTICE
likfunc = @likLogistic;
%hyp = minimize(hyp, @gp, 2, @infLaplace, meanfunc, covfunc, likfunc, training_data, training_label);
[a b c d lp] = gp(hyp, @infLaplace, meanfunc, covfunc, likfunc, training_data, training_label, test_data, test_label);
%a = hyp.cov;
%hyp = minimize(hyp, @gp, 900, @infEP, meanfunc, covfunc, likfunc, training_data, training_label);
%[a b c d lp] = gp(hyp, @infEP, meanfunc, covfunc, likfunc, training_data, training_label, test_data, test_label);


%hyp = minimize(hyp, @gp, -800, @infVB, meanfunc, covfunc, likfunc, training_data, training_label);
%[a b c d lp] = gp(hyp, @infVB, meanfunc, covfunc, likfunc, training_data, training_label, test_data, test_label);

%% plot the test likelyhold
plot(1:n_test, lp);
