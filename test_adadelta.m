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
%label(label==2) = -1;% all the labels are +1 or -1.
test_data = data(1:n_test,2:d);
training_data = data(n_test+1:n,2:d);
%training_data = transpose(mapstd(training_data'));
%training_data = [training_data ones(n,1)];% add 1-offset
[n_train,d] = size(training_data);


% 
% %% logistic regression
% % label contains: 0 or 1 NOT -1 or 1
% training_label(training_label<0)= 0;
% test_label(test_label<0)= 0;
% logi_w = glmfit(training_data,training_label,'binomial', 'link', 'logit');
% train_likelyhold = glmval(logi_w,training_data, 'logit');
% test_likelyhold = glmval(logi_w,test_data, 'logit');
% mean_train_likelyhold = mean(training_label .* log(train_likelyhold) + (1-train_likelyhold) .* log(1-train_likelyhold));
% mean_test_likelyhold = mean(test_label .* log(test_likelyhold) + (1-test_likelyhold) .* log(1-test_likelyhold));


%% initialize variables
T =30000;
train_loss = zeros(T,1);
test_loss = zeros(T,1);
num_nodes_nn = fix(n);

w1 = -10*rand(1,num_nodes_nn);
w2 = randn(num_nodes_nn,n);
b1 = -10;
b2 = randn(num_nodes_nn,1);


%Ada-delta optimization method
rho = 0.9; eepsilon = 1e-6;
E_g_square_old = zeros(n+n*n,1);
E_delta_x_square_old = zeros(n+n*n,1);
E_g_square_old_dual_w1 = zeros(1,num_nodes_nn);
E_g_square_old_dual_w2 = zeros(num_nodes_nn,n);
E_g_square_old_dual_b1 = 0;
E_g_square_old_dual_b2 = zeros(num_nodes_nn,1);
E_delta_x_square_old_dual_w1 = zeros(1,num_nodes_nn);
E_delta_x_square_old_dual_w2 = zeros(num_nodes_nn,n);
E_delta_x_square_old_dual_b1 = 0;
E_delta_x_square_old_dual_b2 = zeros(num_nodes_nn,1);

%SGD optimization method
%alpha_0 = 1e-10;% learning rate for the primal update
%beta_0 =1e-8;%learning rate for the dual update
%theta_sequence = zeros(n+n*n,T);

%w1 = load('w11.mat');
%w1 = w1.w1;
%w2 = load('w22.mat');
%w2 = w2.w2;
%b1 = load('b11.mat');
%b1 = b1.b1;
%b2 = load('b22.mat');
%b2 = b2.b2;

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

%%initialize mu and L
%theta =[zeros(n,1); reshape(eye(n),n*n,1)];% use constant to initialize
%theta = load('theta.mat');%use the end of previous train to initialize
%theta = theta.theta;
%compute the kernel matrice to initialize L
Knn_expectation = zeros(n,n);%ARD kernel matrix
for i=1:n
    for j=1:n
        pair_diff = data(i,2:d+1) - data(j,2:d+1);
        Knn_expectation(i,j) = exp(-1/2*pair_diff * diag(1 ./ (exp(mu_0)*ones(d,1))) * pair_diff')+ 1e-3;
    end
end
Knn_expectation_logdet = logdet(Knn_expectation);
Knn_expectation_inv = inv(Knn_expectation);
L_temp = chol(Knn_expectation,'lower');
theta = [[zeros(n_test,1);training_label]; reshape( L_temp, n*n,1)];%initialize theta
%theta = [zeros(n,1); reshape( L_temp, n*n,1)];%initialize theta
mu_temp = theta(1:n,:);
L_temp = theta(n+1:n+n*n,:);
L_temp = reshape(L_temp,n,n);
%to match the y and the gradient of g1
p_alpha_v_w_expectation = 1e-18*exp(-n/2*log(2*3.14159)-1/2*Knn_expectation_logdet-1/2*transpose(mu_temp+L_temp*zeros(n,1))*Knn_expectation_inv*(mu_temp+L_temp*zeros(n,1)));
%parameters are saved for using in classic gp classification
u_save = zeros(d, T);
for t=1:T
    if mod(t, 50) == 0 
        disp(t);
    end
    Knn = zeros(n,n);%ARD kernel matrix
    %sample v, w
    logw = normrnd(mu_0,sigma_0,d+2,1);
    w = exp(logw);
    %w = exp(mu_0*ones(d+2,1));
    u_0 = 1;%%%NOTICE
    %u_0 = exp(randn(1));
    u = w(2:d+1,:);
    %u = ones(d,1);
    %tau = w(d+2,:);
    tau = 1e-3;
    u_save(:,t) = u;
    %compute the kernel matrice
    kernel_temp = (data(:,2:d+1) .* repmat(transpose(1 ./ u),n,1)) * transpose(data(:,2:d+1));
    kernel_diag = diag(kernel_temp);
    Knn_temp = -2*kernel_temp + ones(n,n)*diag(kernel_diag) +diag( kernel_diag)*ones(n,n);
    Knn = u_0*exp(-1/2*Knn_temp)+tau;
    
    %define the auxilary matrix
    Knn_inv = inv(Knn);
    Knn_logdet = logdet(Knn);
    epsilon = randn(n,1);

    %% update the primal variable
    %compute the stochastic gradients w.r.p.t theta 
    mu_temp = theta(1:n,:);
    L_temp = theta(n+1:n+n*n,:);
    L_temp = reshape(L_temp,n,n);
    y = w1 * (1 ./ exp(-1*(w2*epsilon+b2)))+b1;

    nabla_g1_theta_temp_mu = zeros(n,1);
    for i=1:n
        for j=1:n
            if i==j
                 nabla_g1_theta_temp_mu(i,:) = nabla_g1_theta_temp_mu(i,:) + transpose(mu_temp+L_temp*epsilon)*Knn_inv(:,i)+(mu_temp(i,:)+L_temp(i,:)*epsilon)*Knn_inv(i,i);
            else
                nabla_g1_theta_temp_mu(i,:) = nabla_g1_theta_temp_mu(i,:) + Knn_inv(i,j)*(mu_temp(j,:)+L_temp(j,:)*epsilon);
            end
        end
    end
    nabla_g1_theta_temp_L = zeros(n,n);
    for i=1:n
        for j=1:n
            if i==j
                 nabla_g1_theta_temp_L(i,:) = nabla_g1_theta_temp_L(i,:) + (transpose(mu_temp+L_temp*epsilon)*Knn_inv(:,i) + Knn_inv(i,i)*(mu_temp(i,:)+L_temp(i,:)*epsilon))*epsilon';
            else
                nabla_g1_theta_temp_L(i,:) = nabla_g1_theta_temp_L(i,:) + (Knn_inv(i,j)*(mu_temp(j,:)+L_temp(j,:)*epsilon))*epsilon';
            end
        end
    end    
    
    %temp = (-1/2)*[nabla_g1_theta_temp_mu; reshape(tril(nabla_g1_theta_temp_L),n*n,1)];
    temp = (-1/2)*[nabla_g1_theta_temp_mu; reshape(nabla_g1_theta_temp_L,n*n,1)];
    log_nabla_g1_theta_temp = -n/2*log(2*3.14159)-1/2*Knn_logdet-1/2*transpose(mu_temp+L_temp*epsilon)*Knn_inv*(mu_temp+L_temp*epsilon);

    nabla_g2_theta_I1 = [zeros(n_test,1); training_label ./ (1+exp(training_label .* (mu_temp(n_test+1:n,:)+L_temp(n_test+1:n,:)*epsilon))); reshape([zeros(n_test,n); repmat(training_label ./ (1+exp(training_label .* (mu_temp(n_test+1:n,:)+L_temp(n_test+1:n,:)*epsilon))),1,n) .* repmat(epsilon',n-n_test,1)], n*n,1)];
    nabla_g2_theta_I2 = [zeros(n,1);reshape( inv(L_temp'), n*n,1)];
    nabla_g2_theta = nabla_g2_theta_I1 + nabla_g2_theta_I2;
    %the gradient w.r.t theta
    %nabla_g_y_theta_g = -1*exp(log(-y)+log_nabla_g1_theta_temp)*temp-nabla_g2_theta;
    nabla_g1_theta_temp = exp(log_nabla_g1_theta_temp)*temp;
    nabla_g_y_theta_g = 1/p_alpha_v_w_expectation*y*nabla_g1_theta_temp - nabla_g2_theta;
    
    % update rule for the primal variable: theta
    
    %set the primal step size via ada-delta method
    E_g_square_new = rho*E_g_square_old + (1-rho)*(nabla_g_y_theta_g .^ 2);
    Delta_x = -1*sqrt(E_delta_x_square_old+eepsilon) ./ sqrt(E_g_square_new+eepsilon) .* nabla_g_y_theta_g;
    E_delta_x_square_new = rho*E_delta_x_square_old + (1-rho)*(Delta_x .^ 2);
    
    theta = theta + Delta_x;
    
    E_delta_x_square_old = E_delta_x_square_new;
    E_g_square_old = E_g_square_new;
    
    %set the primal step size via SGD method
    %alpha = alpha_0 /sqrt(t);
    %alpha = alpha_0;
    %theta = theta - alpha*nabla_g_y_theta_g;
    %remember: L is a low-triangle matrix
    L_temp = theta(n+1:n+n*n,:);
    L_temp = reshape(L_temp,n,n);
    L_temp = tril(L_temp);
    theta(n+1:n+n*n,:) = reshape(L_temp,n*n,1);
    %theta_sequence(:,t)  = theta;
    %theta = mean(theta_sequence,2);
    
    
    %% update the dual variable
    %compute the stochastic gradients w.r.t y
    mu_temp = theta(1:n,:);
    L_temp = theta(n+1:n+n*n,:);
    L_temp = reshape(L_temp,n,n);
    %y = 1/p_alpha_v_w_expectation*w1 * (1 ./ exp(-1*(w2*epsilon+b2)))+b1;
    y = w1 * (1 ./ exp(-1*(w2*epsilon+b2)))+b1;
    %p_alpha_v_w = 1/(power(2*3.14159,n/2) * sqrt(Knn_det))*exp(-1/2*transpose(mu_temp + L_temp*epsilon) * Knn_inv*(mu_temp + L_temp*epsilon));
    p_alpha_v_w = exp(-n/2*log(2*3.14159)-1/2*Knn_logdet-1/2*transpose(mu_temp+L_temp*epsilon)*Knn_inv*(mu_temp+L_temp*epsilon));
    
    nabla_g_y_g1_temp =  1 ./ (1+exp(-1*(w2*epsilon+b2)));
    %approximate essential_temp = 1/y+g_v_w(1,1);
    %if exp(log_p_alpha_v_w+log(y)) >1e
     %   essential_temp = exp(log_p_alpha_v_w + exp(- log(y)-log_p_alpha_v_w));
    %else
        %essential_temp = -1*exp(log_p_alpha_v_w + log(-1+exp(-1*(log(-y)+log_p_alpha_v_w))));%%%%%
        essential_temp = 1/p_alpha_v_w_expectation*p_alpha_v_w+1/y;
    %end
    nabla_g_y_w1 = nabla_g_y_g1_temp' * essential_temp;% 1 x num_nodes_nn
    
    nabla_g_y_w2_temp = w1' .* ((exp(-1*(w2*epsilon+b2))) ./ ((1+(exp(-1*(w2*epsilon+b2)))) .^ 2)); 
    nabla_g_y_w2 = essential_temp*(repmat(nabla_g_y_w2_temp,1,n) .* repmat(epsilon',num_nodes_nn,1));
    
    nabla_g_y_b1 = essential_temp;
    nabla_g_y_b2_temp = w1' .* exp(-1*(w2*epsilon+b2)) ./ ((1+exp(-1*(w2*epsilon+b2))) .^ 2);
    nabla_g_y_b2 = essential_temp*nabla_g_y_b2_temp;
    
    % ada-delta method for the dual variable: y
    
    E_g_square_new_dual_w1 = rho*E_g_square_old_dual_w1 + (1-rho)*(nabla_g_y_w1 .^ 2);
    E_g_square_new_dual_w2 = rho*E_g_square_old_dual_w2 + (1-rho)*(nabla_g_y_w2 .^ 2);
    E_g_square_new_dual_b1 = rho*E_g_square_old_dual_b1 + (1-rho)*(nabla_g_y_b1 .^ 2);
    E_g_square_new_dual_b2 = rho*E_g_square_old_dual_b2 + (1-rho)*(nabla_g_y_b2 .^ 2);
    
    Delta_x_dual_w1 = sqrt(E_delta_x_square_old_dual_w1+eepsilon) ./ sqrt(E_g_square_new_dual_w1+eepsilon) .* nabla_g_y_w1;
    Delta_x_dual_w2 = sqrt(E_delta_x_square_old_dual_w2+eepsilon) ./ sqrt(E_g_square_new_dual_w2+eepsilon) .* nabla_g_y_w2;
    Delta_x_dual_b1 = sqrt(E_delta_x_square_old_dual_b1+eepsilon) ./ sqrt(E_g_square_new_dual_b1+eepsilon) .* nabla_g_y_b1;
    Delta_x_dual_b2 = sqrt(E_delta_x_square_old_dual_b2+eepsilon) ./ sqrt(E_g_square_new_dual_b2+eepsilon) .* nabla_g_y_b2;
    
    E_delta_x_square_new_dual_w1 = rho*E_delta_x_square_old_dual_w1 + (1-rho)*(Delta_x_dual_w1 .^ 2);
    E_delta_x_square_new_dual_w2 = rho*E_delta_x_square_old_dual_w2 + (1-rho)*(Delta_x_dual_w2 .^ 2);
    E_delta_x_square_new_dual_b1 = rho*E_delta_x_square_old_dual_b1 + (1-rho)*(Delta_x_dual_b1 .^ 2);
    E_delta_x_square_new_dual_b2 = rho*E_delta_x_square_old_dual_b2 + (1-rho)*(Delta_x_dual_b2 .^ 2);
    
    
    w1_temp = w1 + Delta_x_dual_w1;
    w2_temp = w2 + Delta_x_dual_w2;
    b1_temp = b1 + Delta_x_dual_b1;
    b2_temp = b2 + Delta_x_dual_b2;
    y_temp = w1_temp * (1 ./ exp(-1*(w2_temp*epsilon+b2_temp)))+b1_temp;
    if y_temp>= 0 
        disp('y');
        disp(y_temp);
    else
        w1 = w1 + Delta_x_dual_w1;
        w2 = w2 + Delta_x_dual_w2;
        b1 = b1 + Delta_x_dual_b1;
        b2 = b2 + Delta_x_dual_b2;
        y = w1 * (1 ./ exp(-1*(w2*epsilon+b2)))+b1;
        E_delta_x_square_old_dual_w1 = E_delta_x_square_new_dual_w1;
        E_delta_x_square_old_dual_w2 = E_delta_x_square_new_dual_w2;
        E_delta_x_square_old_dual_b1 = E_delta_x_square_new_dual_b1;
        E_delta_x_square_old_dual_b2 = E_delta_x_square_new_dual_b2;
        
        E_g_square_old_dual_w1 = E_g_square_new_dual_w1;
        E_g_square_old_dual_w2 = E_g_square_new_dual_w2;
        E_g_square_old_dual_b1 = E_g_square_new_dual_b1;
        E_g_square_old_dual_b2 = E_g_square_new_dual_b2;
    end
    
    
    
    % SGD for the dual variable: y
    %beta = beta_0/sqrt(t);
    %w1 = w1+beta*nabla_g_y_w1/p_alpha_v_w_expectation;
    %w2 = w2+beta*nabla_g_y_w2/p_alpha_v_w_expectation;
    %b1 = b1+beta*nabla_g_y_b1/p_alpha_v_w_expectation;
    %b2 = b2+beta*nabla_g_y_b2/p_alpha_v_w_expectation;
    
    
    
    %% evaluate loss: for train data and test data
    %theta_avg = 1/t*sum(theta_sequence,2);
    %mu_temp = theta_avg(1:n,:);
    mu_temp = theta(1:n,:);
    %% evaluate the train loss
    train_loss_temp = 0;
    for i=n_test+1:n
        temp = label(i,:)*mu_temp(i,:);
        train_loss_temp = train_loss_temp -log(1+exp(-temp));
    end
    train_loss(t,:) = train_loss_temp/(n-n_test);
    
    %% evaluate the test loss
    test_loss_temp = 0;
    for i=1:n_test
        temp = label(i,:)*mu_temp(i,:);
        test_loss_temp = test_loss_temp - log(1+exp(-temp));
    end
    test_loss(t,:) = test_loss_temp/n_test;

    %disp('be careful!');
    
end
u_save = mean(u_save,2);
 save('w1.mat','w1')
 save('w2.mat','w2')
 save('b2.mat','b2')
 save('b1.mat','b1')
 save('theta.mat','theta')

%save('u_save.mat','u_save');
save('train_loss.mat','train_loss');
save('test_loss.mat','test_loss');

%% for test 
mu_temp = theta(1:n,:);
%kernel matrix for training dataset
Knn_train_train = zeros(n_train,n_train);
    for i=1:n_train
        for j=1:n_train
                pair_diff = data(n_test+i,2:d+1) - data(n_test+j,2:d+1);
                Knn_train_train(i,j) = exp(-1/2*pair_diff * diag(1 ./ (ones(d,1)*exp(mu_0))) * pair_diff')+ 1e-3;  
        end
    end
%kernel matrix for test dataset x  training dataset
Knn_test_train = zeros(n_test,n_train);
    for i=1:n_test
        for j=1:n_train
                pair_diff = data(i,2:d+1) - data(n_test+j,2:d+1);
                Knn_test_train(i,j) = exp(-1/2*pair_diff * diag(1 ./ (ones(d,1)*exp(mu_0))) * pair_diff')+ 1e-3;  
        end
    end
    
    
    mu_test2 = Knn_test_train*inv(Knn_train_train)*mu_temp(n_test+1:n,:);
    %% evaluate the test loss
    test_loss_temp = 0;
    for i=1:n_test
        temp = label(i,:)*mu_test2(i,:);
        test_loss_temp = test_loss_temp - log(1+exp(-temp));
    end
    test_loss2 = test_loss_temp/n_test;
    
    
