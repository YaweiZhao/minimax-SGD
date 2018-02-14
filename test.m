clear;
rng('default');
%load data
data = load('./heart.mat');
data = data.data;
data = data(1:2:100,:);
[n,d] = size(data);
n_test = fix(n/5);%used to test our algo.
label = data(1:n,1);
test_label = data(1:n_test,1);
%label(label==2) = -1;% all the labels are +1 or -1.
test_data = data(1:n_test,2:d);
training_data = data(1:n,2:d);% ALL data is used to train
%training_data = transpose(mapstd(training_data'));
%training_data = [training_data ones(n,1)];% add 1-offset
[n,d] = size(training_data);

%% initialize variables
T = 10;
alpha_0 = 1e-4;% learning rate for the primal update
beta_0 = 1e-4;%learning rate for the dual update
%alpha_0 = 2e-2;% learning rate for the primal update
%beta_0 = 1e-4;%learning rate for the dual update
theta_sequence = zeros(n+n*n,T);
train_loss = zeros(T,1);
test_loss = zeros(T,1);
%theta = ones(n+n*n,1);%primal variable, mu + L
theta = [1e-3*rand(n,1); 1e-10*rand(n*n,1)];
num_nodes_nn = fix(n);
w1 = [1e-5*rand(1,num_nodes_nn); 1e-5*rand(1,num_nodes_nn)];
w2 = 1e-7*rand(num_nodes_nn,n);
b1 = 1e-4*ones(2,1);
b2 = 1e-4*ones(num_nodes_nn,1);


pair_dist = zeros(n*n,1);
for i=1:n
    for j=1:n
        if i==j
            continue;
        end
        pair_dist((i-1)*n+j,:) = log(norm(training_data(i,:)-training_data(j,:)));
    end
end
pair_dist_ordering = sort(pair_dist);
mu_0 = pair_dist_ordering(fix(n*n/2));% hyper-parameter for sampling w
%sigma_0 = var(pair_dist_ordering);%hyper-parameter for sampling w
sigma_0 = 1;
Knn = zeros(n,n);%ARD kernel matrix

for t=1:T
    disp(t);
    %sample v, w
    logw = normrnd(mu_0,sigma_0,d+2,1);
    w = exp(logw);
    %u_0 = w(1,:);
    u_0 = 1;
    u = w(2:d+1,:);
    %tau = w(d+2,:);
    tau = 1e-6;
    %compute the kernel matrice
    for i=1:n
        for j=1:n
            pair_diff = training_data(i,:)-training_data(j,:);
            Knn(i,j) = u_0*exp(-1/2*pair_diff * diag(1 ./ u) * pair_diff'+ tau);
        end
    end
    %define the auxilary matrix
    Knn_inv = inv(Knn);
    Knn_det = det(Knn);
    
    epsilon = transpose(mvnrnd(zeros(1,n),eye(n)));
    y = w1 * (1 ./ exp(-1*(w2*epsilon+b2)))+b1;
    
    %% update the primal variable
    %compute the stochastic gradients w.r.p.t theta 
    mu_temp = theta(1:n,:);
    L_temp = theta(n+1:n+n*n,:);
    L_temp = reshape(L_temp,n,n);
    p_alpha_v_w = 1/(power(2*3.14159,n/2) * sqrt(Knn_det))*exp(-1/2*transpose(mu_temp + L_temp*epsilon) * Knn_inv*(mu_temp + L_temp*epsilon));
    log_p_q_1 = 0;
    
    %for test
    %y(1,:) = y(1,:) / (1/(power(2*3.14159,n/2) * sqrt(Knn_det))*exp(-1/2*transpose(mu_temp) * Knn_inv*(mu_temp)));
    
    
    for j=1:n
        if j<=n_test
            continue;% During training, the test data is discarded due to lack of labels.
        end
        log_p_q_1 = log_p_q_1 - log(1+exp(-label(j,:)*(mu_temp(j)+L_temp(j,:)*epsilon)));
    end
    log_p_q_2 = -1*(n/2*log(2*3.14159)+1/2*log(det(L_temp*L_temp'))) - 1/2*(epsilon'*epsilon);
    log_p_q = log_p_q_1  - log_p_q_2;
    g_v_w =  [p_alpha_v_w log_p_q];

    nabla_g1_theta_temp_mu = zeros(n,1);
    for i=1:n
        for j=1:n
            if i==j
                 nabla_g1_theta_temp_mu(i,:) = nabla_g1_theta_temp_mu(i,:) + transpose(mu_temp+L_temp*epsilon)*Knn_inv(:,i);
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
    
    nabla_g1_theta = 1/((2*3.14159)^(n/2)*sqrt(Knn_det))*exp(-1/2*transpose(mu_temp+L_temp*epsilon)*Knn_inv*(mu_temp+L_temp*epsilon))*(-1/2)*[nabla_g1_theta_temp_mu; reshape(nabla_g1_theta_temp_L,n*n,1)];
    
    nabla_g2_theta_I1 = [zeros(n_test,1); label(n_test+1:n,:) ./ (1+exp(label(n_test+1:n,:) .* (mu_temp(n_test+1:n,:)+L_temp(n_test+1:n,:)*epsilon))); reshape([zeros(n_test,n); repmat(label(n_test+1:n,:) ./ (1+exp(label(n_test+1:n,:) .* (mu_temp(n_test+1:n,:)+L_temp(n_test+1:n,:)*epsilon))),1,n) .* repmat(epsilon',n-n_test,1)], n*n,1)];
    nabla_g2_theta_I2 = [zeros(n,1);reshape( inv(L_temp'), n*n,1)];
    nabla_g2_theta = nabla_g2_theta_I1 + nabla_g2_theta_I2;
    %the third and fourth items of gradient in Step 5.
    %nabla_g_y_theta_g = y(1,:)*nabla_g1_theta+y(2,:)*nabla_g2_theta;
    nabla_g_y_theta_g = y(1,:)*nabla_g1_theta+y(2,:)*nabla_g2_theta;
    
    
    %nabla_f_star_theta_temp = 1 / (w1(1,:) * (1 ./ (1+exp(-1*w2*epsion-b2))));
    %the fifth item of gradeint in Step 5
    %nabla_f_star_epsilon = nabla_f_star_epsilon_temp * sum(repmat(w1(1,:),num_nodes_nn,1) .* w2' .* repmat(transpose(exp(w2*epsilon+b2) ./ ((1+exp(w2*epsilon+b2)) .^ 2)),n,1),2);
    
    
    % update rule for the primal variable: theta
    alpha = alpha_0 /sqrt(t);
    %theta = theta - alpha*(nabla_g_y_theta_g - nabla_f_star_epsilon);
    theta = theta - alpha*nabla_g_y_theta_g;
    theta_sequence(:,t)  = theta;
    
    
    %% update the dual variable
    %compute the stochastic gradients w.r.p.t y
    mu_temp = theta(1:n,:);
    L_temp = theta(n+1:n+n*n,:);
    L_temp = reshape(L_temp,n,n);
    p_alpha_v_w = 1/(power((2*3.14159),n/2) * sqrt(Knn_det))*exp(-1/2*transpose(mu_temp + L_temp*epsilon) * Knn_inv*(mu_temp + L_temp*epsilon));
    log_p_q_1 = 0;
    for j=1:n
        if j<=n_test
            continue;% During training, the test data is discarded due to lack of labels.
        end
        %% for numerical stability
        temp = -label(j,:)*(mu_temp(j)+L_temp(j,:)*epsilon);
        if temp>100
            log_p_q_1 = log_p_q_1 - (temp + log(1+exp(-1*temp)));
        else
            log_p_q_1 = log_p_q_1 - log(1+exp(temp));
        end
        %disp(log_p_q_1);
    end
    log_p_q_2 = -1*(n/2*log(2*3.14159)+1/2*log(det(L_temp*transpose(L_temp)))) - 1/2*(epsilon'*epsilon);
    log_p_q = log_p_q_1  - log_p_q_2;
    g_v_w =  [p_alpha_v_w log_p_q];
     
    nabla_f_star_temp = 1 / (w1(1,:)*(1 ./ (1+ exp(-1*(w2*epsilon + b2))))+b1(1,:));
    nabla_f_star_w1 = nabla_f_star_temp*[ transpose(1 ./ (1+exp(-1*(w2*epsilon+b2)))); zeros(1,num_nodes_nn)];%2 x num_nodes_nn
    nabla_f_star_w2 = nabla_f_star_temp * repmat(exp(w2*epsilon+b2) ./ ((1+exp(w2*epsilon+b2)) .^ 2), 1,n) .* repmat(transpose(w1(1,:)),1,n) .* repmat(epsilon',num_nodes_nn,1);
    nabla_f_star_b1 = [nabla_f_star_temp; 0];
    nabla_f_star_b2 = nabla_f_star_temp * transpose(w1(1,:)) .* (exp(w2*epsilon+b2) ./ ((1+exp(w2*epsilon+b2)) .^2) );
    
    nabla_g_y_g1_temp = g_v_w(1,1)* (1 ./ (1+exp(-1*(w2*epsilon+b2))));
    nabla_g_y_g2_temp = g_v_w(1,2)* (1 ./ (1+exp(-1*(w2*epsilon+b2))));    
    nabla_g_y_w1 = [nabla_g_y_g1_temp'; nabla_g_y_g2_temp'] ;% 2 x num_nodes_nn
    
    nabla_g_y_w2_temp_g1 = g_v_w(1,1)* transpose(w1(1,:)) .* (exp(w2*epsilon+b2)) ./ ((1+exp(w2*epsilon+b2)) .^ 2);
    nabla_g_y_w2_temp_g2 = g_v_w(1,2) * transpose(w1(2,:)) .* (exp(w2*epsilon+b2)) ./ ((1+exp(w2*epsilon+b2)) .^ 2);
    nabla_g_y_w2 = repmat(nabla_g_y_w2_temp_g1 + nabla_g_y_w2_temp_g2,1,n) .* repmat(epsilon',num_nodes_nn,1);
    
    nabla_g_y_b1 = [g_v_w(1,1);g_v_w(1,2)];
    nabla_g_y_b2 = repmat(g_v_w(1,1),num_nodes_nn,1) .* transpose(w1(1,:)) .* exp(w2*epsilon+b2) ./ ((1+exp(w2*epsilon+b2)) .^ 2) + repmat(g_v_w(1,2),num_nodes_nn,1) .* transpose(w1(2,:)) .* exp(w2*epsilon+b2) ./ ((1+exp(w2*epsilon+b2)) .^ 2);
    
    % update rule for the dual variable: y
    beta = beta_0/sqrt(t);
    w1 = w1+beta*(nabla_g_y_w1 - nabla_f_star_w1);
    w2 = w2+beta*(nabla_g_y_w2 - nabla_f_star_w2);
    b1 = b1+beta*(nabla_g_y_b1 - nabla_f_star_b1);
    b2 = b2+beta*(nabla_g_y_b2 - nabla_f_star_b2);
    %% evaluate loss: for train data and test data
    theta_avg = 1/t*sum(theta_sequence,2);
    mu_temp = theta_avg(1:n,:);
    
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

    disp('be careful!');
    
end
save('train_loss.mat','train_loss');
save('test_loss.mat','test_loss');
%plot the convergence of the loss function 
% plot(1:T,train_loss);
% xlabel('number of iterations');
% ylabel('loss on training data')


