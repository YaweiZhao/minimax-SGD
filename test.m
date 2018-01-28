clear;
%load data
data = load('~/simulation_based_machine_learning_library/dataset/heart/heart.mat');
data = data.data;
[n,d] = size(data);
label = data(:,1);
%label(label==2) = -1;% all the labels are +1 or -1.
training_data = data(:,2:d);
training_data = transpose(mapstd(training_data'));
%training_data = [training_data ones(n,1)];% add 1-offset
[n,d] = size(training_data);

%% initialize variables
T = 10;
alpha_0 = 1e-1;% learning rate for the primal update
beta_0 = 1e-6;%learning rate for the dual update
theta_sequence = zeros(n+n*n,T);
loss = zeros(T,1);
theta = rand(n+n*n,1);%primal variable, mu + L
y = ones(2,1);%dual variable
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
Knn = zeros(n,n);%kernel matrix
%mini-batch trick
b = 1;%mini-batch
stoc_nabla_mu = zeros(n,1);
stoc_nabla_L = zeros(n*n,1);

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
            Knn(i,j) = u_0*exp(-1/2*pair_diff * diag(ones(d,1) ./ u) * pair_diff'+ tau);
        end
    end
    %define the auxilary matrix
    Knn_inv = inv(Knn);
    Knn_det = det(Knn);
    
    epsilon = transpose(mvnrnd(zeros(1,n),eye(n,n)));
    Q = [eye(n) kron(epsilon',eye(n))];%
    %% update the primal variable
    %compute the stochastic gradients w.r.p.t theta
    i = randi(n);
    %the first item of g
    mu_temp = theta(1:n,:);
    L_temp = theta(n+1:n+n*n,:);
    L_temp = reshape(L_temp,n,n);
    stoc_nabla_mu_L_temp_1 = 1/(power(2*3.14159, n/2)*sqrt(Knn_det))*exp(-1/2*transpose(mu_temp + L_temp*epsilon)*Knn_inv*(mu_temp + L_temp*epsilon))*(-1/2)*2*Knn_inv*(mu_temp + L_temp*epsilon);
    stoc_nabla_mu_L_1 = Q'*stoc_nabla_mu_L_temp_1;%
    %the second item of g
    stoc_nabla_mu_L_temp_2 = zeros(n+n*n,1);
    for j=1:n
        stoc_nabla_mu_L_temp_2 = stoc_nabla_mu_L_temp_2 + (label(j)*transpose(Q(j,:)))/(1+exp(label(j)*Q(j,:)*theta));
    end
    stoc_nabla_mu_L_2 = stoc_nabla_mu_L_temp_2 + [zeros(n,1); reshape(inv(L_temp'), n*n,1)];
    stoc_nabla_mu_L = [stoc_nabla_mu_L_1 stoc_nabla_mu_L_2];
    % update rule for the primal variable: theta
    alpha = alpha_0/sqrt(t);
    theta = theta - alpha*stoc_nabla_mu_L*y;
    theta_sequence(:,t)  = theta;
    
    
    %% update the dual variable
    %compute the stochastic gradients w.r.p.t y
    mu_temp = theta(1:n,:);
    L_temp = theta(n+1:n+n*n,:);
    L_temp = reshape(L_temp,n,n);
    p_alpha_v_w = 1/(power((2*3.14159),n/2) * sqrt(Knn_det))*exp(-1/2*transpose(mu_temp + L_temp*epsilon) * Knn_inv*(mu_temp + L_temp*epsilon));
    log_p_q_1 = 0;
    for j=1:n
        log_p_q_1 = log_p_q_1 - log(1+exp(-label(j,:)*(mu_temp(j)+L_temp(j,:)*epsilon)));
    end
    log_p_q_2 = -1*(n/2*log(2*3.14159)+1/2*log(det(L_temp*transpose(L_temp)))) - 1/2*(epsilon'*epsilon);
    log_p_q = log_p_q_1  - log_p_q_2;
    g_v_w =  [p_alpha_v_w log_p_q];
    nabla_f_star_y = 1/([1 0]*y)*[1; 0];
    

    %update rule for the dual variable: y
    beta = beta_0/sqrt(t);
    y = y+beta*(g_v_w' - nabla_f_star_y);
    
    %% evaluate the loss
    theta_avg = 1/t*sum(theta_sequence,2);
    mu_temp = theta_avg(1:n,:);
    L_temp = theta_avg(n+1:n+n*n,:);
    L_temp = reshape(L_temp,n,n);
    log_p_alpha_v_w = -1*(n/2*log(2*3.14159)+1/2*log(Knn_det)) - 1/2*transpose(mu_temp)*Knn_inv*(mu_temp);
    log_p_q_1 = 0;
    for j=1:n
        log_p_q_1 = log_p_q_1 - log(1+exp(-label(j,:)*(mu_temp(j))));
    end
    log_p_q_2 = -1*(n/2*log(2*3.14159)+1/2*log(det(L_temp*transpose(L_temp)))) ;
    log_p_q = log_p_q_1  - log_p_q_2;
    loss(t,:) = log_p_alpha_v_w + log_p_q;
    
    
end
save('loss.mat','loss');
%plot the convergence of the loss function 
%plot([1:T],loss);
%xlabel('number of iterations');
%ylabel('loss')

