clear;
%load data
data = load('/Users/yawei/Documents/MATLAB/simulation based algorithm test library/simulation_based_machine_learning_library/dataset/heart/heart.mat');
data = data.data;
[n,d] = size(data);
label = data(:,1);
%label(label==2) = -1;% all the labels are +1 or -1.
training_data = data(:,2:d);
%training_data = transpose(mapstd(training_data'));
%training_data = [training_data ones(n,1)];% add 1-offset
[n,d] = size(training_data);


%% initialize variables
T = 10;
alpha_0 = 1e-2;% learning rate for the primal update
beta_0 = 1e-2;%learning rate for the dual update
theta_sequence = zeros(T,1);
loss = zeros(T,1);
theta = zeros(n+n*n,1);%primal variable, mu + L
y = zeros(2,1);%dual variable
pair_dist = zeros(n*n,1);
for i=1:n
    for j=1:n
        pair_dist((i-1)*n+j,:) = norm(training_data(i,:)-training_data(j,:));
    end
end
pair_dist_ordering = sort(pair_dist);
mu_0 = pair_dist_ordering(fix(n*n/2));% hyper-parameter for sampling w
sigma_0 = 3*var(pair_dist_ordering);%hyper-parameter for sampling w
Knn = zeros(n,n);%kernel matrix
%mini-batch trick
b = fix(0.1*n);%mini-batch
stoc_nabla_mu = zeros(n,1);
stoc_nabla_L = zeros(n*n,1);

for t=1:T
    %sample v, w
    logw = zeros(1,d+2);
    for i=1:d+2
        logw = mvnrnd(mu_0,sigma_0);
    end
    w = exp(logw);
    u_0 = w(1,:);
    u = w(2:d+1,:);
    tau = w(d+2,:);
    %compute the kernel matrice
    for i=1:n
        for j=1:n
            Knn(i,j) = u_0*exp(1/2*transpose(training_data(i,:)-training_data(j,:)) * diag(ones(1,d) ./ u) * (training_data(i,:)-training_data(j,:))+ tau);
        end
    end
    %define the auxilary matrix
    Knn_inv = inv(Knn);
    Knn_det = det(Knn);
    
    epsilon = mvnrnd(zeros(1,n),eye(n,n));%  is the co-variance matrix right?
    Q = [eye(n) kron(epsilon',eye(n))];%
    %% update the primal variable
    %compute the stochastic gradients w.r.p.t theta
    stoc_nabla_mu_L_1 = zeros(n+n*n,1);
    stoc_nabla_mu_L_2 = zeros(n+n*n,1);
    for j=1:b
        i = randi(n);
        %the first item of g
        mu_temp = [eye(n); zeros(n*n)]*theta;
        L_temp = [zeros(n); eye(n*n)]*theta;
        %stoc_nabla_mu_L_temp_1 = 1/(power(2*3.14159, n/2)*sqrt(Knn_det))*exp(-1/2*theta'*transpose(Q)*Knn_inv*Q*theta)*(-1/2)*Knn_inv*Q*theta*Q;
        stoc_nabla_mu_L_temp_1 = 1/(power(2*3.14159, n/2)*sqrt(Knn_det))*exp(-1/2*transpose(mu_temp + L_temp*epsilon)*Knn_inv*(mu_temp + L_temp*epsilon))*(-1/2)*Knn_inv*(mu_temp + L_temp*epsilon);
        stoc_nabla_mu_L_temp_1 = Q*stoc_nabla_mu_L_temp_1;
        stoc_nabla_mu_L_1 = stoc_nabla_mu_L_1 + stoc_nabla_mu_L_temp_1;
        %the second item of g
        L = reshape(theta(n+1:n+n*n), n,n);
        stoc_nabla_mu_L_temp_2 = ones(1,n)*( transpose(repmat(label,1,n+n*n) .* Q) ./ repmat((exp((repmat(label,1,n+n*n) .* Q) * theta)+1),1, n+n*n));
        stoc_nabla_mu_L_temp_2 = stoc_nabla_mu_L_temp_2' + [zeros(n,1); reshape(inv(L'), n*n,1)];
        stoc_nabla_mu_L_2 = stoc_nabla_mu_L_2 + stoc_nabla_mu_L_temp_2;
        
    end
    stoc_nabla_mu_L = 1/b*[stoc_nabla_mu_L_1 stoc_nabla_mu_L_2];
    
    % update rule 
    alpha = alpha_0/sqrt(T);
    theta = theta - alpha*stoc_nabla_mu_L*y;
    theta_sequence(t,:)  = theta;
    
    %% update the dual variable
    %compute the stochastic gradients w.r.p.t y
    mu_temp = [eye(n); zeros(n*n)]*theta;
    L_temp = [zeros(n); eye(n*n)]*theta;
    p_alpha_v_w = 1/(power((2*3.14159),n/2) * sqrt(det(Knn)))*exp(-1/2*transpose(mu_temp + L_temp*epsilon) * Knn_inv*(mu_temp + L_temp*epsilon));
    log_p_q = -1*one(1,n)*log(1+exp(-1*(repmat(label,1,n+n*n) .* Q)*theta)) + 1/2*(epsilon'*epsilon) + n/2*log(2*3.14159)+1/2*log(det(L_temp*L_temp'));
    g_v_w =  [p_alpha_v_w log_p_q];
    nabla_f_star_y = 1/([1 0]*y)*[1; 0];
    

    %update rule
    beta = beta_0/sqrt(T);
    y = y+beta*(g_v_w' - nabla_f_star_y);
    
    %% evaluate the loss
    mu_temp = [eye(n); zeros(n*n)]*theta;
    L_temp = [zeros(n); eye(n*n)]*theta;
    loss_temp = 1/(power((2*3.14159),n/2)*sqrt(det(Knn)))*exp(-1/2*mu_temp'*inv(Knn)*(mu_temp))    - ones(1,n)*log(1+exp(label .* mu_temp)) - 1/(power((2*3.14159),n/2)*sqrt(det(L*L')));
    loss(t,:) = loss_temp;
    
    
    
end


theta_optimal = mean(theta_sequence);
%% evaluate the loss
% mu_temp = [eye(n); zeros(n*n)]*theta_optimal;
% L_temp = [zeros(n); eye(n*n)]*theta_optimal;
% loss_temp = 1/(power((2*3.14159),n/2)*sqrt(det(Knn)))*exp(-1/2*mu_temp'*inv(Knn)*(mu_temp))    - ones(1,n)*log(1+exp(label .* mu_temp)) - 1/(power((2*3.14159),n/2)*sqrt(det(L*L')));
% loss_optimal = loss_temp;




%plot the convergence of the loss function 

