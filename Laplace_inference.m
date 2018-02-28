function [f, log_q_y_X ] = Laplace_inference(training_data,  training_label,n_train,d,  mu_0, u_0, tau)
%UNTITLED10 Summary of this function goes here
%   Detailed explanation goes here

%initialize covariance matrix, 
kernel_temp = (training_data .* repmat(transpose(1 ./ (mu_0*ones(d,1))),n_train,1)) * training_data';
kernel_diag = diag(kernel_temp);
Knn_temp = -2*kernel_temp + ones(n_train,n_train)*diag(kernel_diag) +diag( kernel_diag)*ones(n_train,n_train);
K = u_0*exp(-1/2*Knn_temp)+tau;


%f=zeros(n_train,1);
f =training_label;
for k=1:500
    disp(k);
    pi_i = 1 ./ (1+exp(-1*training_label .* f));
    W = diag(-1*(-1*pi_i .* (1 - pi_i)));
    W_root = sqrtm(W);
    L = chol(eye(n_train) + W_root*K*W_root);
    b = W*f + ((training_label + ones(n_train,1))/2 - pi_i);
    a = b - W_root*L'\(L\(W_root*K*b));
    f = K*a;
end
log_q_y_X = -1/2*a'*f + (-1*log(1+exp(-1*training_label .* f)));



end

