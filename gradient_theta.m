function [ func,gradient ] = gradient_theta( theta, epsilon,  Knn_inv, training_label, n_test, p_alpha_v_w_expectation, w1,w2,b1,b2,Knn_logdet, n)
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here
mu_temp = theta(1:n,:);
L_temp = theta(n+1:n+n*n,:);
L_temp = reshape(L_temp,n,n);

y = w1 * (1 ./ exp(-1*(w2*epsilon+b2)))+b1;

nabla_g1_mu_L_temp = Knn_inv*(mu_temp+L_temp*epsilon)+transpose(transpose(mu_temp+L_temp*epsilon)*Knn_inv);
nabla_g1_theta_temp_mu = nabla_g1_mu_L_temp;
nabla_g1_theta_temp_L = repmat(nabla_g1_mu_L_temp,1,n) .* repmat(epsilon', n, 1);

temp = (-1/2)*[nabla_g1_theta_temp_mu; reshape(nabla_g1_theta_temp_L,n*n,1)];
log_nabla_g1_theta_temp = -n/2*log(2*3.14159)-1/2*Knn_logdet-1/2*transpose(mu_temp+L_temp*epsilon)*Knn_inv*(mu_temp+L_temp*epsilon);

nabla_g2_theta_I1 = [zeros(n_test,1); training_label ./ (1+exp(training_label .* (mu_temp(n_test+1:n,:)+L_temp(n_test+1:n,:)*epsilon))); reshape([zeros(n_test,n); repmat(training_label ./ (1+exp(training_label .* (mu_temp(n_test+1:n,:)+L_temp(n_test+1:n,:)*epsilon))),1,n) .* repmat(epsilon',n-n_test,1)], n*n,1)];
nabla_g2_theta_I2 = [zeros(n,1);reshape( inv(L_temp' ), n*n,1)];
nabla_g2_theta = nabla_g2_theta_I1 + nabla_g2_theta_I2;
%the gradient w.r.t theta
%nabla_g_y_theta_g = -1*exp(log(-y)+log_nabla_g1_theta_temp)*temp-nabla_g2_theta;
nabla_g1_theta_temp = exp(log_nabla_g1_theta_temp)*temp;
nabla_g_y_theta_g = 1/p_alpha_v_w_expectation*y*nabla_g1_theta_temp - nabla_g2_theta;
gradient = nabla_g_y_theta_g;
func = 1/p_alpha_v_w_expectation*y*exp(log_nabla_g1_theta_temp) - (-1*sum(log(1+exp(-1*(training_label .* (mu_temp(n_test+1:n,:) + L_temp(n_test+1:n,:)*epsilon)))))) +(-n/2*log(2*3.14159)-1/2*logdet(L_temp*L_temp')-1/2*(epsilon'*epsilon));

end

