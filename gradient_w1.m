function [ output_args ] = gradient_w1( w1, theta,p_alpha_v_w_expectation, p_alpha_v_w, y, n )
%UNTITLED9 Summary of this function goes here
%   Detailed explanation goes here
mu_temp = theta(1:n,:);
    L_temp = theta(n+1:n+n*n,:);
    L_temp = reshape(L_temp,n,n);
    
nabla_g_y_g1_temp =  1 ./ (1+exp(-1*(w2*epsilon+b2)));
    essential_temp = 1/p_alpha_v_w_expectation*p_alpha_v_w+1/y;
    nabla_g_y_w1 = nabla_g_y_g1_temp' * essential_temp;% 1 x num_nodes_nn
    

end

