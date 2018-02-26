function [ Knn, Knn_inv,  log_Knn_det] = compute_kernel( data,n,d, mu_0)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
    Knn = zeros(n,n);%ARD kernel matrix
    w = exp(mu_0);
    u_0 = 1;
    u = w(2:d+1,:);
    tau = 1e-3;
    %compute the kernel matrice
    for i=1:n
        for j=1:n
                pair_diff = data(i,2:d+1) - data(j,2:d+1);
                Knn(i,j) = u_0*exp(-1/2*pair_diff * diag(1 ./ u) * pair_diff')+ tau;  
        end
    end
    %define the auxilary matrix
    Knn_inv = inv(Knn);
    log_Knn_det = logdet(Knn);
    
    
    
end

