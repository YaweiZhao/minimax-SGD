clear;
%load data



%% initialize variables
T = 10;
alpha = 1e-2;% learning rate for the primal update
beta = 1e-2;%learning rate for the dual update
theta_sequence = zeros(T,1);
theta = 0;




for t=1:T
    
    
    %% update the primal variable
    % sample 
    
    % compute the stochastic gradient
    
    % update rule 
    
    
    %% update the dual variable
    
    
    
    
    
    
    theta_sequence(t,:)  = theta;
    %% evaluate the loss
    % sample 
    
    
    % compute the stochastic gradient
    
    
    %update rule
    
    
    
end


theta_optimal = mean(theta_sequence);





%plot the convergence of the loss function 

