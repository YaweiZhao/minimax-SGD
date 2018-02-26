clear; close all;
rng('default');
[data, label, training_data, test_data, training_label, test_label, n, d, n_train, n_test ] = prepare_data();

[T, train_loss, test_loss, num_nodes_nn, y_new_plot, w1, w2, b1, b2, mu_0, sigma_0] = initialize_parameters(data, n,d);

%SGD optimization method
alpha_0 = 1e-1;% learning rate for the primal update
%initialize mu and L
theta =[zeros(n,1); reshape(eye(n),n*n,1)];% use constant to initialize


    
n_w = 30;
for t=1:T
    disp(t);
    temp = zeros(n+n*n,1);
    for i_w = 1:n_w
        mu_sample = randn()*mu_0;
        [ Knn, Knn_inv,  log_Knn_det] = compute_kernel( data,n,d, mu_sample);
        
        epsilon = randn(n,1);
        %remember: L is a low-triangle matrix
        mu_temp = theta(1:n,:);
        L_temp = theta(n+1:n+n*n,:);
        L_temp = reshape(L_temp,n,n);
        L_temp = tril(L_temp);
        theta(n+1:n+n*n,:) = reshape(L_temp,n*n,1);
        
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
        
        temp = temp +  (-1/2)*[nabla_g1_theta_temp_mu; reshape(nabla_g1_theta_temp_L,n*n,1)];
    end
    temp = temp / n_w;
    
    nabla_g2_theta_I1 = [zeros(n_test,1); training_label ./ (1+exp(training_label .* (mu_temp(n_test+1:n,:)+L_temp(n_test+1:n,:)*epsilon))); reshape([zeros(n_test,n); repmat(training_label ./ (1+exp(training_label .* (mu_temp(n_test+1:n,:)+L_temp(n_test+1:n,:)*epsilon))),1,n) .* repmat(epsilon',n-n_test,1)], n*n,1)];
    nabla_g2_theta_I2 = [zeros(n,1);reshape( inv(L_temp'), n*n,1)];
    nabla_g2_theta = nabla_g2_theta_I1 + nabla_g2_theta_I2;
    %the gradient w.r.t theta
    nabla_g1_theta_temp = temp;
    nabla_g_y_theta_g = -1*(nabla_g1_theta_temp + nabla_g2_theta);
    
    % update rule for the primal variable: theta
    % set the primal step size via SGD method
    alpha = alpha_0 /sqrt(t);
    theta = theta - alpha*nabla_g_y_theta_g;

    
    %% evaluate loss: for train data and test data
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
 save('w1.mat','w1')
 save('w2.mat','w2')
 save('b2.mat','b2')
 save('b1.mat','b1')
 save('theta.mat','theta')

%save('u_save.mat','u_save');
save('train_loss.mat','train_loss');
save('test_loss.mat','test_loss');
