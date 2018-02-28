function [ train_loss2, test_loss2 ] = conditional_gauss( training_data,test_data,training_label, test_label,n_train, n_test,d, mu_0, u_0, tau)
%UNTITLED11 Summary of this function goes here
%   Detailed explanation goes here
%Laplace inference
latent_variables = Laplace_inference(training_data,  training_label,n_train,d, mu_0, u_0, tau);
mu_train = latent_variables;%only use training data
%kernel matrix for training dataset
Knn_train_train = zeros(n_train,n_train);
    for i=1:n_train
        for j=1:n_train
                pair_diff = training_data(i,:) - training_data(j,:);
                Knn_train_train(i,j) = exp(-1/2*pair_diff * diag(1 ./ (ones(d,1)*exp(mu_0))) * pair_diff')+ tau;  
        end
    end
%kernel matrix for test dataset x  training dataset
Knn_test_train = zeros(n_test,n_train);
    for i=1:n_test
        for j=1:n_train
                pair_diff = test_data(i,:) - training_data(j,:);
                Knn_test_train(i,j) = exp(-1/2*pair_diff * diag(1 ./ (ones(d,1)*exp(mu_0))) * pair_diff')+ tau;  
        end
    end
    
    
    mu_test2 = Knn_test_train*inv(Knn_train_train)*mu_train;

        %% evaluate the train loss
    train_loss_temp = 0;
    for i=1:n_train
        temp = training_label(i,:)*mu_train(i,:);
        train_loss_temp = train_loss_temp - log(1+exp(-temp));
    end
    train_loss2 = train_loss_temp/n_train;
    disp('inference via Laplace, train');
disp(train_loss2);




    %% evaluate the test loss
    test_loss_temp = 0;
    for i=1:n_test
        temp = test_label(i,:)*mu_test2(i,:);
        test_loss_temp = test_loss_temp - log(1+exp(-temp));
    end
    test_loss2 = test_loss_temp/n_test;
disp('inference via Laplace, test');
disp(test_loss2);
end

