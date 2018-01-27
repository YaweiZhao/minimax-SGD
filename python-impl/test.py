import autograd.numpy as np  # Thinly-wrapped numpy
import scipy.io as scio
from autograd import grad    # The only autograd function you may ever need


def tanh(x):
    y = np.exp(-2.0 * x)
    return (1.0 - y) / (1.0 + y)

grad_tanh = grad(tanh)       # Obtain its gradient function
print grad_tanh(1.0)               # Evaluate the gradient at x = 1.0


# define the FIRST item of g: P(v | w)
def P_v_w(mu, L, epsilon, Knn, n):
    temp = mu + L*epsilon;
    return 1.0/(np.power(2*np.pi, n/2) * np.sqrt(np.linalg.det(Knn))*np.exp(-1/2*np.transpose(temp)*np.linalg.inv(Knn)*temp)

# define the SECOND item of g: Log_p_q
def Log_p_q(mu, L, epsilon, Knn, n):
    log_temp1 = 0
    for i in range(1,n):
        log_temp1 = log_temp1 -1*np.log(1+np.exp(-1*label[i,:]*(mu[i,:]*L[i,:]*epsilon))))
    log_temp2 = 1/2*np.transpose(epsilon)*epsilon + (n/2*np.log(2*np.pi) + 1/2*np.log(np.linalg.det(Knn)))
    return log_temp1 + log_temp2

# define the congugate of f
def F_star(s):
    a = [1 0]
    return 1+np.log(a*s)

#load data
data = scio.loadmat("/Users/yawei/Documents/MATLAB/simulation based algorithm test library/simulation_based_machine_learning_library/dataset/heart/test_heart.mat");
data = data['yy']
label = data[:,1]
[n,d] = np.shape(data)
training_data = data[2:d]
[n,d] = np.shape(training_data)

#initialize parameters
T = 10
alpha_0 = 1e-5 # learning rate for the primal update
beta_0 = 1e-5 # learning rate for the dual update
theta_sequence = np.zeros((T,n+n*n))
loss = np.zeros((T,1))
theta = np.ones((n+n*n,1)) # primal variable, mu + L
y = np.ones((2,1)) # dual variable
pair_dist = np.zeros((n*n,1))
for i in range(1, n):
    for j in range(1, n):
        pair_dist[(i-1)*n+j,:] = np.linalg.norm(training_data[i,:]-training_data[j,:])

pair_dist_ordering = np.sort(pair_dist)
mu_0 = pair_dist_ordering(np.floor(n*n/2)) # hyper-parameter for sampling w
sigma_0 = 3*var(pair_dist_ordering) # hyper-parameter for sampling w
Knn = np.zeros((n,n)) # kernel matrix
# mini-batch trick
b = 1 # mini-batch
stoc_nabla_mu = np.zeros((n,1))
stoc_nabla_L = np.zeros((n*n,1)

for t in range(1,T):
    # sample v, w
    logw = np.zeros((d + 2, 1))
    for i in range( 1, d + 2):
        logw[i,:] = np.random.multivariate_normal(mu_0, sigma_0)
    w = np.exp(logw)
    u_0 = w[1,:]
    u = w[2:d + 1,:]
    tau = w[d + 2,:]
#compute the kernel matrice
for i in range(1,n):
    for j in range(1,n):
        pair_diff = training_data[i,:]-training_data[j,:]
        Knn[i, j] = u_0 * np.exp(1 / 2 * pair_diff * np.diag(np.ones[d, 1]. / u) * np.transpose(pair_diff)+ tau)

# define the auxilary matrix
Knn_inv = np.linalg.inv(Knn)
Knn_det = np.linalg.det(Knn)
epsilon = np.transpose(np.random.multivariate_normal(np.zeros((1, n)), np.eye((n, n)))

stoc_nabla_mu_L_2 = zeros(n + n * n, 1);
for j in range(1,b):
    i = np.randi(n)
    # the first item of g
    mu_temp = [eye(n, n) zeros(n, n * n)] * theta;
    L_temp = [zeros(n * n, n) eye(n * n, n * n)] * theta;
    L_temp = reshape(L_temp, n, n);
    stoc_nabla_mu_L_temp_1 =
    stoc_nabla_mu_L_1 = stoc_nabla_mu_L_1 + stoc_nabla_mu_L_temp_1


    # the second item of g
    stoc_nabla_mu_L_temp_2 =
    stoc_nabla_mu_L_2 = stoc_nabla_mu_L_2 + stoc_nabla_mu_L_temp_2

stoc_nabla_mu_L = 1 / b * [stoc_nabla_mu_L_1 stoc_nabla_mu_L_2]

# update rule
alpha = alpha_0 / np.sqrt(T)
theta = theta - alpha * stoc_nabla_mu_L * y
theta_sequence[t,:]  = theta

# update the dual variable
# compute the stochastic gradients w.r.p.t y
mu_temp = [eye(n, n) zeros(n, n * n)] * theta;
L_temp = [zeros(n * n, n) eye(n * n, n * n)] * theta;
L_temp = reshape(L_temp, n, n);
p_alpha_v_w = 1 / (power((2 * 3.14159), n / 2) * sqrt(Knn_det)) * exp(
    -1 / 2 * transpose(mu_temp + L_temp * epsilon) * Knn_inv * (mu_temp + L_temp * epsilon));
log_p_q = -1 * ones(1, n) * log(1 + exp(-1 * (repmat(label, 1, n + n * n). * Q) * theta)) + 1 / 2 * (epsilon
'*epsilon) + n/2*log(2*3.14159)+1/2*log(det(L_temp*L_temp'));
g_v_w = [p_alpha_v_w log_p_q];
nabla_f_star_y = 1 / ([1 0] * y) * [1 0];

# update rule
beta = beta_0 / sqrt(T);
y = y + beta * (g_v_w
' - nabla_f_star_y);

# evaluate the loss
mu_temp = [eye(n, n) zeros(n, n * n)] * theta;
L_temp = [zeros(n * n, n) eye(n * n, n * n)] * theta;
L_temp = reshape(L_temp, n, n);
loss_temp = 1 / (power((2 * 3.14159), n / 2) * sqrt(det(Knn))) * exp(-1 / 2 * mu_temp
'*Knn_inv*(mu_temp))    - ones(1,n)*log(ones(n,1)+exp(-1*label .* (mu_temp+L_temp*epsilon))) + 1/2*(epsilon' * epsilon) +n / 2 * log(
    2 * 3.14159) + 1 / 2 * log(det(L_temp * L_temp
'));
loss(t,:) = loss_temp;

end

theta_optimal = mean(theta_sequence);