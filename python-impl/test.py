import autograd.numpy as np  # Thinly-wrapped numpy
import scipy.io as scio
from autograd import grad  # The only autograd function you may ever need


def tanh(x):
    y = np.exp(-2.0 * x[0])
    return (1.0 - y) / (1.0 + y) + x[1]

# define the FIRST item of g: P(v | w)
def P_v_w(theta):
    mu_temp = theta[0:n]
    L_temp = theta[n:n + n * n]
    L_temp = np.reshape(L_temp, (n, n),
                        order="F")  # reshape by the order of ROW! "F" means the Forturn order, that is used in Matlab
    temp = mu_temp + np.dot(L_temp, epsilon)
    temp_inner = np.dot(np.dot(np.transpose(temp), Knn_inv), temp)
    result = 1.0 / (np.power(2 * np.pi, n / 2) * np.sqrt(Knn_det)) * np.exp(-1.0 / 2 * temp_inner)
    #result = 1.0 / (np.power(2 * np.pi, n / 2) * np.sqrt(Knn_det)) * np.exp(-1.0 / 2 * temp_inner.tolist())
    return result


def Evaluate_P_v_w(theta):
    mu_temp = theta[0:n]
    L_temp = theta[n:n + n * n]
    L_temp = np.reshape(L_temp, (n, n),
                        order="F")  # reshape by the order of ROW! "F" means the Forturn order, that is used in Matlab
    temp = mu_temp + np.dot(L_temp, epsilon)
    temp_inner = np.dot(np.dot(np.transpose(temp), Knn_inv), temp)
    temp_inner2 = -1.0 / 2 * temp_inner[0]
    result = 1.0 / (np.power(2 * np.pi, n / 2) * np.sqrt(Knn_det)) * np.exp(temp_inner2)
    return result


# define the SECOND item of g: Log_p_q
def Log_p_q(theta):
    mu_temp = theta[0:n]
    L_temp = theta[n:n + n * n]
    L_temp = np.reshape(L_temp, (n, n),
                        order="F")  # reshape by the order of ROW! "F" means the Forturn order, that is used in Matlab
    log_temp1 = 0
    for i in range(1, n):
        log_temp1 = log_temp1 - 1 * np.log(1 + np.exp(-1 * label[i] * (mu_temp[i] * np.dot(L_temp[i], epsilon))))
    log_temp2 = 1 / 2 * np.dot(np.transpose(epsilon), epsilon) + (
        n / 2 * np.log(2 * np.pi) + 1 / 2 * np.log(Knn_det))
    return log_temp1 + log_temp2


# define the congugate of f
def F_star(s):
    a = [1, 0]
    return 1 + np.log(a * s)


# define the loss
def Evaluate_loss(theta):
    mu_temp = theta[0:n]
    L_temp = theta[n:n + n * n]
    L_temp = np.reshape(L_temp, (n, n),
                        order="F")  # reshape by the order of ROW! "F" means the Forturn order, that is used in Matlab
    loss_temp_1 = -1.0*n/2.0*np.log(2*np.pi) - 1.0/2*np.log(Knn_det) - 1.0 / 2 * np.dot(np.dot(np.transpose(mu_temp), Knn_inv), mu_temp)
    loss_temp_2_p = 0
    for i in range(0, n):
        temp = -1 * np.log(1 + np.exp(-1 * label[i] * mu_temp[i]))
        loss_temp_2_p = loss_temp_2_p + temp
    loss_temp_2_q = -1.0*n/2*np.log(2*np.pi)-1.0/2*np.log(np.linalg.det(np.dot(L_temp, np.transpose(L_temp)))) - 1.0/2*np.dot(np.transpose(epsilon), epsilon)
    loss_temp = loss_temp_1 + loss_temp_2_p - loss_temp_2_q
    return loss_temp


# load data
data = scio.loadmat(
    "/Users/yawei/Documents/MATLAB/simulation based algorithm test library/simulation_based_machine_learning_library/dataset/heart/test_heart.mat")
data = data['yy']
label = data[:, 0]
[n, d] = np.shape(data)
training_data = np.mat(data[:, 1:d])
[n, d] = np.shape(training_data)

# initialize parameters
T = 10
alpha_0 = 1e-5  # learning rate for the primal update
beta_0 = 1e-5  # learning rate for the dual update
theta_sum_old = np.mat(np.zeros((n + n * n, 1)))
loss = np.mat(np.zeros((T, 1)))
theta = np.mat(np.ones((n + n * n, 1)))  # primal variable, mu + L
y = np.mat(np.ones((2, 1)))  # dual variable
pair_dist = np.mat(np.zeros((n * n, 1)))
for i in range(1, n):
    for j in range(1, n):
        if i == j:
            continue
        pair_dist[(i - 1) * n + j, :] = np.log(np.linalg.norm(training_data[i, :] - training_data[j, :]))

pair_dist_ordering = np.sort(pair_dist, axis=0)
mu_0 = pair_dist_ordering[n * (n + 1) / 2, :]  # hyper-parameter for sampling w
sigma_0 = np.mat(3 * np.var(pair_dist_ordering))  # hyper-parameter for sampling w, initialized by 3

for t in range(0, T):
    #i = np.random.random_integers(1, high=n, size=1)
    i=t
    print "i=", i
    Knn = np.mat(np.zeros((n, n)))  # kernel matrix
    # sample v, w
    logw = np.random.normal(mu_0, sigma_0, (d + 2, 1))
    w = np.exp(logw)
    # u_0 = w[0, :]
    u_0 = 1
    u = w[1:d + 1, 0]  # pick w's row from 1 to d, the d+1-th row is not picked! [different from MATLAB]
    # tau = w[d + 1, :]
    tau = 1e-6 # 1e-6 in default
    # compute the kernel matrice
    for i in range(0, n):
        for j in range(0, n):
            pair_diff = np.mat(training_data[i, :] - training_data[j, :])
            u_temp = np.ones((d, 1)) / u
            diag_temp = np.diag(np.asarray(u_temp))
            Knn[i, j] = u_0 * np.exp(-1.0 / 2 * pair_diff * diag_temp * pair_diff.T + tau)
    # define the auxiliary matrix
    Knn_inv = np.linalg.inv(Knn)
    Knn_det = np.linalg.det(Knn)
    if Knn_det <= 0:
        print "u=[", u, "]"
        print " ERROR: non-positive kernel matrix"
        break
    epsilon = np.mat(np.random.randn(n, 1))

    stoc_nabla_mu_L_temp_1 = grad(P_v_w)
    stoc_nabla_mu_L_1 = stoc_nabla_mu_L_temp_1(theta)
    # the second item of g
    stoc_nabla_mu_L_temp_2 = grad(Log_p_q)
    stoc_nabla_mu_L_2 = stoc_nabla_mu_L_temp_2(theta)

    stoc_nabla_mu_L = np.hstack((stoc_nabla_mu_L_1, stoc_nabla_mu_L_2))

    # update rule
    alpha = alpha_0 / np.sqrt(T)
    theta = theta - alpha * np.dot(stoc_nabla_mu_L, y)
    theta_sum_old = theta_sum_old + theta
    theta_average = 1.0 / (t + 1) * theta_sum_old

    # update the dual variable
    # compute the stochastic gradients w.r.p.t y
    mu_temp = theta[0:n]
    L_temp = theta[n:n + n * n]
    L_temp = np.reshape(L_temp, (n, n),
                        order="F")  # reshape by the order of ROW! "F" means the Forturn order, that is used in Matlab

    p_alpha_v_w = Evaluate_P_v_w(theta)
    log_p_q = Log_p_q(theta)
    log_p_q = log_p_q[0]
    g_v_w = np.hstack((p_alpha_v_w, log_p_q))
    nabla_f_star_y = np.array([1.0 / y[0], 0])

    # update rule
    beta = beta_0 / np.sqrt(T)
    sss = np.transpose(g_v_w - nabla_f_star_y)
    y = y + beta * (np.transpose(g_v_w - nabla_f_star_y))

    # evaluate the loss
    # mu_temp = np.dot(np.hstack((np.eye(n, n), np.zeros((n, n * n)))), theta)
    # L_temp = np.dot(np.hstack((np.zeros((n * n, n)), np.eye(n * n, n * n))), theta)
    # L_temp = np.reshape(L_temp, (n, n),
    #                    order="F")  # reshape by the order of ROW! "F" means the Forturn order, that is used in Matlab
    loss[t, :] = Evaluate_loss(theta)
