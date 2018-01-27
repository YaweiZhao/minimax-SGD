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

training_data = data[2:d]
#initialize parameters



for i in [1,T]:
