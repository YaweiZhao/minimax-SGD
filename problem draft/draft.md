# Problem
## Notations

The objective loss function is:

$L(\theta) = -\frac{n}{2}\log (2\pi) - \frac{1}{2}\log |K_{nn}|+\frac{1}{2}(\mu+L\epsilon)^T K_{nn}^{-1}(\mu+L\epsilon) \\ + (-\sum\limits_{i=1}^{n-n_{test}}\log(1+\exp^{-label(i)(\mu_i+L_i\epsilon)})-(-\frac{n}{2}(\log(2\pi))-\frac{1}{2}\log |LL^T|-\frac{1}{2}\epsilon^T\epsilon)$.

Loss function: $L(\theta) = \log g_1 + g_2$ where $g =[g_1,g_2] =  [P_{\alpha}(v|w), \log \frac{P(D|v)}{q(v|\theta)} ]$. (Section 4.1, Eq. (9))

$P_{\alpha}(v|w) = \frac{1}{(2\pi)^{n/2}|K_{nn}|^{1/2}}\exp(\frac{1}{2}(\mu + L\epsilon)^T K_{nn}^{-1}(\mu+L\epsilon))$. (Section 4.1, Eq. (9))

$P(D|v) = \Pi_{i=1}^n \frac{1}{1+\exp^{-label(i)(\mu_i+L_i\epsilon)}}$.  (Section 4.1, Eq. (9))

$q(v|\theta) = \frac{1}{(2\pi)^{n/2}|LL^{T}|^{1/2}}\exp(-1/2 \epsilon^T\epsilon)​$. (Section 4.1, Eq. (9))

$\theta = [\mu, vec(L)]$

## Update of primal variables

$\theta = \theta - \alpha \langle \nabla g(\theta), y\rangle$, and $\nabla g(\theta) = [\frac{\partial P_{\alpha}(v|w)}{\partial \theta}, \frac{\partial \log \frac{P(D|v)}{q(v|\theta)}}{\partial \theta}]$

$\frac{\partial P_{\alpha}(v|w)}{\partial \theta} = \frac{1}{(2\pi)^{n/2}|K_{nn}|^{1/2}}\exp(-\frac{1}{2}(\mu + L\epsilon)^T K_{nn}^{-1}(\mu+L\epsilon)) K_{nn}^{-1}(\mu+L\epsilon)\frac{\partial \mu+L\epsilon}{\partial \theta}$ Here, $\exp(-\frac{1}{2}(\mu + L\epsilon)^T K_{nn}^{-1}(\mu+L\epsilon))$ is very small. The reason is that $\frac{1}{2}(\mu + L\epsilon)^T K_{nn}^{-1}(\mu+L\epsilon)$ is large ($>10000$). Therefore, when I begin to compute the gradient  of $P_{\alpha}(v|w)$  with respect to $\theta=[\mu, vec(L)]$, I find that the gradient is very small (see the figure).  

![image](file:///Users/yawei/Documents/source code/minimax-SGD/problem draft/mu_L_1.png)

The second item of $g$ consist of $P(D|v)$ and $q(v|\theta)$. The gradient of $P(D|v)$ is computed as following codes:

```matlab
%the second item of g
    stoc_nabla_mu_L_temp_2 = zeros(n+n*n,1);
    for j=1:n
        if j<=n_test
            continue;% During training, the test data is discarded due to lack of labels.
        end
        stoc_nabla_mu_L_temp_2 = stoc_nabla_mu_L_temp_2 + (label(j)*transpose(Q(j,:)))/(1+exp(label(j)*Q(j,:)*theta));
    end
```

Its gradeint with respect to $\mu_{test data}$ is $0$ because the labels of test data is not used during the training of parameters. 

The gradient of $q(v|\theta)$ with respect to $\mu$ is $0$. Because it is a function with respect to $L$. 


Therefore,  during training iterations, the $\mu$ corresponding to the test data (dimensions from $1$ to $10$) do not have any changes:

![image](file:///Users/yawei/Documents/source code/minimax-SGD/problem draft/theta_sequence.png)







## Update of dual variables

$y = y + \beta ( g(\theta) -  \nabla f^\ast (y) )$















