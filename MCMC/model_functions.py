
# coding: utf-8

# ### This notebook has all the functions for the Lorenz63 model.
# 
# 

# In[1]:

# Calling libraries:
from __future__ import division
get_ipython().magic('matplotlib inline')
import numpy as np
import time
from scipy.stats import norm, uniform, bernoulli
from scipy.linalg import sqrtm 
import math 
from resampling_functions import resample, systematic_resampling
import scipy
import multiprocessing as mp
import parmap


# * Here $W$ is $N$ iid copies of $N(0,1)$.

# In[2]:

def propagate_Lorenz63(X, W, dt, delta, theta) :
    """
    usual: 3D Lorenz model + multiplicative noise
    compute one step forward, usual Euler-Maruyama discretization
    """
    sigma = theta[0]
    rho   = theta[1]
    beta  = theta[2]
    noise_intensity = theta[3]
    #ODE forward + noise
    N = np.shape(X)[1]
    sqdt = np.sqrt(dt)

    for i in range( int(delta/dt) ) :
        xx = np.zeros(( 3, N ))
        xx[0,:] = X[0,:] + dt*sigma*(X[1,:] - X[0,:]) + noise_intensity*X[0,:]*sqdt*W[i,:]
        xx[1,:] = X[1,:] + dt*(X[0,:]*(rho - X[2,:]) - X[1,:]) + noise_intensity*X[1,:]*sqdt*W[i,:]
        xx[2,:] = X[2,:] + dt*(X[0,:]*X[1,:] - beta*X[2,:]) + noise_intensity*X[2,:]*sqdt*W[i,:]
        X [:,:] = xx[:,:]
    
    return X


# #### Simulate data:
# 
# * Define a function to simulate observations.
# 
# * $\theta = (\sigma, \rho, \beta, \sigma_{\text{latent}})$.

# In[3]:

def simulate_observations(x_0, T, dt, delta, H, R, theta) : 
    """
    generate a sequence of observations x_0 : initialization
    T = number of observations
    delta = time between observations
    """
    m  = np.shape(H)[0]
    y  = np.zeros(( m, T ))
    X  = np.zeros(( 3, 1 ))
    trajectory = np.zeros(( 3, T ))
    X[:,0] = x_0
    
    for t in range(T) :
        W = np.random.randn(int(delta/dt),1)
        X = propagate_Lorenz63(X, W, dt, delta, theta)
        trajectory[:,t] = X[:,0]
        y[:,t] = np.dot(H, X[:,0]) + np.random.multivariate_normal(np.zeros(m), R, 1)
                        
    return y, trajectory


# * First define $g$, the error functiom (that is, the density of $Y_t$ given $X_t$).

# In[4]:

def g(y, X, H, R) :
    
    E = y - np.transpose(np.dot(H,X))          # denotes error
    R_inv = np.linalg.inv(R)
    
    return np.exp(-0.5*np.diag(np.dot(E, np.dot( R_inv, np.transpose(E) )))) 


# ### Bootstrap particle filter.

# In[5]:

def bootstrap_PF(n_particles, x_0, theta, y, dt, delta, g, H, R) :
    
    scipy.random.seed()
    np.random.seed()
    
    T  = np.shape(y)[1]
    n  = len(x_0) 
    particles = np.zeros(( n, n_particles, T+1 )) 
    weights   = [1/n_particles]*n_particles 
    log_NC    = np.zeros(T+1) 
    for particle in range(n_particles) :
        particles[:, particle, 0] = x_0
        
    # Run the coupled particle filter:
    start_time = time.clock()
    for t in range(T):
        
        W = np.random.randn(int(delta/dt),n_particles)
        # mutate:
        particles[:,:,t+1] = propagate_Lorenz63(particles[:,:,t], W, dt, delta, theta)
        weights           *= g(y[:,t], particles[:,:,t+1], H, R)
        log_NC[t+1]        = log_NC[t] + np.log(np.sum(weights))
        weights           /= np.sum(weights)
        
        # Resample
        if 1/np.sum(weights**2) < 0.5*n_particles :
            u = uniform.rvs(0,1,1)
            particles[:,:,t+1] = particles[:, systematic_resampling(weights, n_particles, u).astype(int), t+1 ]
            weights = [1/n_particles]*n_particles
        
    return log_NC, particles


# ### Mutate using same noise:

# In[1]:

def mutate_coupled_PF(particles, theta_values, weights, log_NC, y, dt, delta, g, H, R) :
    
    scipy.random.seed()
    np.random.seed()
    
    n, n_particles, _ = np.shape(particles)
    delta_log_NC = np.zeros(2)
    
    # propagate forward:
    W = np.random.randn(int(delta/dt), n_particles) # The common noise variables used 
    for i in range(2) :
        particles[:,:,i] = propagate_Lorenz63(particles[:,:,i], W, dt, delta, theta_values[:,i])
            
    # update weights and log-likelihood:
    for i in range(2) :
        weights[:,i]   *= g(y, particles[:,:,i], H, R)
        delta_log_NC[i] = np.log(np.sum(weights[:,i]))
        weights[:,i]   /= np.sum(weights[:,i])    
        
    del W
        
    return particles, weights, delta_log_NC


# ### Coupled resampling function.
# 
# * Perform adaptive resampling based on ESS. Resample if either of the ESS's (for the two trajectories) falls below a threshold.

# In[7]:

def perform_coupled_resampling(particles, weights, coupling_method, 
                               clambda=50, n_Sinkhorn=50, q=50, uv_threshold=10**10) : 
    
    scipy.random.seed()
    np.random.seed()
    
    n_particles = np.shape(particles)[1]
    
    c = 0.5
    if (1/np.sum(weights[:,0]**2) > c*n_particles)*(1/np.sum(weights[:,1]**2) > c*n_particles) == 0 : 
        U1, U2 = uniform.rvs(0,1,2)
        R = resample(np.transpose(particles[:,:,0]), np.transpose(particles[:,:,1]), 
                     weights[:,0], weights[:,1], U1, U2, 
                     coupling_method, clambda, n_Sinkhorn, q, uv_threshold)
        
        P1, P2, weights[:,0], weights[:,1] = R
        particles[:,:,0] = np.transpose(P1)
        particles[:,:,1] = np.transpose(P2)
                                     
    return particles, weights


# ### Multilevel particle filter

# In[ ]:

def mutate_MLPF(particles, theta, weights, log_NC, y, dt, delta, g, H, R) :
    
    scipy.random.seed()
    np.random.seed()
    
    n, n_particles, _ = np.shape(particles)
    delta_log_NC = np.zeros(2)
    
    # propagate forward:
    W_f = np.random.randn(int(2*delta/dt), n_particles) # The common noise variables used 
    W_c = np.sum(np.reshape(W_f, (int(delta/dt),2,n_particles)),axis=1)/np.sqrt(2)
    particles[:,:,0] = propagate_Lorenz63(particles[:,:,0], W_f, dt/2, delta, theta)
    particles[:,:,1] = propagate_Lorenz63(particles[:,:,1], W_c, dt  , delta, theta)
            
    # update weights and log-likelihood:
    for i in range(2) :
        weights[:,i]   *= g(y, particles[:,:,i], H, R)
        delta_log_NC[i] = np.log(np.sum(weights[:,i]))
        weights[:,i]   /= np.sum(weights[:,i])    
        
    del W_f, W_c
        
    return particles, weights, delta_log_NC


# In[1]:

def MLPF(n_particles, theta, y, x_0, dt, delta, g, H, R, test_function,
         coupling_method, clambda=50, n_Sinkhorn=50, q=50, uv_threshold=10**10) : 
                  
    T = np.shape(y)[1] 
    n = len(x_0) 
    
    particles = np.zeros(( n, n_particles, 2 )) 
    weights   = np.multiply( np.ones(( n_particles, 2 )), 1/n_particles ) 
    log_NC    = np.zeros(( 2, T+1 ))  
    for particle in range(n_particles) :
        for i in range(2) :
            particles[:, particle, i] = x_0
    test_fn_values = np.zeros(( 2, T ))
    running_times = np.zeros(T)
            
    # Running the particle filter: 
    
    n_delta = delta/dt
    start_time = time.clock()
    for t in range(T) :
        
        # Mutate:
        M = mutate_MLPF(particles, theta, weights, log_NC[:,t], y[:,t], dt, delta, g, H, R)
        
        particles, weights = M[0], M[1]
        log_NC[:,t+1] = log_NC[:,t] + M[2]

        # Resample:
        particles, weights = perform_coupled_resampling(particles, weights, coupling_method, 
                                                        clambda, n_Sinkhorn, q, uv_threshold)
        running_times[t] = time.clock() - start_time
        for i in range(2) :
            test_fn_values[i,t] = np.sum(test_function(particles[:,:,i])*weights[:,i])
        
    del particles, weights
        
    return log_NC[:,1:], test_fn_values, running_times


# In[ ]:




# ### Coupled MCMC.

# ### Correlated psuedo-marginal.
# 
# * Implement the correlated psuedo-marginal method.
# 
# * First, define a function to drive the two particle systems forward given (correlated) values of the auxiliary variables.

# In[9]:

def mutate_MCMC_cpm(particles, weights, log_NC, W_values, theta_values, y, dt, delta, g, H, R) :
    
    n, n_particles, _ = np.shape(particles)
    delta_log_NC = np.zeros(2)
    
    n_delta = delta/dt
    
    for i in range(2) :     
        #W = W_values[:,j,i]
        theta = theta_values[:,i]
        particles[:,:,i] = propagate_Lorenz63(particles[:,:,i], W_values[:,:,i], dt, delta, theta)
            
    for i in range(2) :
        weights[:,i]   *= g(y, particles[:,:,i], H, R)
        delta_log_NC[i] = np.log(np.sum(weights[:,i]))
        weights[:,i]   /= np.sum(weights[:,i])         
        
    return particles, weights, delta_log_NC


# * Then define a function to perform a coupled re-sampling given the two (correlated) $U(0,1)$ random variables used for resampling.
# 
# * Do adaptive resampling. resample if either of the ESS's for the two trajectories falls below a threhold.

# In[10]:

def coupled_resampling_MCMC_cpm(particles, weights, coupling_method, U1, U2, 
                                clambda=50, n_Sinkhorn=50, q=50, uv_threshold=10**10) : 
    
    n_particles = np.shape(particles)[1]
    
    c = 0.5
    lambda_final = -100
    if (1/np.sum(weights[:,0]**2) > c*n_particles)*(1/np.sum(weights[:,1]**2) > c*n_particles) == 0 : 
        R = resample(np.transpose(particles[:,:,0]), np.transpose(particles[:,:,1]), 
                     weights[:,0], weights[:,1], U1, U2, 
                     coupling_method, clambda, n_Sinkhorn, q, uv_threshold)
        
        P1, P2, weights[:,0], weights[:,1] = R  
        particles[:,:,0] = np.transpose(P1)
        particles[:,:,1] = np.transpose(P2)
                                           
    return particles, weights


# * Then, given values of (correlated) auxiliary variables and parameters, compute the log-likelihoods at the two parameter values using these correlated auxiliary  variables.

# In[11]:

def likelihood_PF_cpm(n_particles, theta_values, W_values, y, x_0, dt, delta, g, H, R,
                      coupling_method, clambda=50, n_Sinkhorn=50, q=50, uv_threshold=10**10) : 
                  
    T = np.shape(y)[1] 
    n = len(x_0) 
    
    particles = np.zeros(( n, n_particles, 2 )) 
    weights   = np.multiply( np.ones(( n_particles, 2 )), 1/n_particles ) 
    log_NC    = np.zeros(2)  
    for particle in range(n_particles) :
        for i in range(2) :
            particles[:, particle, i] = x_0

    # Running the particle filter: 
    
    n_delta = int(delta/dt)
    for t in range(T) :
        
        # Mutate:
        M = mutate_MCMC_cpm(particles, weights, log_NC,
                            W_values[ t*n_delta:(t+1)*n_delta, 0:n_particles, : ],
                            theta_values, y[:,t], dt, delta, g, H, R)        
                
        particles, weights = M[0], M[1]
        log_NC += M[2]

        # Resample:
        particles, weights = coupled_resampling_MCMC_cpm(particles, weights,coupling_method,
                                                         norm.cdf(W_values[0,-1,0]),
                                                         norm.cdf(W_values[0,-1,1]),
                                                         clambda, n_Sinkhorn, q, uv_threshold)
        
    del particles, weights
    
    return log_NC


# * Finally, define a function to run the correlated psuedo-marginal method.
# 
# * Correlate the auxiliary random variables as $W_{n+1} = \rho\, W_n + \sqrt{1-\rho^2} \, W$, where $W$ is an array of $N(0,1)$ random variables of the appropriate dimension.

# In[12]:

def coupled_PMH_cpm(n_particles, theta_0, y, x_0, prior, n_MCMC, sigma_theta, rho, dt, delta, g, H, R,
                    coupling_method, clambda=50, n_Sinkhorn=50, q=50, uv_threshold=10**10) : 
                
    scipy.random.seed()
    np.random.seed()
    
    T    = np.shape(y)[1] 
    n    = len(x_0) 
    theta_dim = len(theta_0)
    
    accept  = 0
    N_delta = int(T*delta/dt)
    
    theta_MCMC_values    = np.zeros((theta_dim, n_MCMC + 1)) 
    theta_MCMC_values[:,0] = theta_0
    W_current        = np.random.randn( N_delta, n_particles+1 ) 
    
    for n in range(n_MCMC) : 

        theta_current  = theta_MCMC_values[:,n]
        theta_proposed = theta_current + sigma_theta*np.random.randn(theta_dim)
        
        W_proposed = rho*W_current + np.sqrt(1-rho**2)*np.random.randn(  N_delta, n_particles+1 )
        W_values   = np.zeros(( N_delta, n_particles+1, 2 ))
        W_values[:,:,0] = W_current
        W_values[:,:,1] = W_proposed
        
        theta_values    = np.zeros(( theta_dim, 2 )) 
        theta_values[:,0] = theta_current
        theta_values[:,1] = theta_proposed
        
        l = likelihood_PF_cpm(n_particles, theta_values, W_values, y, x_0, dt, delta, g, H, R,
                              coupling_method, clambda, n_Sinkhorn, q, uv_threshold)
        
        acceptance_ratio = min(1, prior(theta_proposed)/prior(theta_current)*np.exp(l[1]-l[0]) ) 
        
        u = uniform.rvs(0,1,1)
        if u < acceptance_ratio : 
            theta_current[:] = theta_proposed 
            W_current[:,:]   = W_proposed
            accept          += 1
        
        del W_proposed, W_values, theta_proposed, theta_values
        
        theta_MCMC_values[:,n+1] = theta_current
    
    return theta_MCMC_values, accept/n_MCMC


# ### Same auxiliary variables:
# 
# * Now define a function to run a coupled MCMC using the same auxiliary variables to calculate the log-likelihoods for each pair $(\theta, \theta')$.
# 
# * This means that at each step of the MCMC, use the same auxiliary variables $W_n$ to calculate the log-likelihoods for the pair $(\theta_n, \theta_{\text{proposed}})$.
# 
# * The sequence of auxiliary variables $W_0, W_1, \ldots$ are i.i.d.
# 
# * First, define a function to calculate the log-likelihood at a pair of $\theta$-values using the same auxuliary varables.

# In[13]:

def likelihood_PF_proposed(n_particles, theta_values, y, x_0, dt, delta, g, H, R,
                           coupling_method, clambda=50, n_Sinkhorn=50, q=50, uv_threshold=10**10) : 
                  
    T = np.shape(y)[1] 
    n = len(x_0) 
    
    particles = np.zeros(( n, n_particles, 2 )) 
    weights   = np.multiply( np.ones(( n_particles, 2 )), 1/n_particles ) 
    log_NC    = np.zeros(2)  
    for particle in range(n_particles) :
        for i in range(2) :
            particles[:, particle, i] = x_0
            
    # Running the particle filter: 
    
    n_delta = delta/dt
    
    for t in range(T) :
        
        # Mutate:
        M = mutate_coupled_PF(particles, theta_values, weights, log_NC, y[:,t], dt, delta, g, H, R)
        
        particles, weights = M[0], M[1]
        log_NC += M[2]

        # Resample:
        particles, weights = perform_coupled_resampling(particles, weights, coupling_method, 
                                                        clambda, n_Sinkhorn, q, uv_threshold)
        
    del particles, weights
        
    return log_NC


# * Then define a function to run a coupled MCMC.

# In[14]:

def coupled_PMH_proposed(n_particles, theta_0, y, x_0, prior, n_MCMC, sigma_theta, dt, delta, g, H, R,
                         coupling_method, clambda=50, n_Sinkhorn=50, q=50, uv_threshold=10**10) : 
                
    scipy.random.seed()
    np.random.seed()
    
    T = np.shape(y)[1] 
    n = len(x_0) 
    theta_dim = len(theta_0)
    
    accept  = 0
    
    theta_MCMC_values      = np.zeros(( theta_dim, n_MCMC + 1 )) 
    theta_MCMC_values[:,0] = theta_0
    
    for n in range(n_MCMC) : 

        theta_current  = theta_MCMC_values[:,n]
        theta_proposed = theta_current + sigma_theta*np.random.randn(theta_dim)
        
        theta_values      = np.zeros(( theta_dim, 2 ))
        theta_values[:,0] = theta_current
        theta_values[:,1] = theta_proposed
        
        l = likelihood_PF_proposed(n_particles, theta_values, y, x_0, dt, delta, g, H, R,
                                   coupling_method, clambda, n_Sinkhorn, q, uv_threshold)
        
        acceptance_ratio = min(1, prior(theta_proposed)/prior(theta_current)*np.exp(l[1]-l[0]) ) 
        
        u = uniform.rvs(0,1,1)
        if u < acceptance_ratio : 
            theta_current = theta_proposed 
            accept       += 1
                
        theta_MCMC_values[:,n+1] = theta_current
    
    return theta_MCMC_values, accept/n_MCMC


# ### Representative jumps.
# 
# * Define a function to calculate the log-likelihoods for a pair $(\theta, \theta')$ using the correlated psuedo-marginal method. 
# 
# * That is, when the auxiliary random variables are correlated by $W_2 = \rho \, W_1 + \sqrt{1-\rho^2} \, W$, where $W$ is an array of $N(0,1)$ random variables of the appropriate dimensions.

# In[15]:

def likelihood_cpm(n_particles, theta_values, y, x_0, rho_cpm, dt, delta, g, H, R) :
    
    scipy.random.seed()
    np.random.seed()
    
    T    = np.shape(y)[1]
    n    = len(x_0)
    N_delta = int(T*delta/dt) 
    
    W_1 = np.random.randn( N_delta, n_particles+1 ) 
    W_2 = rho_cpm*W_1 + np.sqrt(1-rho_cpm**2)*np.random.randn( N_delta, n_particles+1 )
    
    W_values        = np.zeros(( N_delta, n_particles+1, 2 ))
    W_values[:,:,0] = W_1
    W_values[:,:,1] = W_2
    
    l = likelihood_PF_cpm(n_particles, theta_values, W_values, y, x_0, dt, delta, g, H, R, 'independent')
        
    del W_1, W_2, W_values
    
    return l


# #### Define a function to calculate variances of some representative jumps.
# 
# * Consider some representative jumps of the parameter value $\theta$ for a given proposal density of the MCMC.
# 
# * These are random jumps obtained from the proposal.
# 
# * For each representative jump, calculate the variance of the delta log-likelihood using (a) correlated psuedo marginal method, and (b) coupled particle filter with optimal transport resampling and same auxiliary random variables.

# In[16]:

def representative_variances(theta_0, y, x_0, N_particles, rep, n_jumps, sigma_theta, g, H, R,
                             rho_cpm, dt, delta, clambda=50, n_Sinkhorn=50, q=50, uv_threshold=10**10) :

    scipy.random.seed()
    np.random.seed()
    
    n = len(x_0)
    theta_dim = len(theta_0)
    
    theta_values      = np.zeros((theta_dim, 2) ) 
    theta_values[:,0] = theta_0

    variances = np.zeros(( n_jumps, 2 ))
    run_times = np.zeros(2)

    delta_loglikelihood = np.zeros(( rep, 2 ))

    for n in range(n_jumps) : 

        scipy.random.seed()
        theta_values[0:3,1] = theta_values[0:3,0] + sigma_theta[0:3]*np.random.randn(theta_dim-1)
        theta_values[-1 ,1] = np.exp( np.log(theta_values[-1,0]) + sigma_theta[-1]*np.random.randn(1) ) 
        
        start_time = time.clock()
        L_proposed = parmap.map(likelihood_PF_proposed, [N_particles[1]]*rep,
                                theta_values, y, x_0, dt, delta, g, H, R, 
                                'OT', clambda, n_Sinkhorn, q, uv_threshold)
        
        run_times[1] += time.clock() - start_time
        
        start_time = time.clock()
        L_cpm = parmap.map(likelihood_cpm, [N_particles[0]]*rep,
                           theta_values, y, x_0, rho_cpm, dt, delta, g, H, R)
        
        run_times[0] += time.clock() - start_time
        
        for r in range(rep) :
            delta_loglikelihood[r,0] = L_cpm[r][1] - L_cpm[r][0]
            delta_loglikelihood[r,1] = L_proposed[r][1] - L_proposed[r][0]
            
        variances[n,0] = np.var(delta_loglikelihood[:,0])        
        variances[n,1] = np.var(delta_loglikelihood[:,1])
                
    return variances


# In[ ]:



