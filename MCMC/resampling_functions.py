
# coding: utf-8

# ### This notebook contains all the resampling functions used.
# 
# * Basic ingredients: two particle clouds with two discrete marginal densities.
# 
# * We use systematic resampling. 

# In[12]:

# Calling libraries:
from __future__ import division 
import numpy as np
import time
import math 
from scipy.sparse import lil_matrix 
import scipy.spatial as spatial

from scipy.sparse import lil_matrix, csr_matrix
from scipy.stats import norm, uniform, bernoulli
from scipy.linalg import sqrtm 
import scipy


# * First, define a systematic resampling function.

# In[ ]:

def systematic_resampling(proba_vec, n_particles, U):
    """
    # description:
        fast implementation of systematic resampling. Works best when "proba_vec" is well balanced
    # argument:
        proba_vec: a numpy vector representing a probability distribution
        U: a sample of a uniform random number
        n_particles: number of particles to sample from "proba_vec"
    # output:
        index of the particles that have been resampled
    """
    dim = len(proba_vec)
    CS = (np.cumsum(proba_vec) - U / n_particles) * n_particles
    CS = np.floor(CS)
    CS_jump = np.zeros(dim)
    CS_jump[0] = CS[0]+1
    CS_jump[1:] = CS[1:] - CS[:-1]
    CS_jump = CS_jump.astype(int)
    
    resampled_particles = np.zeros(n_particles)
    n_particles_done = 0
    for k in np.unique(CS_jump):
        index_jump_k = np.where(CS_jump == k)[0]
        nb_of_index_jump_k = len(index_jump_k)
        resampled_particles[n_particles_done:(n_particles_done + k*nb_of_index_jump_k)] = np.repeat(index_jump_k,k)
        n_particles_done += k*nb_of_index_jump_k
    
    return(resampled_particles)  


# ### Independent resampling 
# 
# * Resample the two particle clouds independently.

# In[8]:

"""
particle_cloud_1 and particle_cloud_2 are the two particle clouds
r and c are the marginals 
u_1 and u_2 are the U(0,1) random variables used for systematic resampling
"""

def independent_resample(particle_cloud_1, particle_cloud_2, r, c, u_1, u_2):        
    
    M = len(r)               # number of particles
    particle_cloud_1 = particle_cloud_1[systematic_resampling(r, M, u_1).astype(int), :]
    particle_cloud_2 = particle_cloud_2[systematic_resampling(c, M, u_2).astype(int), :]
    
    return particle_cloud_1, particle_cloud_2, [1/M]*M, [1/M]*M 


# ### Maximal coupling resampling
# 
# * Consider the maximal coupling between the two marginals and use this to resample.

# In[9]:

"""
particle_cloud_1 and particle_cloud_2 are the two particle clouds
weights_1 and weights_2 are the marginals 
u_1 and u_2 and the U(0,1) random variables used for systematic resampling
"""

def maximal_coupling_resample(particle_cloud_1, particle_cloud_2, weights_1, weights_2, u_1, u_2) :    
    
    M, d = np.shape(particle_cloud_1)
    
    particle_cloud_1_resampled = np.zeros(( M, d )) 
    particle_cloud_2_resampled = np.zeros(( M, d ))
    
    weights_min = np.minimum(weights_1,weights_2) 
    
    r = np.arange(0,M) 
    b = np.random.binomial(n=1, p=np.sum(weights_min), size=M)
    
    r_1 = r[b>0]; r_2 = r[b==0]
    w_min = weights_min/np.sum(weights_min)
    
    w_1 = (weights_1 - weights_min) / np.sum(weights_1 - weights_min)     
    w_2 = (weights_2 - weights_min) / np.sum(weights_2 - weights_min)
    
    if len(r_1) == 0 :                                       # nothing coupled
        particle_cloud_1_resampled = particle_cloud_1[systematic_resampling(w_1, M, u_1).astype(int), :]
        particle_cloud_2_resampled = particle_cloud_2[systematic_resampling(w_2, M, u_2).astype(int), :]
    
    else:
        resampled_index_coupled = systematic_resampling(w_min, len(r_1), u_1).astype(int)
        particle_cloud_1_resampled[r_1,:] = particle_cloud_1[resampled_index_coupled,:]
        particle_cloud_2_resampled[r_1,:] = particle_cloud_2[resampled_index_coupled,:]
        
        if len(r_2) > 0 :                                     # some not coupled
            particle_cloud_1_resampled[r_2,:] = particle_cloud_1[systematic_resampling(w_1, len(r_2), u_1).astype(int), :] 
            particle_cloud_2_resampled[r_2,:] = particle_cloud_2[systematic_resampling(w_2, len(r_2), u_2).astype(int), :]
    
    return particle_cloud_1_resampled, particle_cloud_2_resampled, [1/M]*M, [1/M]*M


# ### Approximate optimal transport resample:
# 
# * Use Euclidean distance.
# 
# * Use Sinkhorn iteration.
# 
# * Dene matrix implementation.

# In[ ]:

def UVK(u, v, K):
    """
    computes U * K * V
    """
    UU = np.diag(u)
    VV = np.diag(v)    
    return UU.dot(K.dot(VV))

def compute_marginals(C):
    return C.sum(axis=0), C.sum(axis=1).flatten()

def OT_cost(C, d_matrix, sparse_algebra = False):
    if sparse_algebra:
        return np.sum( C.multiply(d_matrix))
    else:
        return np.sum( C * d_matrix)

def distance_matrix(X_1, X_2):
    dim, ndata = X_1.shape
    d_matrix = np.zeros((ndata, ndata))
    index_all = np.array(range(ndata))
    for k in range(ndata):
        d_matrix[index_all, k] = np.sqrt(np.sum( (X_1[:,index_all] - X_2[:,k+np.zeros(ndata).astype(int)])**2, axis = 0))
    return d_matrix


# * Let $d_q$ be the $q$-th percentile of the distance matrix.
# 
# * We choose $\lambda$ such that $\exp(-\lambda d) = 10^{-c_\lambda} \Rightarrow \lambda = (c_\lambda \ln 10) / d_q$.
# 
# * Keep a threshold for $u$ and $v$, such that, if they cross the threshold then we adjust them.

# In[1]:

def solve_Sinkhorn_dense(X_1, X_2, r, c, clambda=50, n_Sinkhorn=50, q=50, uv_threshold=10**10):
    
    ndata = len(r)
    u = np.copy(r)
    v = np.copy(c)
    lambda_iter = 0
    
    #compute NN and build distance matrix
    D_matrix = distance_matrix(X_1, X_2)
    
    #build K matrix
    d = np.percentile(D_matrix, q)
    K_matrix = np.exp( -clambda*np.log(10)/d * D_matrix )
    
    #adaptive sinkhorn iteration
    for iteration in range(n_Sinkhorn) :        
        #check if u, v hit the threshold
        if np.maximum(np.max(u), np.max(v)) > uv_threshold :
            #decrease lambda and update K_matrix
            clambda = 0.7 * clambda
            K_matrix = np.exp( -clambda*np.log(10)/d * D_matrix )
            #u /= np.max(u)
            #v /= np.max(v)
            
        #update u and v
        u[:] = r / K_matrix.dot(v)
        v[:] = c / K_matrix.transpose().dot(u)

    C = UVK(u, v, K_matrix)
    C = C / np.sum(C)
    del D_matrix, K_matrix
    
    return C


# In[ ]:

""" U is the U(0,1) random variable used """

def OT_systematic_resample(coupling_matrix, U):
    """
    # description:
        fast implementation of systematic resampling. Works best when "proba_vec" is well balanced
    # argument:
        coupling_matrix: a 2D matrix representing a probability distribution
        U: a sample of a uniform random number
    # output:
        a list of array: [[array],[array]] containing first/second coordinate of resampled particles
    """
    dim, _ = coupling_matrix.shape
    resampled_particles = systematic_resampling(coupling_matrix.ravel(), dim, U)
    indices_resampled = [ (resampled_particles // dim).astype(int), (resampled_particles % dim).astype(int) ]
    return indices_resampled


# In[ ]:

def approximate_OT_resample(X_1, X_2, r, c, U, clambda=50, n_Sinkhorn=50, q=50, uv_threshold=10**10) :
    
    ndata, dim = np.shape(X_1)
    coupling_matrix = solve_Sinkhorn_dense(np.transpose(X_1), np.transpose(X_2), 
                                           r, c, clambda, n_Sinkhorn, q, uv_threshold) 
    r_indices = OT_systematic_resample(coupling_matrix, U) 
    
    return X_1[r_indices[0],:], X_2[r_indices[1],:], [1/ndata]*ndata, [1/ndata]*ndata


# ### Combined resampling function: 

# In[1]:

"""
particle_cloud_1 and particle_cloud_2 are the two particle clouds
r and c are the marginals 
U1 and U2 are the U(0,1) random variables used for systematic resampling
"""

def resample(X_1, X_2, r, c, U1, U2, coupling_method, clambda=50, n_Sinkhorn=50, q=50, uv_threshold=10**10) :
    
    ndata = len(r)
    r /= np.sum(r)
    c /= np.sum(c)
    
    lambda_final = -10
    lambda_iter  = -1
    
    if coupling_method == 'independent' :
        X_1, X_2, r, c = independent_resample(X_1, X_2, r, c, U1, U2)
    
    if coupling_method == 'maximal' :
        X_1, X_2, r, c = maximal_coupling_resample(X_1, X_2, r, c, U1, U2)
        
    if coupling_method == 'OT' : 
        X_1, X_2, r, c = approximate_OT_resample(X_1, X_2, r, c, U1, clambda, n_Sinkhorn, q, uv_threshold)
    
    return X_1, X_2, r, c


# In[ ]:



