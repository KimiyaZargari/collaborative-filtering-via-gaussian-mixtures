"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, d = X.shape
    K = mixture.mu.shape[0]
    
    # Extract mixture parameters
    mu, var, p = mixture.mu, mixture.var, mixture.p
    
    # Initialize responsibilities matrix
    post = np.zeros((n, K))
    log_likelihood = 0.0
    
    for i in range(n):
        observed_indices = X[i, :] != 0
        d_obs = np.sum(observed_indices)
        
        for j in range(K):
            mu_j_obs = mu[j, observed_indices]
            X_i_obs = X[i, observed_indices]
            var_j = var[j]
            
            # Calculate the log probability of the observed data under component j
            coeff = np.log(p[j]) - 0.5 * d_obs * np.log(2 * np.pi * var_j)
            exponent = -0.5 * np.sum((X_i_obs - mu_j_obs) ** 2) / var_j
            log_prob = coeff + exponent
            
            post[i, j] = log_prob
        
        # Normalize across components
        log_total_prob = logsumexp(post[i, :])
        post[i, :] = np.exp(post[i, :] - log_total_prob)
        
        # Add to the total log-likelihood
        log_likelihood += log_total_prob
    
    return post, log_likelihood

def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    [n,d] = np.shape(X)
    k = len(mixture.p)

    n_k = np.sum(post, axis=0) # (1,k) -> (k,)
    p_k = n_k/n # (1,k) -> (k,)

    # allocation array
    mu_k = np.zeros([k,d])
    var_k = np.zeros(k)

    # allocate list to append 
    non_zero_length = []

    # non-zero length for each sample 
    for i in range(n):

        non_zero_index = np.where(X[i] != 0)[0]
        non_zero_length.append(len(non_zero_index)) # list, (1,n)

    non_zero_length = np.asarray(non_zero_length) # (n,1) -> (n,)
    
    
    # mean estimation, (k,d) 
    for i in range(k):

        for j in range(d):
            index = np.where(X[:,j] != 0)[0] # index where X not zero
            
            # update condition
            if np.sum(post[index,i]) >= 1:
                mu_k[i,j] = np.inner(X[index,j], post[index,i]) / np.sum(post[index,i])
            else:
                mu_k[i,j] = mixture.mu[i,j]

    # var estimation, (1,k) -> (k,)
    B = np.zeros([n,k])

    for i in range(n):

        for j in range(k):
            index = np.where(X[i] != 0)[0] # index where X not zero

            A = np.linalg.norm(X[i,index] - mu_k[j,index])
            B[i,j] = post[i,j]*A**2

    var_k = np.sum(B, axis=0) / np.matmul(post.T, non_zero_length)
    
    # var criteria 
    index = np.where(var_k <= min_variance)[0]
    var_k[index] = min_variance

    return GaussianMixture(mu_k, var_k, p_k)

import copy
def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    prev_log_likelihood = None
    tol = 1e-6  # Convergence threshold
    max_iter = 1000  # Maximum number of iterations

    # Create a deep copy of X to avoid modifying the input array
    X = copy.deepcopy(X)

    for _ in range(max_iter):
        # E-step: Compute the responsibilities and log-likelihood
        post, log_likelihood = estep(X, mixture)
        
        # M-step: Update the Gaussian mixture parameters
        mixture = mstep(X, post, mixture)

        # Check for convergence
        if prev_log_likelihood is not None:
            improvement = log_likelihood - prev_log_likelihood
            if improvement <= tol * abs(log_likelihood):
                break
        
        prev_log_likelihood = log_likelihood

    return mixture, post, log_likelihood


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    n,_ = np.shape(X)
    k = len(mixture.p)
    NK = np.zeros([n,k])
    post, _ = estep(X, mixture)


    # just a copy
    X_pred = np.copy(X)

    # expectation value 
    update = post @ mixture.mu # (n,d)

    # selection Hu
    for i in range(n):
        Cu = np.where(X[i] != 0)[0] # return tuple so need [0]
        Hu = np.where(X[i] == 0)[0]

        X_pred[i,Hu] = update[i,Hu]

    return X_pred
