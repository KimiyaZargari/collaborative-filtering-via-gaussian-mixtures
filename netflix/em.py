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
    
    # Compute Gaussian likelihoods and responsibilities
    for j in range(K):
        # For each component, calculate the probability density function
        diff = X - mu[j]  # (n, d) difference
        exponent = -0.5 * np.sum(diff ** 2, axis=1) / var[j]
        coeff = p[j] / ((2 * np.pi * var[j]) ** (d / 2))
        post[:, j] = coeff * np.exp(exponent)
    
    # Normalize the responsibilities across all components
    total_post = np.sum(post, axis=1).reshape(-1, 1)
    post /= total_post
    
    # Calculate the log-likelihood of the data given the current mixture
    log_likelihood = np.sum(np.log(total_post))
    
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
    n, d = X.shape
    K = post.shape[1]
    
    # Compute the new mixture weights
    nj = np.sum(post, axis=0)  # (K,)
    p = nj / n  # (K,)
    
    # Compute the new means
    mu = (post.T @ X) / nj[:, np.newaxis]  # (K, d)
    
    # Compute the new variances
    var = np.zeros(K)
    for j in range(K):
        diff = X - mu[j]  # (n, d)
        var[j] = np.sum(post[:, j] * np.sum(diff ** 2, axis=1)) / (d * nj[j])
    
    # Return the updated Gaussian mixture
    return GaussianMixture(mu, var, p)


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
    tol = 1e-6  # Convergence threshold multiplier
    max_iter = 1000  # Maximum number of iterations

    for _ in range(max_iter):
        # E-step: Calculate responsibilities (soft counts) and log-likelihood
        post, log_likelihood = estep(X, mixture)
        
        # M-step: Update the mixture parameters
        mixture = mstep(X, post)
        
        # Check for convergence based on the specified criterion
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
    raise NotImplementedError
