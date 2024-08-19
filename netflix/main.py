import numpy as np
import kmeans
import common
import naive_em
import em
import os
from matplotlib import pyplot as plt

# Load the data
X = np.loadtxt("netflix\\toy_data.txt")
#Define range of K values
K_values = [1, 2, 3, 4]
seeds = [0, 1, 2, 3, 4]

#Iterate over k and seed values

output_dir = "k-means-plots"
os.makedirs(output_dir, exist_ok=True)
for K in K_values:
    for seed in seeds:
        # Initialize mixture and post
        mixture, post = common.init(X, K, seed)
        
        # Run K-means algorithm
        mixture, post, cost = kmeans.run(X, mixture, post)
        
        # Plot the results
        plot_title = f'k = {K}, seed = {seed}'

        plot_filename = os.path.join(output_dir,f'plot_k{K}_seed{seed}.png')
        common.plot(X, mixture, post, plot_title, plot_filename)

        # Print the cost
        print(f'Cost for k = {K}, seed = {seed}: {cost}')

output_dir = "EM-plots"
os.makedirs(output_dir, exist_ok=True)
# Run EM for each K and each seed
for K in K_values:
    best_log_likelihood = -np.inf
    best_mixture = None
    best_post = None

    print(f"\nResults for K = {K}:")

    for seed in seeds:
        np.random.seed(seed)
        
        # Initialize Gaussian Mixture Model parameters here (e.g., randomly)
        initial_mixture, initial_post = common.init(X, K, seed)
        
        # Run the EM algorithm
        mixture, post, log_likelihood = em.run(X, initial_mixture, initial_post)
        
        # Print the log-likelihood for this run
        print(f"Seed {seed}: Log-Likelihood = {log_likelihood}")
        plot_title = f'k = {K}, seed = {seed}'
        plot_filename = os.path.join(output_dir,f'plot_k{K}_seed{seed}.png')
        common.plot(X, mixture, post, plot_title, plot_filename)
    