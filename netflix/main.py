import numpy as np
import kmeans
import common
import naive_em
import em
import os
from matplotlib import pyplot as plt
# Load the data
X = np.loadtxt("oy_data.txt")


# Iterate over k and seed values
for k in range(1, 5):  # k = 1 to 4
    for seed in range(5):  # seed = 0 to 4
        # Initialize mixture and post
        mixture, post = common.init(X, k, seed)
        
        # Run K-means algorithm
        mixture, post, cost = kmeans.run(X, mixture, post)
        
        # Plot the results
        plot_title = f'k = {k}, seed = {seed}'
        plot_filename = f'plot_k{k}_seed{seed}.png'
        common.plot(X, mixture, post, plot_title, plot_filename)
        
        

        
        # Print the cost
        print(f'Cost for k = {k}, seed = {seed}: {cost}')