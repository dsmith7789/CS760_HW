# source: https://stackoverflow.com/questions/73486167/sampling-from-multivariate-mixed-gaussian-distributions-with-python

import matplotlib.pyplot as plt
import numpy as np
import random

# Bivariate example
dim = 2

# Settings
n = 100
NumberOfMixtures = 2

# Mixture weights (non-negative, sum to 1)
w = [0.3, 0.7]

# Mean vectors and covariance matrices
MeanVectors = [ [5,0], [-5,0] ]
CovarianceMatrices = [ [[1, 0.25], [0.25, 1]], [[1, -0.25], [-0.25, 1]] ]

# Initialize arrays
samples = np.empty( (n,dim) ); samples[:] = np.NaN
componentlist = np.empty( (n,1) ); componentlist[:] = np.NaN

# Generate samples
for iter in range(n):
    # Get random number to select the mixture component with probability according to mixture weights
    DrawComponent = random.choices(range(NumberOfMixtures), weights=w, cum_weights=None, k=1)[0]
    # Draw sample from selected mixture component
    DrawSample = np.random.multivariate_normal(MeanVectors[DrawComponent], CovarianceMatrices[DrawComponent], 1)
    # Store results
    componentlist[iter] = DrawComponent
    samples[iter, :] = DrawSample

# Report fractions
print('Fraction of mixture component 0:', np.sum(componentlist==0)/n)
print('Fraction of mixture component 1:',np.sum(componentlist==1)/n)
#print('Fraction of mixture component 2:',np.sum(componentlist==2)/n)

# Visualize result
fig, ax = plt.subplots()
ax.scatter(samples[:, 0], samples[:, 1])
ax.set_xlim(-10, 10)
ax.set_ylim(-5, 5)
plt.title( 'Mixture Sample' )
plt.show()

# plt.plot(samples[:, 0], samples[:, 1], '.', alpha=0.5)
# plt.grid()