import numpy as np
import matplotlib.pyplot as plt

sample = np.random.multivariate_normal(np.transpose([1, -1]), 2 * np.identity(2), 100)
sample = np.transpose(sample)
print(sample[0])
print(sample[1])
fig, ax = plt.subplots()
ax.scatter(sample[0], sample[1])
ax.set_xlim(-10, 10)
ax.set_ylim(-5, 5)
plt.title( 'Gaussian 2D Sample' )
plt.show()
print(sample)