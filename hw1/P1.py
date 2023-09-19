import numpy as np

X = np.matrix([[3, 2], [-7, -5]])
y = np.matrix([[2], [1]])
z = np.matrix([[1], [-1]])

res1 = np.dot(np.transpose(y), X)
print(res1)

res2 = np.dot(res1, z)
print(res2)