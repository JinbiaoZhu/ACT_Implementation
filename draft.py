import numpy as np

i = np.arange(0, 5)
m = 0.95
weight = np.exp(-1 * m * i)
print(weight)
print(weight.shape)
