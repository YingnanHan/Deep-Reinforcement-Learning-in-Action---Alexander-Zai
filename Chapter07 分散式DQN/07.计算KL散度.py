import numpy as np

p = np.array([0.1,0.1])
q = np.array([0.5,0.6])

def lr(p,q):
    x = q * np.log(p/q)
    x = -1 * np.sum(x)
    return x

print(lr(p,q))