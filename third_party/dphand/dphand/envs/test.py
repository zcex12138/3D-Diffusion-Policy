import numpy as np
from numpy._typing import _128Bit

def calc_p(a):
    a[0] = 2
    return a

a = np.array([1,2,3,4,5,6])
print(np.clip(a, 2, 4))
print(a)