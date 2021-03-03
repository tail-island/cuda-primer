import numpy as np
import cupy as cp
import sys
import timeit


M = 1001
N = 1002
K = 1003


a = cp.reshape(cp.array(np.linspace(0.0, 1.0, M * K), dtype=cp.float32), (M, K))
b = cp.reshape(cp.array(np.linspace(0.0, 1.0, K * N), dtype=cp.float32), (K, N))

print(f'{timeit.timeit(stmt=lambda: cp.matmul(a, b), number=10) / 10}', file=sys.stderr)

c = cp.matmul(a, b)

print(c[0][0])
print(c[0][1])
print(c[0][2])

print(c[-1][-3])
print(c[-1][-2])
print(c[-1][-1])
