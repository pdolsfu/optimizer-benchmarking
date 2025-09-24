# python_functions.py, defines the functions for the SOO unconstrained functions
import numpy as np

# rosenbrock 
def rosenbrock(x: np.ndarray) -> float:
    return np.sum(100.0*(x[1:] - x[:-1]**2.)**2. + (1 - x[:-1])**2.)

# shubert
def shubert(x: np.ndarray) -> float:
    result = 1.0
    for xi in x:
        inner_sum = sum(j * np.cos((j + 1) * xi + j) for j in range(1, 6))
        result *= inner_sum
    return result

# michalewicz
def michalewicz(x: np.ndarray, m: int = 10) -> float:
    d = len(x)
    result = 0.0
    for i in range(d):
        xi = x[i]
        term = np.sin(xi) * np.sin(((i + 1) * xi ** 2) / np.pi) ** (2 * m)
        result += term
    return -result