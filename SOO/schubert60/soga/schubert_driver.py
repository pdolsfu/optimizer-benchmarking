import sys
import numpy as np

def shubert(x: np.ndarray) -> float:
    result = 1.0
    for xi in x:
        inner_sum = sum(j * np.cos((j + 1) * xi + j) for j in range(1, 6))
        result *= inner_sum
    return result

def read_params(filename, n_variables):
    with open(filename) as f:
        lines = f.readlines()

    x = []
    for i, line in enumerate(lines):
        if 'variables' in line:
            for j in range(n_variables):
                x_line = lines[i + j + 1].split()
                x.append(float(x_line[0]))
            break
    return x

def write_results(filename, value):
    with open(filename, 'w') as f:
        f.write(f"{value:.6f}\n")

if __name__ == "__main__":
    # adjust dimensionality here if needed
    n_dim = 10
    x = read_params(sys.argv[1], n_dim)
    y = shubert(x)
    write_results(sys.argv[2], y)