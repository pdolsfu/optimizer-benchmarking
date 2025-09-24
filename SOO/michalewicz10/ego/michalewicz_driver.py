import sys
import numpy as np

def michalewicz(x: np.ndarray, m: int = 10) -> float:
    d = len(x)
    result = 0.0
    for i in range(d):
        xi = x[i]
        term = np.sin(xi) * (np.sin(((i + 1) * xi ** 2) / np.pi) ** (2 * m))
        result += term
    return -result

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
    # optional: allow m override via second positional argument or env var
    m = 10
    if len(sys.argv) >= 4:
        try:
            m = int(sys.argv[3])
        except ValueError:
            pass

    x = read_params(sys.argv[1], n_dim)
    y = michalewicz(x, m=m)
    write_results(sys.argv[2], y)