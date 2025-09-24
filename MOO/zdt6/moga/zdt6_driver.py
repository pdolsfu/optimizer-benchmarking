import sys
import numpy as np
import re

def zdt6(x):
    x = np.asarray(x, dtype=float)
    n = x.size
    f1 = 1.0 - np.exp(-4.0 * x[0]) * np.sin(6.0 * np.pi * x[0])**6
    g  = 1.0 + 9.0 * (np.sum(x[1:]) / (n - 1))**0.25
    h  = 1.0 - (f1 / g)**2.0
    f2 = g * h
    return [f1, f2]

def read_params(filename, n_variables):
    """
    Read Dakota’s annotated params file and extract the first n_variables
    design‐variable values from the lines labelled 'cdv_#'.
    """
    x = []
    # match one float (possibly in scientific notation) before 'cdv_<index>'
    pat = re.compile(r'^\s*([+\-]?\d*\.?\d+(?:[eE][+\-]?\d+)?)\s+cdv_\d+')
    with open(filename, 'r') as f:
        for line in f:
            m = pat.match(line)
            if m:
                x.append(float(m.group(1)))
                if len(x) == n_variables:
                    break

    if len(x) != n_variables:
        raise RuntimeError(
            f"Expected {n_variables} design variables but found {len(x)} in {filename}"
        )
    return x

def write_results(filename, values):
    """
    Write each objective value on its own line, in the order Dakota expects.
    """
    with open(filename, 'w') as f:
        for v in values:
            f.write(f"{v:.6f}\n")

if __name__ == "__main__":
    params_file  = sys.argv[1]
    results_file = sys.argv[2]
    print(f"\n--- DEBUG: dumping first 10 lines of {params_file} ---")
    with open(params_file) as f:
        for i, L in enumerate(f):
            print(repr(L.rstrip('\n')))
    print("--- end debug dump ---\n")
    x = read_params(params_file, 10)
    obj_vals = zdt6(x)
    write_results(results_file, obj_vals)
    