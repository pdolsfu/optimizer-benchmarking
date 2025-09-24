import sys
import numpy as np
import re

def geartrain(x):
    x = np.asarray(x, dtype=float)

    # If you want strict integer teeth, uncomment this rounding:
    # x = np.clip(np.rint(x), 12, 60)

    f1 = (1.0 / 6.931 - (x[2] * x[1]) / (x[0] * x[3])) ** 2
    f2 = float(np.max(x))  # minimize max teeth count
    return [f1, f2]

def read_params(filename, n_variables):
    """Read Dakota’s annotated params file and collect 'cdv_i' values."""
    xs = []
    pat = re.compile(r'^\s*([+\-]?\d*\.?\d+(?:[eE][+\-]?\d+)?)\s+cdv_\d+')
    with open(filename, 'r') as f:
        for line in f:
            m = pat.match(line)
            if m:
                xs.append(float(m.group(1)))
                if len(xs) == n_variables:
                    break
    if len(xs) != n_variables:
        raise RuntimeError(
            f"Expected {n_variables} design variables but found {len(xs)} in {filename}"
        )
    return xs

def write_results(filename, values):
    with open(filename, 'w') as f:
        for v in values:
            f.write(f"{v:.6f}\n")

if __name__ == "__main__":
    params_file  = sys.argv[1]
    results_file = sys.argv[2]

    print(f"\n--- DEBUG: dumping first 10 lines of {params_file} ---")
    with open(params_file) as f:
        for i, L in enumerate(f):
            if i >= 10: break
            print(repr(L.rstrip('\n')))
    print("--- end debug dump ---\n")

    x = read_params(params_file, 4)
    vals = geartrain(x)
    write_results(results_file, vals)
