import sys
import numpy as np
import re

def osy(x):
    x = np.asarray(x, dtype=float)
    f1 = -(
        25.0 * (x[0] - 2.0) ** 2
        + (x[1] - 2.0) ** 2
        + (x[2] - 1.0) ** 2
        + (x[3] - 4.0) ** 2
        + (x[4] - 1.0) ** 2
    )
    f2 = float(np.sum(x ** 2))
    return [f1, f2]

def osy_constraints_leq0(x):
    # OSY (feasible if g_i >= 0). Convert to c_i <= 0 by negating each g_i.
    x1, x2, x3, x4, x5, x6 = x
    c1 = 2 - x1 - x2                        # -(x1 + x2 - 2)
    c2 = x1 + x2 - 6                        # -(6 - x1 - x2)
    c3 = x2 - x1 - 2                        # -(2 - x2 + x1)
    c4 = x1 - 3*x2 - 2                      # -(2 - x1 + 3*x2)
    c5 = (x3 - 3)**2 + x4 - 4               # -[4 - (x3-3)^2 - x4]
    c6 = 4 - (x5 - 3)**2 - x6               # -[(x5-3)^2 + x6 - 4]
    return [c1, c2, c3, c4, c5, c6]


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

    # Optional debug dump (matches your zdt2 driver style)
    print(f"\n--- DEBUG: dumping first 10 lines of {params_file} ---")
    with open(params_file) as f:
        for i, L in enumerate(f):
            if i >= 10: break
            print(repr(L.rstrip('\n')))
    print("--- end debug dump ---\n")

    x = read_params(params_file, 6)
    f1, f2 = osy(x)
    con = osy_constraints_leq0(x)
    write_results(results_file, [f1, f2] + con)