# takes parameter and results files from Dakota and computes true function [g7] evaluations, writing them back to Dakota
# note to user: ensure that line 50 reflects the dimensionality of the [g7] problem being tested

import sys
import numpy as np

# parse Dakota parameter file
def read_params(filename, n_variables):
    with open(filename) as f:
        lines = f.readlines()

    x = []  # List to store numeric values for variables

    for i, line in enumerate(lines):
        if 'variables' in line:  # Look for the 'variables' section
            for j in range(n_variables):
                x_line = lines[i + j + 1].split()  # Get the next line (after 'variables')
                x.append(float(x_line[0]))  # Extract only the numeric value (x1) and add to the array
            break
    return x     

# write results
def write_results(filename, values):
    with open(filename, 'w') as f:
        for val in values:
            f.write(f"{val:.6f}\n")

def g7(x):
    return (
        x[0]**2 + x[1]**2 + x[0]*x[1] - 14*x[0] - 16*x[1]
        + (x[2] - 10)**2 + 4*(x[3] - 5)**2 + (x[4] - 3)**2
        + 2*(x[5] - 1)**2 + 5*x[6] + 7*(x[7] - 11)**2
        + 2*(x[8] - 10)**2 + (x[9] - 7)**2 + 45
    )

# Define inequality constraints g_i(x) <= 0
def constraints(x):
    g = [0]*8
    g[0] =  4*x[0] + 5*x[1] - 3*x[6] + 9*x[7] - 105
    g[1] = 10*x[0] - 8*x[1] - 17*x[6] + 2*x[7]
    g[2] = -8*x[0] + 2*x[1] + 5*x[8] - 2*x[9] - 12
    g[3] = 3*(x[0]-2)**2 + 4*(x[1]-3)**2 + 2*x[2]**2 - 7*x[3] - 120
    g[4] = 5*x[0]**2 + 8*x[1] + (x[2]-6)**2 - 2*x[3] - 40
    g[5] = 0.5*(x[0]-8)**2 + 2*(x[1]-4)**2 + 3*x[4]**2 - x[5] - 30
    g[6] = x[0]**2 + 2*(x[1]-2)**2 - 2*x[0]*x[1] + 14*x[4] - x[5]
    g[7] = -3*x[0] + 6*x[1] + 12*(x[8]-8)**2 - 7*x[9]
    return g

if __name__ == "__main__":
    x = read_params(sys.argv[1], 10)           # modify this line depending on the number of dimensions
    obj = g7(x)
    cons = constraints(x)
    write_results(sys.argv[2], [obj] + cons)
