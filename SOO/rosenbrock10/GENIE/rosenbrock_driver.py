# takes parameter and results files from Dakota and computes true function valuations, writing them back to Dakota
# ensure that line 30 reflects the dimensionality of the problem being tested

import sys
import numpy as np

def rosen(x):
    return sum(100 * (x[i]**2 - x[i+1])**2 + (x[i] - 1)**2 for i in range(len(x) - 1))

# reading a set of values
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

# writing results
def write_results(filename, value):
    with open(filename, 'w') as f:
        f.write(f"{value:.6f}\n")

if __name__ == "__main__":
    x = read_params(sys.argv[1], 10)           # modify this line depending on the number of dimensions
    y = rosen(x)
    write_results(sys.argv[2], y)
