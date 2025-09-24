# graph.py, this script graphs the Pareto Frontiers and creates the runtime files for all MOO solvers (Python, OASIS, Dakota) on all problems 

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV

'''
Overlay all solver Pareto fronts (and OASIS) in one plot.
'''
def plot_pareto_frontiers(func_folder, solvers):
    plt.figure()

    def _load_interp_and_plot(path, label, style):
        data = np.loadtxt(path, delimiter='\t', skiprows=1)
        x, y = data[:,0], data[:,1]
        idx = np.argsort(x)            # ensure monotonic x
        x_s, y_s = x[idx], y[idx]
        xs = np.linspace(x_s.min(), x_s.max(), 200)
        ys = np.interp(xs, x_s, y_s)
        plt.plot(xs, ys, style, label=label)
        plt.scatter(x, y, s=10)

    for i, sol in enumerate(solvers):
        # choose style by position
        if i < 4:
            style = '-'       # first 4: solid
        elif i == 4:
            style = '-.'      # fifth: dot-dash
        else:
            style = '--'      # others: dashed

        p = os.path.join(func_folder, sol, f"{sol}_pareto.tsv")
        try:
            _load_interp_and_plot(p, sol.upper(), style)
        except Exception as e:
            print(f"Warning loading {p}: {e}")

    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.title(f"{func_folder} Pareto Frontiers")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{func_folder}/{func_folder}_pareto_frontier.png", dpi=300)

def calculate_hypervolume(func_folder, solvers):
    all_points = []
    solver_points = {}

    # Load Pareto points for each solver
    for sol in solvers:
        p = os.path.join(func_folder, sol, f"{sol}_pareto.tsv")
        if not os.path.isfile(p):
            print(f"Warning: {p} not found.")
            continue
        try:
            data = np.loadtxt(p, delimiter='\t', skiprows=1)
            if data.ndim == 1:  # only one point
                data = data.reshape(1, -1)
            solver_points[sol] = data
            all_points.append(data)
        except Exception as e:
            print(f"Warning reading {p}: {e}")

    if not all_points:
        raise RuntimeError(f"No Pareto data found for {func_folder}")

    # Combine all solver points to define reference point
    all_points = np.vstack(all_points)
    ref = [all_points[:,0].max() * 1.01, all_points[:,1].max() * 1.01]

    results = []
    for sol, pts in solver_points.items():
        hv = HV(ref_point=ref).do(pts)
        results.append((sol, hv))

    # Write TSV
    out_path = os.path.join(func_folder, f"{func_folder}_hypervolume.tsv")
    with open(out_path, "w") as f:
        f.write("solver\thypervolume\n")
        for sol, hv in results:
            f.write(f"{sol}\t{hv:.5}\n")

    print(f"✔ Hypervolume written to {out_path}")

def combine_time_files(func_name: str, solvers: list, instances: int, out_name: str = None):
    """
    Reads each {func_name}/{solver}/{solver}_time.tsv and writes a combined
    Run\t<solver1>\t<solver2>… file to {func_name}/{out_name}.
    """
    # Prepare output filename
    if out_name is None:
        out_name = f"{func_name}_time.tsv"
    out_path = os.path.join(func_name, out_name)
    os.makedirs(func_name, exist_ok=True)

    # Initialize a dict of per-solver time lists (filled with NaN)
    times = {solver: [math.nan] * instances for solver in solvers}

    # Read each solver’s file
    for solver in solvers:
        tsv_path = os.path.join(func_name, solver, f"{solver}_time.tsv")
        if not os.path.isfile(tsv_path):
            print(f"Warning: missing timing file for {solver}: {tsv_path}")
            continue

        with open(tsv_path, 'r') as f:
            # skip header
            next(f)
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                run_str, time_str = parts
                try:
                    run_idx = int(run_str)
                    t = float(time_str)
                except ValueError:
                    continue
                if 1 <= run_idx <= instances:
                    times[solver][run_idx - 1] = t

    # Write combined TSV
    with open(out_path, 'w') as f:
        # Header row
        f.write("Run\t" + "\t".join(solvers) + "\n")
        # One row per run
        for i in range(instances):
            row = [str(i + 1)]
            for solver in solvers:
                t = times[solver][i]
                row.append(f"{t:.5f}" if not math.isnan(t) else "nan")
            f.write("\t".join(row) + "\n")

    print(f"Wrote combined timing file → {out_path}")

def main(func_names, solvers, instances):
    for func_name in func_names:
        os.makedirs(func_name, exist_ok=True)
        combine_time_files(func_name, solvers, instances)
        plot_pareto_frontiers(func_name, solvers)
        calculate_hypervolume(func_name, solvers)

if __name__ == "__main__":
    func_names = ["zdt1", "zdt2", "zdt3", "zdt6", "geartrain", "osy"]
    solvers = ["MOEAD", "NSGA2", "OMADS", "SPEA2", "OASIS", "ea", "moga"]
    
    main(func_names, solvers, 10)