# python_batch_optmizers.py, the main script for G7 that sets up file paths, runs the solvers and post-processes the data

from python_solvers import Problem, SciPyDE, SciPyDA, SciPyDIRECT, NGIoh, PymooGA, AxBO, OMADSSolver

import os, time
import numpy as np

# G7 function definition
def gseven(x: np.ndarray) -> float:
    return (
        x[0]**2 + x[1]**2 + x[0]*x[1] - 14*x[0] - 16*x[1] +
        (x[2] - 10)**2 + 4*(x[3] - 5)**2 +
        (x[4] - 3)**2 + 2*(x[5] - 1)**2 +
        5*x[6]**2 + 7*(x[7] - 11)**2 +
        2*(x[8] - 10)**2 + (x[9] - 7)**2 + 45
    )

# G7 constraint definitions, signs (greater/less than) of constraints are handled appropriately by individual solvers
gseven_constraints = [
    lambda x: 105 - (4*x[0] + 5*x[1] - 3*x[6] + 9*x[7]) + 1e-6,                          # g1(x) ≤ 0
    lambda x: -(10*x[0] - 8*x[1] - 17*x[6] + 2*x[7]) + 1e-6,                             # g2
    lambda x: -(8*x[0] - 2*x[1] + 5*x[8] - 2*x[9]) + 1e-6,                               # g3
    lambda x: 120 - (3*(x[0] - 2)**2 + 4*(x[1] - 3)**2 + 2*x[2]**2 - 7*x[3]) + 1e-6,     # g4
    lambda x: 40 - (5*x[0]**2 + 8*x[1] + (x[2] - 6)**2 - 2*x[3])  + 1e-6,                # g5
    lambda x: 30 - (0.5*(x[0] - 8)**2 + (x[1] - 4)**2 + 3*x[4]**2) + 1e-6,               # g6
    lambda x: x[0]*x[1] + x[0]*x[5] - 2*x[0]**2 - 2*x[1]**2 + 14*x[5] - 6*x[6] + 1e-6,   # g7
    lambda x: -(3*x[0] - 6*x[1] + 12*(x[8] - 8)**2 - 7*x[9]) + 1e-6                      # g8
]

objective_configs = {
    "gseven10": {
        "func": gseven,
        "ndim": 10,
        "budget": 500,
        "bounds": [(-10,10)],
        "has_constraints": True,
        "constraints": gseven_constraints
    }
}


# Handles files setup for each solver, runs each solver 10 times
def runopt(func_name, func_folder, obj_func, bounds, constraints, has_constraints, ndim, budget, solvers, instances):
    print(f"--- {func_name}D budget={budget}, with {instances} instances ---")

    for s in solvers:
        solver_name = s.name
        solver_folder = os.path.join(func_folder, solver_name)
        os.makedirs(solver_folder, exist_ok=True)

        per_solver_time_file = os.path.join(solver_folder, f"{solver_name}_time.tsv")
        with open(per_solver_time_file, "w") as psf:
            psf.write("Run\tTime\n")

            for instance in range(1, instances + 1):
                prob = Problem(
                    dim=ndim,
                    instance=instance,
                    bounds=bounds,
                    constraints=constraints if has_constraints else None,
                    objective=obj_func,
                )

                start = time.time()
                res = s.solve(prob)
                end = time.time()
                elapsed = end - start

                # write per-solver time immediately
                psf.write(f"{instance}\t{elapsed:.5f}\n")

                if "error" in res:
                    print(f"{solver_name:12s} | ERROR: {res['error']}")
                    continue

                fval = res.get("f")
                if isinstance(fval, list):
                    fval = fval[0]

                print(f"{solver_name:12s} | f*={fval:<10.3e} | evals={res.get('n_eval')}")

                history = res.get("best")
                if history:
                    # assuming history is a flat list of best-so-far values per evaluation
                    filename = os.path.join(solver_folder, f"results_{solver_name}{instance}.dat")
                    with open(filename, "w") as f:
                        for val in history:
                            try:
                                num = float(val)
                                f.write(f"{num:10.3e}\n")
                            except (ValueError, TypeError):
                                f.write("nan\n")
                else:
                    print(f"ERROR: empty output for {solver_name}")

# Gets the max row count across the results of all 10 runs of a solver
def get_max_row_count(func_folder, solver, instances):
    max_rows = 0

    for i in range(1, instances+1):
        results_file = os.path.join(func_folder, solver.name, f"results_{solver.name}{i}.dat")
        num_rows = 0
        with open(results_file, 'r') as f:
            num_rows = sum(1 for _ in f)
        if num_rows > max_rows:
            max_rows = num_rows
    return max_rows

# Creates a .dat file for the average running minimum for each solver
def solver_sum(solvers, func_folder, instances):
    for s in solvers:
        solver_folder = os.path.join(func_folder, s.name)
        max_rows = get_max_row_count(func_folder, s, instances)
        all_runs_val = []

        for instance in range(1, instances + 1):
            results_file = os.path.join(solver_folder, f"results_{s.name}{instance}.dat")
            if not os.path.exists(results_file):
                print(f"[solver_sum] Missing file {results_file}; padding with NaNs")
                all_runs_val.append([np.nan] * max_rows)
                continue

            # Read values line-wise, ignoring blank lines
            values = []
            with open(results_file, "r") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        values.append(float(stripped.split()[0]))
                    except ValueError:
                        values.append(np.nan)

            # Pad to max_rows
            if len(values) < max_rows:
                values.extend([np.nan] * (max_rows - len(values)))
            else:
                values = values[:max_rows]

            all_runs_val.append(values)

        # Transpose so shape is (FE, instance)
        all_runs_np = np.array(all_runs_val).T  # shape: (FE, instance)

        output_file = os.path.join(solver_folder, f"{s.name}_avg_running_minima.tsv")
        with open(output_file, "w") as out_f:
            header_parts = ["FE"] + [f"Run {i}" for i in range(1, instances + 1)] + ["Average"]
            out_f.write("\t".join(header_parts) + "\n")

            for fe_idx, row in enumerate(all_runs_np, start=1):
                avg = np.nanmean(row)
                row_strs = [f"{val:.5e}" if not np.isnan(val) else "nan" for val in row]
                avg_str = f"{avg:.5e}" if not np.isnan(avg) else "nan"
                out_f.write(f"{fe_idx}\t" + "\t".join(row_strs) + f"\t{avg_str}\n")

        print(f"[solver_sum] wrote {output_file}")

# creates a .dat file for the best minima for each solver
def best_minima_per_instance_per_solver(solvers, func_name, instances):
    """
    For each solver, write a file in its subfolder under func_folder named
    {solver}_best_minima_per_instance.tsv containing:
      Instance    {solver.name}
    with the best (minimum) value from each instance.
    """
    for s in solvers:
        solver_folder = os.path.join(func_name, s.name)
        os.makedirs(solver_folder, exist_ok=True)

        output_file = os.path.join(solver_folder, f"{s.name}_best_minima_per_instance.tsv")

        with open(output_file, "w") as out_f:
            # Header: Instance <tab> SolverName
            out_f.write(f"Instance\t{s.name}\n")

            for instance in range(1, instances + 1):
                results_file = os.path.join(solver_folder, f"results_{s.name}{instance}.dat")
                best = float("nan")
                if os.path.exists(results_file):
                    try:
                        with open(results_file, "r") as f:
                            # parse numeric lines
                            values = []
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    values.append(float(line.split()[0]))  # first token
                                except ValueError:
                                    continue
                            if values:
                                best = np.nanmin(values)
                    except Exception as e:
                        print(f"Warning: failed to read {results_file}: {e}")
                else:
                    print(f"Warning: missing result file {results_file}")

                best_str = f"{best:.5e}" if not np.isnan(best) else "nan"
                out_f.write(f"{instance}\t{best_str}\n")

if __name__ == "__main__":
    func_name = "gseven10"    # options are rosenbrock{ndim}, michalewicz{ndim}, shubert{ndim}, or gseven10
    instances = 10
    os.makedirs(func_name, exist_ok=True)

    # instantiate the specifications for that function name
    config = objective_configs[func_name]
    
    # matching configurations from dictionary to variables for the problem
    ndim = config["ndim"]
    budget = config["budget"]
    bounds = config["bounds"] * ndim  # expand per-dim
    obj_func = config["func"]
    has_constraints = config["has_constraints"] # boolean value
    constraints = config["constraints"]
    
    # list solvers being used
    if has_constraints:
        solvers = [SciPyDE(budget), PymooGA(budget), AxBO(budget), OMADSSolver(budget)]
    else:
        solvers = [NGIoh(budget), OMADSSolver(budget), PymooGA(budget), SciPyDA(budget), SciPyDE(budget), SciPyDIRECT(budget)] 
    
    # run all the necessary functions
    runopt(func_name, func_name, obj_func, bounds, constraints, has_constraints, ndim, budget, solvers, instances)
    solver_sum(solvers, func_name, instances)
    best_minima_per_instance_per_solver(solvers, func_name, instances)
