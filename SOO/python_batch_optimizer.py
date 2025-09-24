# python_batch_optmizers, the main script that sets up file paths, runs the solvers and post-processes the data, for all SOO problems

from python_solvers import Problem, SciPyDE, SciPyDA, SciPyDIRECT, NGIoh, PymooGA, OMADSSolver #, AxBO, SMTEGO
from python_definitions import objective_configs

import os, time
import numpy as np

# Handles folder/files setup for each solver, runs each solver 10 times
def runopt(func_name, func_folder, obj_func, bounds, ndim, budget, solvers, instances):
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
                    objective=obj_func,
                )

                setattr(s, "log_dir", solver_folder)
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

# Returns the max row count across the results of all 10 runs of a solver
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

# Creates a .dat file with the average running minimum for each solver, in each function folder
def solver_sum(solvers, func_folder, instances):
    for s in solvers:
        solver_folder = os.path.join(func_folder, s.name)
        max_rows = get_max_row_count(func_folder, s, instances)
        all_runs_val = []

        for instance in range(1, instances + 1):
            results_file = os.path.join(solver_folder, f"results_{s.name}{instance}.dat")
            if not os.path.exists(results_file):
                print(f"[SOO_solver_sum] Missing file {results_file}; padding with NaNs")
                all_runs_val.append([np.nan] * max_rows)
                continue

            # Read numeric values line-wise, ignoring blanks/non-numeric
            values = []
            with open(results_file, "r") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    tok = stripped.split()[0]
                    try:
                        values.append(float(tok))
                    except ValueError:
                        values.append(np.nan)

            # Enforce running minima (non-increasing sequence)
            if values:
                arr = np.array(values, dtype=float)
                # Treat NaNs as +inf so they don't reduce the running min
                arr_clean = np.where(np.isfinite(arr), arr, np.inf)
                run_min = np.minimum.accumulate(arr_clean)
                # Convert +inf back to NaN where no finite value seen yet
                run_min[np.isinf(run_min)] = np.nan
                values = run_min.tolist()

            # Pad/trim to max_rows
            if len(values) < max_rows:
                values.extend([np.nan] * (max_rows - len(values)))
            else:
                values = values[:max_rows]

            all_runs_val.append(values)

        # Shape: (FE, instances)
        all_runs_np = np.array(all_runs_val, dtype=float).T

        output_file = os.path.join(solver_folder, f"{s.name}_avg_running_minima.tsv")
        with open(output_file, "w") as out_f:
            header = ["FE"] + [f"Run {i}" for i in range(1, instances + 1)] + ["Average"]
            out_f.write("\t".join(header) + "\n")

            for fe_idx, row in enumerate(all_runs_np, start=1):
                avg = np.nanmean(row)
                row_strs = [f"{v:.5e}" if np.isfinite(v) else "nan" for v in row]
                avg_str = f"{avg:.5e}" if np.isfinite(avg) else "nan"
                out_f.write(f"{fe_idx}\t" + "\t".join(row_strs) + f"\t{avg_str}\n")

        print(f"[solver_sum] wrote {output_file}")

# Creates a .dat file with the best minima for each solver, in each function folder
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
        print(f"[best_minima_per_instance_solver] wrote {output_file}")

# Loops through the functions and configures the variables for each aspect of the functions before running the optimization loop and post-processing functions
def main(func_names, instances=10):
    for func_name in func_names:
        os.makedirs(func_name, exist_ok=True)

        # Instantiate the specifications for that function name
        config = objective_configs[func_name]
        
        # Match configurations from dictionary to variables for the problem
        ndim = config["ndim"]
        budget = config["budget"]
        bounds = config["bounds"] * ndim 
        obj_func = config["func"]
        
        # solvers = [AxBO(budget), SMTEGO(budget)]
        solvers = [NGIoh(budget), OMADSSolver(budget), PymooGA(budget), SciPyDA(budget), SciPyDE(budget), SciPyDIRECT(budget)] 
        
        # Run optimization loop, data congregation for convergence plot, and data congregation for boxplot
        runopt(func_name, func_name, obj_func, bounds, ndim, budget, solvers, instances)
        solver_sum(solvers, func_name, instances)
        best_minima_per_instance_per_solver(solvers, func_name, instances)
 
if __name__ == "__main__":
    func_names = ["rosenbrock10", "michalewicz10", "schubert10", "michalewicz30", "schubert30", "michalewicz60", "schubert60", "rosenbrock50"]
    main(func_names)