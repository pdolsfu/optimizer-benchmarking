# python_batch_optmizers, the main script that sets up file paths, runs the solvers and post-processes the data from Python solvers and OASIS, for all MOO problems, and runs the external script for OMADS optimization

import numpy as np
import os, time
import numpy as np

from python_solvers import Problem, PymooNSGA2, PymooMOEAD, PymooSPEA2
import python_OMADS_MOO
from python_definitions import objective_configs

"""
For each solver in `solvers`:
    - runs `instances` seeds,
    - saves results_F and results_X in func_name/solver.name/,
    - writes func_name/solver.name/solver.name_time.tsv
    with columns [Run, <solver.name>] listing each seed’s elapsed time.
"""

def runopt_multi(func_name, obj_func, bounds, constraints, has_constraints, ndim, solvers, instances):
    print(f"--- {func_name} (Multi-Objective) | {instances} instances ---")

    for s in solvers:
        folder_path = os.path.join(func_name, s.name)
        os.makedirs(folder_path, exist_ok=True)

        method_times = []
        for instance in range(1, instances+1):
            prob = Problem(
                dim=ndim,
                instance=instance,
                bounds=bounds,
                constraints=(constraints if has_constraints else None),
                objective=obj_func
            )

            start = time.time()
            res = s.solve(prob)
            elapsed = time.time() - start
            method_times.append(elapsed)

            if "error" in res:
                print(f"{s.name:12s} | ERROR: {res['error']}")
                continue

            print(f"{s.name:12s} | #Front={len(res['F'])} | evals={res['n_eval']}")

            # Save Pareto front F and decision vars X
            fn_F = os.path.join(folder_path, f"results_{s.name}{instance}_F.dat")
            fn_X = os.path.join(folder_path, f"results_{s.name}{instance}_X.dat")
            np.savetxt(fn_F, res['F'], fmt="%.5e", delimiter="\t")
            np.savetxt(fn_X, res['X'], fmt="%.5e", delimiter="\t")

        # Write per-solver timing file
        time_path = os.path.join(folder_path, f"{s.name}_time.tsv")
        with open(time_path, "w") as tf:
            # header with solver name as second column
            tf.write(f"Run\t{s.name}\n")
            for run_idx, t in enumerate(method_times, start=1):
                tf.write(f"{run_idx}\t{t:.5f}\n")

        print(f"[{s.name}] Wrote timing → {time_path}")

# checks an array for dominated solutions
def is_dominated(y, others):
    return np.any(np.all(others <= y, axis=1) & np.any(others < y, axis=1))

# does the Pareto preparation for Python solvers, creates a .dat file
def filter_and_write_pareto(func_folder, solvers, instances):
    """
    For each solver in `solvers`:
      - Load all results_{solver.name}{i}_F.dat
      - Remove dominated points
      - Sort by f2 (primary) then f1 (secondary)
      - Write to {func_folder}/{solver.name}/{solver.name}_pareto.dat
    """
    for solver in solvers:
        all_runs = []
        solver_dir = os.path.join(func_folder, solver.name)

        # 1) load each instance's Pareto front
        for inst in range(1, instances + 1):
            path = os.path.join(solver_dir, f"results_{solver.name}{inst}_F.dat")
            try:
                data = np.loadtxt(path, delimiter='\t')
            except Exception as e:
                print(f"Warning: could not read {path}: {e}")
                continue
            if data.ndim == 1:
                data = data.reshape(1, -1)
            all_runs.append(data)

        if not all_runs:
            print(f"[{solver.name}] no data found, skipping.")
            continue

        # 2) stack and filter dominated
        F = np.vstack(all_runs)
        keep = [not is_dominated(f, np.delete(F, i, axis=0)) for i, f in enumerate(F)]
        pareto = F[keep]

        # 3) sort by f2 then f1
        order = np.lexsort((pareto[:,0], pareto[:,1]))
        pareto_sorted = pareto[order]

        # 4) write out
        out_path = os.path.join(solver_dir, f"{solver.name}_pareto.tsv")
        header = "\t".join(f"f{i+1}" for i in range(pareto_sorted.shape[1])) + "\n"
        with open(out_path, 'w') as fout:
            fout.write(header)
            for row in pareto_sorted:
                fout.write("\t".join(f"{v:.10g}" for v in row) + "\n")

        print(f"[{solver.name}] Wrote {len(pareto_sorted)} points to {out_path}")


# does the Pareto preparation for Python solvers, creates a .dat file
def filter_and_write_oasis(func_folder, instances, ndim, tol=1e-6):
    """
    Read all results_OASIS{inst}.csv in func_folder/OASIS,
    keep only feasible rows (by is_feasible flag and, for constrained problems,
    by all c_i >= 0),
    collect (f1,f2) using problem-specific column positions,
    recompute global Pareto front, sort by f2 then f1,
    and write to func_folder/OASIS/OASIS_pareto.tsv.

    Layouts:
      - Unconstrained (zdt*, geartrain): x1..x_d, f1, f2, is_feasible, pareto_frontier
      - OSY (constrained):              x1..x_d, f1, f2, c1..c6, is_feasible, pareto_frontier
    """
    def _truthy(s):
        s = str(s).strip().lower()
        return s in ("1", "true", "t", "yes", "y")

    # Determine problem name and whether it has constraints from your config
    problem = os.path.basename(os.path.normpath(func_folder)).lower()
    has_constr = objective_configs.get(problem, {}).get("has_constraints", False)

    # OSY specifics: 6 constraints AFTER f1,f2
    n_constr = 6 if (problem == "osy" and has_constr) else 0

    # Column indices by schema
    if problem == "osy" and n_constr > 0:
        # Row: x1..x_d, f1, f2, c1..c6, is_feasible, pareto_frontier
        f1_idx = ndim
        f2_idx = ndim + 1
        constr_idx_start = ndim + 2
        feasible_idx = constr_idx_start + n_constr
    else:
        # Row: x1..x_d, f1, f2, is_feasible, pareto_frontier
        f1_idx = ndim
        f2_idx = ndim + 1
        constr_idx_start = None
        feasible_idx = ndim + 2

    all_pts = []
    oasis_dir = os.path.join(func_folder, "OASIS")
    for inst in range(1, instances + 1):
        path = os.path.join(oasis_dir, f"results_OASIS{inst}.csv")
        try:
            with open(path, 'r', newline='') as fin:
                lines = fin.readlines()
        except FileNotFoundError:
            print(f"Warning: {path} not found, skipping.")
            continue
        if not lines:
            continue

        # header + rows
        rows = lines[1:]

        for line in rows:
            cols = [c.strip() for c in line.strip().split(',')]
            if len(cols) <= max(f1_idx, f2_idx):
                continue

            # parse f1,f2
            try:
                f1 = float(cols[f1_idx])
                f2 = float(cols[f2_idx])
            except Exception:
                continue

            # determine feasibility
            feasible = True

            # (a) if is_feasible flag present, honor it
            if 0 <= feasible_idx < len(cols) and cols[feasible_idx] != "":
                feasible = _truthy(cols[feasible_idx])

            # (b) if constraints present (OSY), require all c_i >= 0 with tolerance
            if feasible and n_constr > 0 and constr_idx_start is not None:
                feas_c = True
                for k in range(n_constr):
                    idx = constr_idx_start + k
                    if idx >= len(cols):
                        feas_c = False
                        break
                    try:
                        ck = float(cols[idx])
                    except Exception:
                        feas_c = False
                        break
                    if ck > tol:
                        feas_c = False
                        break
                feasible = feasible and feas_c

            if feasible:
                all_pts.append([f1, f2])

    if not all_pts:
        print("No OASIS feasible points loaded—nothing to write.")
        return

    F = np.array(all_pts)
    keep = [not is_dominated(f, np.delete(F, i, axis=0)) for i, f in enumerate(F)]
    pareto = F[keep]

    # sort by f2 then f1
    order = np.lexsort((pareto[:, 0], pareto[:, 1]))
    pareto_sorted = pareto[order]

    out_path = os.path.join(func_folder, "OASIS", "OASIS_pareto.tsv")
    with open(out_path, 'w', newline='') as fout:
        fout.write("f1\tf2\n")
        for f1, f2 in pareto_sorted:
            fout.write(f"{f1:.10g}\t{f2:.10g}\n")

    print(f"Wrote {len(pareto_sorted)} Pareto points to {out_path}")

def main(func_name, instances=10):
    os.makedirs(func_name, exist_ok=True) 

    # instantiate the specifications for that function name
    config = objective_configs[func_name]
    
    # matching configurations from dictionary to variables for the problem
    ndim = config["ndim"]
    budget = config["budget"]
    raw_bounds = config["bounds"]
    bounds = raw_bounds if len(raw_bounds) > 1 else raw_bounds * ndim
    obj_func = config["func"]
    has_constraints = config["has_constraints"] # boolean value
    constraints = config["constraints"]   

    solvers = [PymooNSGA2(budget), PymooMOEAD(budget), PymooSPEA2(budget)]
    
    # run all the solvers
    #runopt_multi(func_name, obj_func, bounds, constraints, has_constraints, ndim, solvers, instances)
    python_OMADS_MOO.main(str(func_name.upper()), 10)

    # post-processing 
    filter_and_write_pareto(func_name, solvers, instances)
    filter_and_write_oasis(func_name, instances, ndim, tol=1e-6)

if __name__ == "__main__":
    func_names = ["zdt1", "zdt2", "zdt3"]
    # func_names = ["zdt1", "zdt2", "zdt3", "zdt6", "geartrain", "osy"]

    for func in func_names:
        main(func, 10)
