# dakota_batch_optimizer.py, runs the Dakota solvers on a MOO problem (defined by func name + dimension in main()). Setup for driver and .in files has already been completed.

import os, re, subprocess, time, shutil
import numpy as np

"""
- `modify_and_run()` edits the .in for each run, runs Dakota
- `run_optimization()` loops over methods & runs, times each, and writes {func_folder}/{method}_time.tsv.
"""

"""
Create method subfolders and copy driver & input template into each.
"""
def setup(func_folder: str, func: str, method_types: list):
    driver_file = os.path.join("drivers_and_templates", f"{func}_driver.py")
    if not os.path.isfile(driver_file):
        raise FileNotFoundError(f"Missing driver file: {driver_file}")

    for method in method_types:
        method_folder = os.path.join(func_folder, method)
        os.makedirs(method_folder, exist_ok=True)

        # copy the driver script
        shutil.copy(driver_file, method_folder)

        # copy the Dakota input template
        in_file = os.path.join("drivers_and_templates", f"{method}_{func}.in")
        if not os.path.isfile(in_file):
            raise FileNotFoundError(f"Missing input file: {in_file}")
        shutil.copy(in_file, method_folder)

"""
Modify the Dakota .in, run Dakota
"""
def modify_and_run(func_folder: str, method: str, run_idx: int, edit_lines: list):
    method_folder = os.path.join(func_folder, method)
    dakota_in  = f"{method}_{os.path.basename(func_folder)}.in"
    dakota_out = f"{method}_{os.path.basename(func_folder)}{run_idx}.out"
    in_path = os.path.join(method_folder, dakota_in)

    # read and patch .in
    with open(in_path, 'r') as f:
        lines = f.readlines()
    for li in edit_lines:
        if li < len(lines):
            if 'tabular_data_file' in lines[li]:
                lines[li] = re.sub(
                    r"'[^']+\.dat'",
                    f"'results_{method}{run_idx}.dat'",
                    lines[li]
                )
            elif 'seed' in lines[li]:
                lines[li] = re.sub(
                    r'(seed\s*=\s*)\d+',
                    lambda m: f"{m.group(1)}{run_idx}",
                    lines[li]
                )
    with open(in_path, 'w') as f:
        f.writelines(lines)

    # run Dakota
    subprocess.run(
        ['dakota', '-i', dakota_in, '-o', dakota_out],
        cwd=method_folder,
        check=True
    )

"""
For each method and run, modify input, run Dakota, time it,
and write a TSV under {func_folder}/{method}_time.tsv.
"""
def run_optimization(func_folder: str, method_types: list, instances: int):
    os.makedirs(func_folder, exist_ok=True)
    edit_lines = [4, 8]    # Lines to be modified for changing the seed of each instance
    for method in method_types:
        time_file = os.path.join(func_folder, method, f"{method}_time.tsv")

        with open(time_file, 'w') as tf:
            tf.write("Run\tTime(s)\n")
            for j in range(1, instances + 1):
                start = time.time()
                modify_and_run(
                    func_folder=func_folder,
                    method=method,
                    run_idx=j,
                    edit_lines=edit_lines
                )
                elapsed = time.time() - start
                tf.write(f"{j}\t{elapsed:.2f}\n")
                print(f"[{method}] run {j} completed in {elapsed:.2f}s")


"""
Reads each results_{method}{i}.dat, skips any header or blank lines,
splits on whitespace, and pulls out the last two tokens as floats.
Returns an (N×2) numpy array of all points.
"""
def collect_objectives(func_folder, method, instances=10):
    pts = []
    for i in range(1, instances+1):
        path = os.path.join(func_folder, method, f"results_{method}{i}.dat")
        if not os.path.isfile(path):
            print(f"  skipped missing file: {path}")
            continue
        with open(path) as f:
            for ln in f:
                ln = ln.strip()
                # skip blank lines or Dakota comments
                if not ln or ln.startswith('%') or ln.lower().startswith('eval_id'):
                    continue
                toks = ln.split()
                # need at least two columns
                if len(toks) < 2:
                    continue
                try:
                    f1, f2 = float(toks[-2]), float(toks[-1])
                except ValueError:
                    continue
                pts.append((f1, f2))
    return np.array(pts)

"""
Given an (M×2) array of objectives, returns only non-dominated rows.
"""
def pareto_filter(points):
    if points.size == 0:
        return points
    dominated = np.zeros(points.shape[0], dtype=bool)
    for i, p in enumerate(points):
        # any other point strictly better in both?
        better = np.all(points <= p, axis=1) & np.any(points < p, axis=1)
        if np.any(better):
            dominated[i] = True
    return points[~dominated]

"""
Writes header + f1\tf2 lines to {method}_pareto.dat
"""
def write_pareto(func_folder, method, pareto_pts, out_name=None):
    if out_name is None:
        out_name = f"{method}_pareto.tsv"
    out_path = os.path.join(func_folder, method, out_name)
    with open(out_path, 'w') as f:
        f.write("f1\tf2\n")
        for f1, f2 in pareto_pts:
            f.write(f"{f1:.10g}\t{f2:.10g}\n")
    print(f"[{method}] Wrote {len(pareto_pts)} Pareto points → {out_path}")

def main(func, instances, method_types, setup_flag):
    if setup_flag:
        setup(func, func, method_types)
        return

    print("\nRunning optimization loop:")
    run_optimization(func, method_types, instances) 
 
    print("\nCollecting & writing Pareto fronts:")
    for method in method_types:
        print(f" • {method}")
        pts = collect_objectives(func, method, instances)
        pareto = pareto_filter(pts)
        write_pareto(func, method, pareto)

if __name__ == '__main__':
    func_name      = 'zdt1'                       # options are zdt1, zdt2, zdt3, zdt6, osy and geartrain
    instances = 10
    setup_flag = False
    methods   = ['ea', 'moga']

    main(func_name, instances, methods, setup_flag)


