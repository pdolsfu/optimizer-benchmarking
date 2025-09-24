# oasis_processor.py, copies and post-processes the data from oasis_raw_results to create relevant .tsv files for plotting and numerical comparison

import os
import re
import numpy as np
import csv

FILENAME_RE = re.compile(
    r"run\s+\d+\s+.+?\s+d(?P<ndim>\d+)\s*-\s*NFE\s*(?P<nfe>\d+)-Stage\s*(?P<stage>\d+)-OASIS AI Worker\s*(?P<worker>\d+)-result-table\.csv",
    flags=re.IGNORECASE
)


def process_oasis(problem_folders, stage=1, workers=range(1, 11)):
    for problem in problem_folders:
        base_in_dir = os.path.join("oasis_raw_results", problem)
        if not os.path.isdir(base_in_dir):
            print(f"[OASIS] Skipping missing problem directory: {base_in_dir}")
            continue

        # Discover files and group by worker
        worker_files = {}
        inferred_ndim = None
        for fname in os.listdir(base_in_dir):
            if not fname.lower().endswith("-result-table.csv"):
                continue
            m = FILENAME_RE.match(fname)
            if not m:
                print(f"[OASIS] Filename did not match expected pattern: {fname}")
                continue
            file_stage = int(m.group("stage"))
            if file_stage != stage:
                continue
            worker = int(m.group("worker"))
            ndim = int(m.group("ndim"))
            if inferred_ndim is None:
                inferred_ndim = ndim
            elif inferred_ndim != ndim:
                print(f"[OASIS] Warning: inconsistent ndim in {fname} (was {inferred_ndim}, now {ndim})")
            worker_files[worker] = os.path.join(base_in_dir, fname)

        solver_name = "OASIS"
        out_dir = os.path.join(problem, solver_name)
        os.makedirs(out_dir, exist_ok=True)

        if inferred_ndim is None:
            print(f"[OASIS] No valid input files found for problem {problem}; skipping.")
            continue

        # Read each worker, extract objective column at index inferred_ndim (0-based)
        all_runs = []
        sorted_workers = sorted(workers)
        for worker in sorted_workers:
            filepath = worker_files.get(worker)
            objectives = []
            if not filepath or not os.path.exists(filepath):
                print(f"[OASIS] Missing file for worker {worker} in {problem}; filling with empty run")
                all_runs.append([])
                continue

            try:
                with open(filepath, newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    header = next(reader, None)  # skip header if present
                    for row in reader:
                        if len(row) <= inferred_ndim:
                            continue
                        try:
                            val = float(row[inferred_ndim])
                            objectives.append(val)
                        except ValueError:
                            objectives.append(np.nan)
            except Exception as e:
                print(f"[OASIS] Failed to read {filepath}: {e}")
                all_runs.append([])
                continue

            # Compute running minima (lower is better)
            running_minima = []
            current_min = np.inf
            for v in objectives:
                if not np.isnan(v) and v < current_min:
                    current_min = v
                running_minima.append(current_min if current_min != np.inf else np.nan)

            all_runs.append(running_minima)

            # Also write per-instance results file (like results_OASIS{worker}.dat)
            inst_file = os.path.join(out_dir, f"results_{solver_name}{worker}.dat")
            with open(inst_file, "w") as f_out:
                for val in running_minima:
                    if np.isnan(val):
                        f_out.write("nan\n")
                    else:
                        f_out.write(f"{val:.5e}\n")

        # Pad to uniform length
        max_len = max((len(r) for r in all_runs), default=0)
        padded_runs = []
        for run in all_runs:
            if len(run) < max_len:
                padded = run + [np.nan] * (max_len - len(run))
            else:
                padded = run[:max_len]
            padded_runs.append(padded)

        # ----- Write average running minima file as TSV -----
        avg_file = os.path.join(out_dir, f"{solver_name}_avg_running_minima.tsv")

        with open(avg_file, "w") as out_f:
            header_parts = ["FE"] + [f"Run {w}" for w in sorted_workers] + ["Average"]
            out_f.write("\t".join(header_parts) + "\n")
            all_np = np.array(padded_runs).T  # shape (FE, worker)
            for fe_idx, row in enumerate(all_np, start=1):
                avg = np.nanmean(row)
                row_strs = [f"{val:.5e}" if not np.isnan(val) else "nan" for val in row]
                avg_str = f"{avg:.5e}" if not np.isnan(avg) else "nan"
                out_f.write(f"{fe_idx}\t" + "\t".join(row_strs) + f"\t{avg_str}\n")

        # ----- Write best minima per instance file -----
        best_file = os.path.join(out_dir, f"{solver_name}_best_minima_per_instance.tsv")
        with open(best_file, "w") as out_f:
            out_f.write("Instance\t" + solver_name + "\n")
            for idx, run in enumerate(padded_runs, start=1):
                if run:
                    best = np.nanmin(run)
                else:
                    best = np.nan
                best_str = f"{best:.5e}" if not np.isnan(best) else "nan"
                out_f.write(f"{idx}\t{best_str}\n")

        # ----- Write time file (placeholder) -----
        time_file = os.path.join(out_dir, f"{solver_name}_time.tsv")
        with open(time_file, "w") as out_f:
            out_f.write("Run\tTime\n")
            for worker in sorted_workers:
                out_f.write(f"{worker}\tnan\n")

        print(f"[OASIS] Processed problem '{problem}' into {out_dir}")

problem_folders = ["rosenbrock50", "schubert30", "schubert60", "michalewicz30", "michalewicz60"]
process_oasis(problem_folders, stage=1, workers=range(1, 11))