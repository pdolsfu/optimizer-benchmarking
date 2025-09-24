# dakota_batch_optimizer.py, runs the Dakota solvers on the SOO problems (defined by func name + dimension in main()). Setup for driver and .in files has already been completed.

import numpy as np
import os, re, subprocess, time, glob
import shutil

'''
Modify the Dakota .in file and run Dakota.
'''

def modify_and_run(func, ndim, dakota_input, dakota_output, line_indices_to_edit, index_of_number, method):
    method_folder = method
    dakota_input_path = os.path.join(f"{func}{ndim}", method_folder, dakota_input)
    print(f"[modify_and_run] modifying {dakota_input_path}")

    if not os.path.isfile(dakota_input_path):
        raise FileNotFoundError(f"Input file not found: {dakota_input_path}")

    # Read input file
    with open(dakota_input_path, "r") as f:
        lines = f.readlines()

    # Modify specified lines
    for line_index in line_indices_to_edit:
        if line_index < len(lines):
            parts = lines[line_index].strip().split()
            if not parts:
                print(f"Line {line_index+1} empty; skipping")
                continue
             
            # Sets the last group of characters to be the "token" to be replaced
            last_token = parts[-1]
            
            # Replacement of the line with the .dat file specification
            if last_token.startswith("'") and last_token.endswith(".dat'"):
                stripped = last_token.strip("'")
                new_filename = re.sub(
                    r"(\D+)(\d+)(\.dat)",
                    lambda m: f"{m.group(1)}{index_of_number}{m.group(3)}",
                    stripped,
                )
                parts[-1] = f"'{new_filename}'"
                lines[line_index] = "  ".join(parts) + "\n"
            else:
                # Numeric replacement for the line "seeds = X"
                try:
                    float(last_token)
                    parts[-1] = str(index_of_number)
                    lines[line_index] = "  ".join(parts) + "\n"
                except ValueError:
                    print(f"Skipping line {line_index+1}: unrecognized token '{last_token}'")
        else:
            print(f"Line {line_index+1} does not exist; skipping")

    # Write back entire file with modified lines
    with open(dakota_input_path, "w") as f:
        f.writelines(lines)

    # Run Dakota in the f"{problem}/{method}" folder
    out_folder = os.path.join(f"{func}{ndim}", method_folder)
    try:
        subprocess.run(
            ["dakota", "-i", dakota_input, "-o", dakota_output],
            check=True,
            cwd=out_folder,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running Dakota for {method} on input {dakota_input}: {e}")
        raise

    # Cleanup special SBO outputs
    if method.lower() == "sbo":
        pattern = os.path.join(method_folder, "finaldatatruth*")
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
            except FileNotFoundError:
                pass

"""
Run each method `instances` times, modifying and executing its Dakota input.
Writes per-solver time TSV files: {method}/{method}_time.tsv.
Assumes append_running_min(path, ndim) exists and produces results_{method}{j}.dat.
"""
def run_optimization(func, ndim, method_types, instances):
    func_folder = f"{func}{ndim}"
    for method in method_types:
        method_folder = os.path.join(func_folder, method)
        os.makedirs(method_folder, exist_ok=True)
        time_file = os.path.join(method_folder, f"{method}_time.tsv")

        # determine which lines to edit (results_{method}{instance}.dat and seed specifications)
        if method == "sbo":
            lines_to_change = [4, 23]
        elif method == "NCSU":
            lines_to_change = [4]
        else:
            lines_to_change = [4, 8]

        # prepare per-solver time file and run a .in file
        with open(time_file, "w") as tf:
            tf.write("Run\tTime\n")
            for j in range(1, instances + 1):
                dakota_input_file = f"{func}{ndim}_{method}.in"
                dakota_output_file = f"{func}{ndim}_{method}{j}.out"
                start = time.time()
                modify_and_run(
                    func = func,
                    ndim = ndim,
                    dakota_input=dakota_input_file,
                    dakota_output=dakota_output_file,
                    line_indices_to_edit=lines_to_change,
                    index_of_number=j,
                    method=method,
                )
                end = time.time()
                elapsed = end - start
                tf.write(f"{j}\t{elapsed:.2f}\n")

"""
Append the running minima at each NFE to the .dat file of each instance of each solver
"""
def append_running_min(func, ndim, method_types, instances):
    func_folder = f"{func}{ndim}"
    for method in method_types:
        for j in range(1, instances + 1):
            source_file = os.path.join(func_folder, method, f"results_{method}{j}.dat")
            if not os.path.isfile(source_file):
                print(f"[append_running_min] Missing file: {source_file}")
                continue

            with open(source_file, "r") as f:
                lines = f.readlines()

            if len(lines) < 3:
                print(f"[append_running_min] Not enough lines in {source_file}; skipping")
                continue

            header1 = lines[0]
            header2 = lines[1]
            data_lines = lines[2:]

            try:
                data = np.genfromtxt(source_file, skip_header=2)
                if data.ndim == 1:
                    data = data.reshape(-1, data.size)
            except Exception as e:
                print(f"[append_running_min] Failed to parse {source_file}: {e}")
                continue

            obj_col_idx = ndim + 2
            if data.size == 0 or data.shape[1] <= obj_col_idx:
                print(f"[append_running_min] Not enough columns in {source_file} for objective at index {obj_col_idx}")
                continue

            obj_values = data[:, obj_col_idx]
            running_min = np.minimum.accumulate(np.where(np.isnan(obj_values), np.inf, obj_values))
            running_min = np.where(running_min == np.inf, np.nan, running_min)

            with open(source_file, "w") as f:
                f.write(header1.rstrip("\n") + "\tbest_so_far\n")
                f.write(header2)
                for i, original_line in enumerate(data_lines):
                    original_line = original_line.rstrip("\n")
                    best = running_min[i] if i < len(running_min) else np.nan
                    best_str = "nan" if np.isnan(best) else f"{best:.5e}"
                    f.write(f"{original_line}\t{best_str}\n")

"""
Create a file for each solver with average running minima data stacked in a column, empty values at a FE of instances are treated as nan
"""
def create_running_minima_files(func, ndim, method_types, instances):
    target_col_index = ndim + 3          # Locates the column with the running minima
    
    for method in method_types:
        method_folder = os.path.join(f"{func}{ndim}", method)
        output_file = os.path.join(method_folder, f"{method}_avg_running_minima.tsv") # name of the output file
        file_suffix_range = range(1, instances + 1)

        # Determine max number of data lines of the solver across instances (after 2 headers)
        max_fes = 0
        for i in file_suffix_range:
            fname = os.path.join(method_folder, f"results_{method}{i}.dat")
            if os.path.exists(fname):
                with open(fname, 'r') as f:
                    lines = f.readlines()
                max_fes = max(max_fes, max(0, len(lines) - 2))
            else:
                print(f"Path doesn't exist for: \"results_{method}{i}.dat\"")


        with open(output_file, "w") as out_f:
            header = ["FE"] + [f"Run {i}" for i in file_suffix_range] + ["Average"]
            out_f.write("\t".join(header) + "\n")

            # running minimum for the Average column
            prev_avg = None  # None means "no previous"

            for fe in range(1, max_fes + 1):  # FE starts at 1
                per_instance_vals = []
                for i in file_suffix_range:
                    fname = os.path.join(method_folder, f"results_{method}{i}.dat") # raw data files from Dakota
                    if not os.path.exists(fname):
                        per_instance_vals.append(np.nan)
                        continue
                    
                    with open(fname, 'r') as f:
                        lines = f.readlines()

                    data_idx = 2 + (fe - 1)  # skip two headers, zero-based offset
                    if data_idx >= len(lines):
                        per_instance_vals.append(np.nan)
                        continue

                    tokens = lines[data_idx].strip().split()
                    
                    if len(tokens) <= target_col_index:
                        print(f"[DEBUG] {fname} line {data_idx+1} has only {len(tokens)} tokens, "
                            f"expected >= {target_col_index+1}")
                    try:
                        val = float(tokens[target_col_index])
                        per_instance_vals.append(val)
                    except (IndexError, ValueError) as e:
                        print(f"[DEBUG] Could not parse value from {fname}, line {data_idx+1}, tokens={tokens}, error={e}")
                        per_instance_vals.append(np.nan)

                # compute average across instances at this FE
                avg = np.nanmean(per_instance_vals)

                # enforce running minimum on the Average column
                if prev_avg is None or np.isnan(prev_avg):
                    out_avg = avg  # first row: take whatever we have (can be NaN)
                else:
                    if np.isnan(avg):
                        out_avg = prev_avg  # Repeat previous if current is NaN
                    else:
                        out_avg = min(prev_avg, avg)  # keep running minima

                prev_avg = out_avg
                if np.isnan(out_avg):
                    print(f"[DEBUG] Average is NaN at FE={fe}, method={method}, per_instance_vals={per_instance_vals}")

                # write row
                row = [str(fe)] + [("nan" if np.isnan(v) else str(v)) for v in per_instance_vals]
                row.append("nan" if np.isnan(out_avg) else str(out_avg))
                out_f.write("\t".join(row) + "\n")

        print(f"[create_running_minima_files] wrote {output_file}")

"""
Create a file for each solver with the best minima for each instance stacked in a column
"""
def create_absolute_minima_per_instance(func, ndim, method_types, instances):
    for method in method_types:
        method_folder = os.path.join(f"{func}{ndim}", method)
        output_file = os.path.join(method_folder, f"{method}_best_minima_per_instance.tsv")

        with open(output_file, "w") as out_f:
            out_f.write(f"Instance\t{method}\n")
            method = method.lower()
            for i in range(1, instances + 1):
                fname = os.path.join(method_folder, f"results_{method}{i}.dat")
                best = np.nan
                if os.path.exists(fname):
                    try:
                        data = np.genfromtxt(fname, skip_header=2)
                        if data.ndim == 1:
                            data = data.reshape(1, -1)
                        obj_col_idx = ndim + 2              # evaluation number | interface id | {ndim} variables | obj function value
                        if data.shape[1] > obj_col_idx: 
                            col = data[:, obj_col_idx]
                            best = np.nanmin(col)
                    except Exception as e:
                        print(f"[boxplot] error reading {fname}: {e}")
                out_f.write(f"{i}\t{('nan' if np.isnan(best) else f'{best:.5e}')}\n")
            print(f"[create_absolute_minima_per_instance] wrote {output_file}")

"""
Create copies of the driver and template.in / sbo.in file into each method subfolder for the problem, including all necessary folders
"""
def setup(func_folder, func, ndim, method_types):
    # Define source files, from appropriate subfolder
    driver_file = os.path.join("drivers_and_templates", f"{func}_driver.py")
    input_file = os.path.join("drivers_and_templates", f"template_{func}{ndim}.in")
    sbo_input_file = os.path.join("drivers_and_templates", f"sbo_{func}{ndim}.in")

    # Check that the required files exist in the current working directory
    if not os.path.isfile(driver_file):
        raise FileNotFoundError(f"Missing required file: {driver_file}")
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Missing required file: {input_file}")

    for method in method_types:
        method_folder = os.path.join(func_folder, method)
        os.makedirs(method_folder, exist_ok=True)

        # Copy files into the {method} folder
        shutil.copy(driver_file, method_folder)
        
        if method == "sbo":
            new_input_filename = f"{func}{ndim}_sbo.in"
            dest_input_path = os.path.join(method_folder, new_input_filename)
            shutil.copy(sbo_input_file, dest_input_path)
        else:
            new_input_filename = f"{func}{ndim}_{method}.in"
            dest_input_path = os.path.join(method_folder, new_input_filename)
            shutil.copy(input_file, dest_input_path)


"""
Runs the optimization and post-processing loop for all Dakota solvers for one problem at a time  
"""
def main(func, ndim, setup_flag, instances=10):
    func_folder = f"{func}{ndim}"
    method_types = ["ea", "GENIE", "sbo", "soga"]

    if setup_flag == True:
        setup(func_folder, func, ndim, method_types)
        return

    run_optimization(func, ndim, method_types, instances)
    append_running_min(func, ndim, method_types, instances)
    
    create_absolute_minima_per_instance(func, ndim, method_types, instances)
    create_running_minima_files(func, ndim, method_types, instances)

if __name__ == "__main__":
    # the lines below run all the SOO problems, with the setup flagged set to false
    
    main("rosenbrock", 10, False)
    main("rosenbrock", 50, False)
    main("michalewicz", 10, False)
    main("michalewicz", 30, False)
    main("michalewicz", 60, False)
    main("schubert", 10, False)
    main("schubert", 30, False)
    main("schubert", 60, False)

