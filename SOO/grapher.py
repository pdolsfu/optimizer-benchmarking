# graph.py, this script creates the boxplot, convergence charts, runtime summaries, and solver min, max, and avg minima summaries for all the SOO problems

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from itertools import cycle

from oasis_processor import process_oasis

# Constructs the boxplots from {solver}_best_minima_per_instance.tsv files
def construct_boxplot(problem_folders, solver_names):
    for problem in problem_folders:
        solver_best = {}
        for solver in solver_names:
            file_path = os.path.join(problem, solver, f"{solver}_best_minima_per_instance.tsv")
            if not os.path.exists(file_path):
                print(f"[boxplot] missing file {file_path}, skipping {solver} for {problem}")
                continue
            df = pd.read_csv(file_path, sep="\t")
            if solver in df.columns:
                bests = pd.to_numeric(df[solver], errors="coerce")
                solver_best[solver] = bests.values
            else:
                print(f"[boxplot] unexpected format in {file_path}, no column '{solver}'")
        if not solver_best:
            continue
        bp_df = pd.DataFrame(solver_best)

        plt.figure(figsize=(10, 6))
        data_to_plot = [bp_df[s].dropna() for s in bp_df.columns]

        # ✅ uppercase only specific solvers
        labels = [col.upper() if col.lower() in ["sbo", "ea", "soga"] else col for col in bp_df.columns]

        box = plt.boxplot(data_to_plot, labels=labels, patch_artist=True, showfliers=False)

        # Define colors
        color_python = "#7fbf7f"   # e.g., greenish
        color_oasis = "#ff7f0e" # e.g., orange
        color_dakota = "#1f77b4"    # e.g., blue

        for idx, patch in enumerate(box["boxes"]):
            if idx <= 5:
                patch.set_facecolor(color_python)
            elif idx == 6:
                patch.set_facecolor(color_oasis)
            else:
                patch.set_facecolor(color_dakota)
            patch.set_edgecolor("black")

        # Optionally color medians/whiskers to match or differentiate
        for median in box["medians"]:
            median.set_color("black")
        for whisker in box["whiskers"]:
            whisker.set_color("black")
        for cap in box["caps"]:
            cap.set_color("black")

        plt.yscale("symlog", linthresh=1e-3)
        plt.ylabel("Best minima (symlog scale)")
        plt.title(f"Best minima per instance for {problem}")
        plt.grid(True, which="both", axis="y", alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.001)

        ax = plt.gca()
        # optional: wrap long labels into two lines at ~10 chars
        labels = [l if len(l) <= 10 else l[:10] + "\n" + l[10:] for l in labels]
        ax.set_xticklabels(labels)

        # stagger tick labels vertically
        for i, lbl in enumerate(ax.get_xticklabels()):
            lbl.set_y(-0.05 if i % 2 == 0 else -0.1)
            lbl.set_verticalalignment("top")

        plt.subplots_adjust(bottom=0.3)

        out_path = os.path.join(problem, f"{problem}_boxplot.png")
        plt.savefig(out_path)
        plt.close()
        print(f"[boxplot] saved {out_path}")

# Constructs the convergence charts from {solver}_average_running_minima.tsv files
def construct_convergence(problem_folders, solver_names):
    for problem in problem_folders:
        plt.figure(figsize=(10, 6))
        any_plotted = False
        color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

        for idx, solver in enumerate(solver_names):
            avg_file = os.path.join(problem, solver, f"{solver}_avg_running_minima.tsv")
            if not os.path.exists(avg_file):
                print(f"[convergence] missing {avg_file}, skipping")
                next(color_cycle)
                continue

            df = pd.read_csv(avg_file, sep="\t")
            if "FE" not in df.columns or "Average" not in df.columns:
                print(f"[convergence] malformed {avg_file}, skipping")
                next(color_cycle)
                continue

            # Parse
            fe  = pd.to_numeric(df["FE"], errors="coerce").to_numpy()
            avg = pd.to_numeric(df["Average"], errors="coerce").to_numpy()

            # Filter to finite FE and non-NaN avg
            mask = np.isfinite(fe) & ~np.isnan(avg)
            fe, avg = fe[mask], avg[mask]

            if fe.size == 0 or avg.size == 0:
                next(color_cycle)
                continue

            # Ensure FE is increasing left→right
            order = np.argsort(fe)
            fe, avg = fe[order], avg[order]

            # Enforce monotone non-increasing running minima on the plotted values
            avg = np.minimum.accumulate(avg)

            color = next(color_cycle)

            # line style by group
            if idx <= 5:
                ls, lw = "-", 1.5
            elif idx == 6:
                ls, lw = "-.", 2.5
            else:
                ls, lw = "--", 1.5

            label = solver.upper() if solver in ["sbo", "ea", "soga"] else solver

            # Debug
            print(
                f"[DEBUG] Plotting {label}: len(fe)={len(fe)}, len(avg)={len(avg)}, "
                f"fe range=({fe.min()}, {fe.max()}), "
                f"avg range=({np.nanmin(avg)}, {np.nanmax(avg)})"
            )

            plt.plot(fe, avg, label=label, color=color, linestyle=ls, linewidth=lw)
            any_plotted = True

        if not any_plotted:
            plt.close()
            continue

        plt.xlabel("Function Evaluations (FE)")
        plt.ylabel("Average Running Minima")
        plt.title(f"Convergence for {problem}")

        # Wide dynamic range (both signs)
        plt.yscale("symlog", linthresh=1e5)

        # Scientific notation formatting
        ax = plt.gca()
        sci_formatter = ScalarFormatter(useMathText=True)
        sci_formatter.set_scientific(True)
        sci_formatter.set_powerlimits((-2, 2))
        ax.xaxis.set_major_formatter(sci_formatter)
        ax.yaxis.set_major_formatter(sci_formatter)

        # Force x-axis to exactly [0, 1000]
        ax.set_xlim(0, 1000)

        plt.grid(True, which="both", ls="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(problem, f"{problem}_convergence.png")
        plt.savefig(out_path)
        plt.close()
        print(f"[convergence] saved {out_path}")

def construct_time_table(problem_folders, solver_names):
    for problem in problem_folders:
        combined = {}
        max_runs = 0
        for solver in solver_names:
            time_file = os.path.join(problem, solver, f"{solver}_time.tsv")
            if not os.path.exists(time_file):
                print(f"[time_table] missing {time_file}, skipping {solver}")
                continue
            df = pd.read_csv(time_file, sep="\t")
            if "Run" not in df.columns or "Time" not in df.columns:
                print(f"[time_table] unexpected format in {time_file}")
                continue
            df = df.sort_values("Run")
            combined[solver] = df["Time"].to_list()
            max_runs = max(max_runs, len(df))
        
        if not combined:
            continue

        # Pad all lists to the same length
        table = {}
        for solver, times in combined.items():
            padded = times + [np.nan] * (max_runs - len(times))
            table[solver] = padded
        
        table_df = pd.DataFrame(table)
        table_df.insert(0, "Run", list(range(1, max_runs + 1)))

        # Compute total time row
        total_row = {"Run": "total (hr)"}
        for solver in solver_names:
            if solver in table_df.columns:
                total_row[solver] = np.nansum(table_df[solver])/3600
            else:
                total_row[solver] = np.nan

        # Append total row
        table_df = pd.concat([table_df, pd.DataFrame([total_row])], ignore_index=True)

        # Save the file
        out_path = os.path.join(problem, f"{problem}_time_table.tsv")
        table_df.to_csv(out_path, sep="\t", index=False, float_format="%.5f", na_rep="nan")
        print(f"[time_table] saved {out_path}")

def construct_solver_summary_stats(problem_folders, solver_names):
    for problem in problem_folders:
        summary_rows = []
        for solver in solver_names:
            file_path = os.path.join(problem, solver, f"{solver}_best_minima_per_instance.tsv")
            if not os.path.exists(file_path):
                print(f"[summary_stats] missing {file_path}, skipping {solver} for {problem}")
                continue
            df = pd.read_csv(file_path, sep="\t")
            if solver not in df.columns:
                print(f"[summary_stats] unexpected format in {file_path}, no column '{solver}'")
                continue
            vals = pd.to_numeric(df[solver], errors="coerce").dropna()
            if vals.empty:
                min_val = max_val = avg_val = np.nan
            else:
                min_val = vals.min()
                max_val = vals.max()
                avg_val = vals.mean()
            summary_rows.append({
                "Solver": solver,
                "Min": min_val,
                "Max": max_val,
                "Average": avg_val,
            })

        if not summary_rows:
            continue
        summary_df = pd.DataFrame(summary_rows)
        # order columns as requested
        summary_df = summary_df[["Solver", "Min", "Max", "Average"]]

        out_path = os.path.join(problem, f"{problem}_solver_stats.tsv")
        # write with scientific formatting
        with open(out_path, "w") as f:
            f.write("Solver\tMin\tMax\tAverage\n")
            for _, row in summary_df.iterrows():
                def fmt(x):
                    return "nan" if pd.isna(x) else f"{x:.2e}"
                f.write(f"{row['Solver']}\t{fmt(row['Min'])}\t{fmt(row['Max'])}\t{fmt(row['Average'])}\n")
        print(f"[summary_stats] saved {out_path}")

if __name__ == "__main__":   
    problem_folders = ["rosenbrock10"]
    # problem_folders = ["michalewicz10", "schubert10", "rosenbrock10", "michalewicz30", "schubert30", "michalewicz60", "schubert60", "rosenbrock50"] 
    
    # Be sure to change the indexes that indicate the index of the last Python solver if adding other solvers for graphical comparison, with the "idx" variable 
    solver_names = ["Nevergrad","OMADS", "pymoo-GA", "SciPy-DIRECT", "SciPy-DE", "SciPy-DA",# "Ax-BO", "SMT-EGO",
                    "OASIS",
                    "ea", "GENIE", "sbo", "soga"#,  "ego", "COLINY", "genie_opt_darts", "NCSU"
                    ]
    
    process_oasis(problem_folders, stage=1) # only needed once for each function
    
    construct_boxplot(problem_folders, solver_names)
    construct_convergence(problem_folders, solver_names)

    construct_time_table(problem_folders, solver_names)
    construct_solver_summary_stats(problem_folders, solver_names)