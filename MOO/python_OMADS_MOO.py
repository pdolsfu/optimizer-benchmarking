# python_OMADS_BOO.py, runs OMADS on each MOO problem and post-processes the solver data

import os
import time
import numpy as np
from OMADS import MADS
from typing import List, Tuple

# your ZDT1 evaluator
def MO_ZDT1(x: List[float]) -> Tuple[List[float], List[float]]:
    f1 = x[0]
    g  = 1 + 9 * sum(x[1:]) / (len(x)-1)
    h  = 1 - np.sqrt(f1/g)
    f2 = g*h
    return [[f1, f2], [0]]  # no constraints

def MO_ZDT2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    f1 = x[0]

    n = len(x)
    g = 1 + 9 / (n - 1) * np.sum(x[1:])
    h = 1 - (f1 / g) ** 2
    f2 = g * h
    return [[f1, f2], [0]]  # no constraints

def MO_ZDT3(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    f1 = x[0]

    n = len(x)
    g = 1 + 9 / (n - 1) * np.sum(x[1:])
    h = 1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1)
    f2 = g * h

    return [[f1, f2], [0]]  # no constraints

def MO_ZDT6(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    f1 = 1 - np.exp(-4 * x[0]) * (np.sin(6 * np.pi * x[0]) ** 6)

    n = len(x)
    g = 1 + 9 * ((np.sum(x[1:]) / (n - 1)) ** 0.25)
    h = 1 - (f1 / g) ** 2
    f2 = g * h

    return [[f1, f2], [0]]  # no constraints

def MO_OSY(x):
    x = np.asarray(x, dtype=float)

    # objectives
    f1 = -(
        25 * (x[0] - 2) ** 2
        + (x[1] - 2) ** 2
        + (x[2] - 1) ** 2
        + (x[3] - 4) ** 2
        + (x[4] - 1) ** 2
    )
    f2 = float(np.sum(x**2))

    # constraints in OMADS form (g_i(x) <= 0 is feasible)
    g = [
        -(x[0] + x[1] - 2),                 # C1: x1 + x2 - 2 >= 0
        -(6 - x[0] - x[1]),                 # C2: 6 - x1 - x2 >= 0
        -(2 - x[1] + x[0]),                 # C3: 2 - x2 + x1 >= 0
        -(2 - x[0] + 3 * x[1]),             # C4: 2 - x1 + 3x2 >= 0
        -((4 - (x[2] - 3) ** 2 - x[3])),    # C5: 4 - (x3-3)^2 - x4 >= 0
        -(((x[4] - 3) ** 2 + x[5] - 4)),    # C6: (x5-3)^2 + x6 - 4 >= 0
    ]

    return [[float(f1), float(f2)], g]

# ─── Gear Train evaluator ─────────────────────────────────────────────────────
def MO_GEARTRAIN(x):
    x = np.asarray(x, dtype=float)
    x = np.rint(x) # ensures that OMADS evaluates Geartrain with discrete variables

    # objectives
    f1 = (1.0 / 6.931 - (x[2] * x[1]) / (x[0] * x[3])) ** 2
    f2 = float(np.max(x))  # minimize the maximum number of teeth

    # no inequality constraints beyond box bounds
    return [[float(f1), float(f2)], [0]]

# common definition for all functions
def common_dict():
    outDict: dict = {
        "evaluator": {"blackbox": None},
        "param": {
            "baseline": None,
            "lb": None,
            "ub": None,
            "var_names": ["x", "y"],
            "fun_names": ["f1", "f2"],
            # "constraints_type": ["PB", "PB"],
            "nobj": 2,
            "isPareto": True,
            "scaling": None,
            "LAMBDA": [1E5, 1E5],
            "RHO": 1.0,
            "h_max": np.inf,
            "meshType": "GMESH",
            "post_dir": None
        },
        "options": {
            "seed": 0,
            "budget": 2000,
            "tol": 1e-12,
            "psize_init": 1,
            "display": False,
            "opportunistic": False,
            "check_cache": True,
            "store_cache": True,
            "collect_y": False,
            "rich_direction": True,
            "precision": "high",
            "save_results": True,
            "save_coordinates": False,
            "save_all_best": False,
            "parallel_mode": False
        },
        "search": {
            "type": "sampling",
            "s_method": "ACTIVE",
            "ns": 10,
            "visualize": False
        },
    }
    return outDict

def run_ZDT1(base, instances):
    for i in range(1, instances+1):
        d = 30
        data = common_dict()
        data["evaluator"]["blackbox"] = MO_ZDT1
        data["param"]["name"] = "MO_ZDT1"
        np.random.seed(seed=i)
        data["param"]["baseline"] = np.random.rand(d)
        data["param"]["var_names"] = [f'x{i}' for i in range(d)]
        data["param"]["lb"] = [0]*d
        data["param"]["ub"] = [1]*d
        data["param"]["meshType"] = "GMESH"
        data["param"]["constraints_type"] = []
        data["param"]["scaling"] = [1]*d
        data["param"]["post_dir"] = os.path.join(base, f"run{i}")
        data["options"]["budget"] = 250
        data["options"]["seed"] = i

        os.makedirs(data["param"]["post_dir"], exist_ok=True)
        print(f"✔ Running ZDT1 seed {i} → {data['param']['post_dir']}")
        MADS.main(data)

def run_ZDT2(base, instances):
    for i in range(1, instances+1):
        d = 30
        data = common_dict()
        data["evaluator"]["blackbox"] = MO_ZDT2
        data["param"]["name"] = "MO_ZDT2"
        np.random.seed(seed=i)
        data["param"]["baseline"] = np.random.rand(d)
        data["param"]["var_names"] = [f'x{i}' for i in range(d)]
        data["param"]["lb"] = [0]*d
        data["param"]["ub"] = [1]*d
        data["param"]["meshType"] = "GMESH"
        data["param"]["constraints_type"] = []
        data["param"]["scaling"] = [1]*d
        data["param"]["post_dir"] = os.path.join(base, f"run{i}")
        data["options"]["budget"] = 250
        data["options"]["seed"] = i

        os.makedirs(data["param"]["post_dir"], exist_ok=True)
        print(f"✔ Running ZDT2 seed {i} → {data['param']['post_dir']}")
        MADS.main(data)

def run_ZDT3(base, instances):
    for i in range(1, instances+1):
        d = 30
        data = common_dict()
        data["evaluator"]["blackbox"] = MO_ZDT3
        data["param"]["name"] = "MO_ZDT3"
        np.random.seed(seed=i)
        data["param"]["baseline"] = np.random.rand(d)
        data["param"]["var_names"] = [f'x{i}' for i in range(d)]
        data["param"]["lb"] = [0]*d
        data["param"]["ub"] = [1]*d
        data["param"]["meshType"] = "GMESH"
        data["param"]["constraints_type"] = []
        data["param"]["scaling"] = [1]*d
        data["param"]["post_dir"] = os.path.join(base, f"run{i}")
        data["options"]["budget"] = 250
        data["options"]["seed"] = i

        os.makedirs(data["param"]["post_dir"], exist_ok=True)
        print(f"✔ Running ZDT3 seed {i} → {data['param']['post_dir']}")
        MADS.main(data)

def run_ZDT6(base, instances):
    for i in range(1, instances+1):
        d = 30
        data = common_dict()
        data["evaluator"]["blackbox"] = MO_ZDT6
        data["param"]["name"] = "MO_ZDT6"
        np.random.seed(seed=i)
        data["param"]["baseline"] = np.random.rand(d)
        data["param"]["var_names"] = [f'x{i}' for i in range(d)]
        data["param"]["lb"] = [0]*d
        data["param"]["ub"] = [1]*d
        data["param"]["meshType"] = "OMESH" # same as OMADS' library specification
        data["param"]["constraints_type"] = []
        data["param"]["scaling"] = [1]*d
        data["param"]["post_dir"] = os.path.join(base, f"run{i}")
        data["options"]["budget"] = 250
        data["options"]["seed"] = i

        os.makedirs(data["param"]["post_dir"], exist_ok=True)
        print(f"✔ Running ZDT6 seed {i} → {data['param']['post_dir']}")
        MADS.main(data)

def run_OSY(base, instances):
    # OSY variable bounds


    for i in range(1, instances + 1):
        data = common_dict()
        data["evaluator"]["blackbox"] = MO_OSY
        data["param"]["name"] = "MO_OSY"
        np.random.seed(seed=i)
        data["param"]["baseline"] = [3, 2, 2, 0, 5, 10]
        data["param"]["var_names"] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
        data["param"]["lb"] = [0, 0, 1, 0, 1, 0]
        data["param"]["ub"] = [10, 10, 5, 6, 5, 10]
        data["param"]["meshType"] = "GMESH"
        data["param"]["constraints_type"] = ["PB"]*6  # progressive barrier
        data["param"]["scaling"] = [10, 10, 4, 6, 4, 10]
        data["param"]["post_dir"] = os.path.join(base, f"run{i}")
        data["options"]["budget"] = 10000
        data["options"]["seed"] = i

        os.makedirs(data["param"]["post_dir"], exist_ok=True)
        print(f"✔ Running OSY seed {i} → {data['param']['post_dir']}")
        MADS.main(data)

def run_GEARTRAIN(base, instances):
    lb = np.array([12, 12, 12, 12], dtype=float)
    ub = np.array([60, 60, 60, 60], dtype=float)
    d = len(lb)

    for i in range(1, instances + 1):
        data = common_dict()
        data["evaluator"]["blackbox"] = MO_GEARTRAIN
        data["param"]["name"] = "MO_GEARTRAIN"
        np.random.seed(seed=i)
        data["param"]["baseline"] = (lb + (ub - lb) * np.random.rand(d)).tolist()
        data["param"]["var_names"] = [f"x{j+1}" for j in range(d)]
        data["param"]["lb"] = lb.tolist()
        data["param"]["ub"] = ub.tolist()
        data["param"]["meshType"] = "GMESH"
        data["param"]["constraints_type"] = []
        data["param"]["scaling"] = [1] * d
        data["param"]["post_dir"] = os.path.join(base, f"run{i}")
        data["options"]["budget"] = 250
        data["options"]["seed"] = i

        os.makedirs(data["param"]["post_dir"], exist_ok=True)
        print(f"✔ Running GEARTRAIN seed {i} → {data['param']['post_dir']}")
        MADS.main(data)
def extract_f1f2(infile, outfile):
    with open(infile) as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if not lines:
        raise RuntimeError(f"{infile} is empty")

    # Parse header
    header = [col.strip() for col in lines[0].split(",") if col.strip()]
    try:
        f1_idx = header.index("f1")
        f2_idx = header.index("f2")
    except ValueError:
        raise RuntimeError(f"Header does not contain f1/f2: {header}")

    pts = []
    for ln in lines[1:]:
        cols = [tok.strip() for tok in ln.split(",") if tok.strip()]
        if len(cols) <= max(f1_idx, f2_idx):
            continue
        try:
            f1 = float(cols[f1_idx])
            f2 = float(cols[f2_idx])
        except ValueError:
            continue
        pts.append((f1, f2))

    if not pts:
        raise RuntimeError(f"No data extracted from {infile}")

    with open(outfile, "w") as f:
        f.write("f1\tf2\n")
        for a, b in pts:
            f.write(f"{a:.15e}\t{b:.15e}\n")

    print(f"  → wrote {len(pts)} points to {outfile}")



def is_dominated(p: np.ndarray, others: np.ndarray) -> bool:
    return np.any(np.all(others <= p, axis=1) & np.any(others < p, axis=1))


def combine_and_filter(base_dir: str, n_seeds: int, out_path: str):
    all_pts = []
    for i in range(1, n_seeds+1):
        fn = os.path.join(base_dir, f"f1f2_seed_{i}.dat")
        data = np.loadtxt(fn, delimiter="\t", skiprows=1)
        all_pts.append(data)
    F = np.vstack(all_pts)

    keep_idx = []
    for idx in range(F.shape[0]):
        if not is_dominated(F[idx], np.delete(F, idx, axis=0)):
            keep_idx.append(idx)
    pareto = F[keep_idx]

    order = np.lexsort((pareto[:,0], pareto[:,1]))
    pareto = pareto[order]

    with open(out_path, "w") as f:
        f.write("f1\tf2\n")
        for a, b in pareto:
            f.write(f"{a:.15e}\t{b:.15e}\n")
    print(f"Wrote combined Pareto ({len(pareto)} pts) → {out_path}")

PROBLEM_RUNNERS = {
    "ZDT1": run_ZDT1,
    "ZDT2": run_ZDT2,
    "ZDT3": run_ZDT3,
    "ZDT6": run_ZDT6,
    "OSY": run_OSY,
    "GEARTRAIN": run_GEARTRAIN
}


def main(func_name: str, instances: int):
    try:
        fn = PROBLEM_RUNNERS[func_name]
    except KeyError:
        raise ValueError(
            f"Unknown problem {func_name!r}. Valid choices are: {list(PROBLEM_RUNNERS)}"
        )

    # 1) Build base folder
    base = os.path.join(func_name, "OMADS")
    os.makedirs(base, exist_ok=True)

    # 2) Monkey‐patch MADS.main to capture per‐seed times
    original_mads_main = MADS.main
    omads_times = []
    def timed_mads_main(data):
        start = time.time()
        result = original_mads_main(data)
        end = time.time()
        omads_times.append(end - start)
        return result
    MADS.main = timed_mads_main

    # 3) Run all seeds
    fn(base, instances)

    # 4) Restore original
    MADS.main = original_mads_main

    # 5) Post‐process each seed: extract f1/f2
    for i in range(1, instances+1):
        data_in = os.path.join(base, f"run{i}", f"MO_{func_name}_ND", f"MO_{func_name}_Pareto.out")
        data_out = os.path.join(base, f"f1f2_seed_{i}.dat")
        print(f"Processing seed {i}: {data_in}")
        extract_f1f2(data_in, data_out)

    # 6) Combine & filter into one Pareto front
    combined = os.path.join(base, f"OMADS_pareto.tsv")
    combine_and_filter(base, instances, combined)

    # 7) Write out the timing table
    time_file = os.path.join(func_name, "OMADS", "OMADS_time.tsv")
    # make sure the directory exists
    os.makedirs(os.path.dirname(time_file), exist_ok=True)
    with open(time_file, "w") as tf:
        # header
        tf.write("Run\tOMADS\n")
        # each row: run index and elapsed time
        for run_idx, elapsed in enumerate(omads_times, start=1):
            tf.write(f"{run_idx}\t{elapsed:.5f}\n")
    print(f"Wrote timing table → {time_file}")

# ─── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main("ZDT1", 10) # as an example, this function is called through python_batch_optimizer.py
    