# python_solvers.py, contains the definitions for the BaseSolver class and the SOO solver classes

from typing import Callable, Sequence, Tuple, Dict
import numpy as np

from scipy.optimize import differential_evolution
from scipy.optimize import dual_annealing
from scipy.optimize import direct

import nevergrad as ng
from OMADS import MADS

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem as PymooProblem
from pymoo.optimize import minimize

# Only used in for the initial test on Rosenbrock 10d
# from smt.applications.ego import EGO
# from smt.sampling_methods import LHS
# from smt.surrogate_models import KRG
# from smt.design_space import DesignSpace
# from ax.service.managed_loop import optimize


# ===================================
# 1) PROBLEM
# ===================================
class Problem:
    def __init__(self, dim: int, instance: int, bounds: Sequence[Tuple[float, float]], objective: Callable[[np.ndarray], float]):
        self.dim = dim
        self.instance = instance
        self.bounds = bounds
        self.history = []
        self.best_so_far = []

        orig_obj = objective

        def wrapped(x):
            x = np.asarray(x).flatten()
            val = orig_obj(x)
            self.history.append(val)
            if not self.best_so_far:
                self.best_so_far.append(val)
            else:
                self.best_so_far.append(min(self.best_so_far[-1], val))
            return val

        self.objective = wrapped

# ===================================
# 2) BASE SOLVER, defines the budget and problem
# ===================================
class BaseSolver:
    name: str = "Base"
    def __init__(self, budget: int = 1000):
        self.budget = budget
    def solve(self, prob: Problem) -> Dict:
        raise NotImplementedError

# ===================================
# 3) SOLVER WRAPPERS 
# ===================================

class SciPyDE(BaseSolver):
    name = "SciPy-DE"
    def solve(self, prob: Problem) -> Dict:
        popsize = 50 
        n_evals_per_iter = popsize * prob.dim
        maxiter = int((self.budget - n_evals_per_iter) / n_evals_per_iter)
        res = differential_evolution(prob.objective, prob.bounds, maxiter=maxiter, popsize=popsize, disp=False, polish=False, seed=prob.instance)
        return dict(x=res.x, f=res.fun, n_eval=res.nfev, history=prob.history.copy(), best=prob.best_so_far.copy())

class SciPyDA(BaseSolver):
    name = "SciPy-DA"
    def solve(self, prob: Problem) -> Dict:
        res = dual_annealing(prob.objective, bounds=prob.bounds, maxfun=self.budget, no_local_search=True, seed=prob.instance)
        return dict(x=res.x, f=res.fun, n_eval=res.nfev, history=prob.history.copy(), best=prob.best_so_far.copy())

class SciPyDIRECT(BaseSolver):
    name = "SciPy-DIRECT"
    def solve(self, prob: Problem) -> Dict:
        res = direct(prob.objective, prob.bounds, maxfun=self.budget)
        return dict(x=res.x, f=res.fun, n_eval=res.nfev, history=prob.history.copy(), best=prob.best_so_far.copy())

class NGIoh(BaseSolver):
    name = "Nevergrad"
    def solve(self, prob: Problem) -> Dict:
        np.random.seed(prob.instance)
        instr = ng.p.Array(shape=(prob.dim,)).set_bounds(*zip(*prob.bounds))

        # Obtain a robust scale so the transform behaves well across problems
        rng = np.random.default_rng(prob.instance)
        probes = np.array([
            rng.uniform([b[0] for b in prob.bounds], [b[1] for b in prob.bounds])
            for _ in range(min(16, 2 * prob.dim + 4))
        ])
        vals = []
        for p in probes:
            v = prob.objective(p)
            if np.isfinite(v):
                vals.append(abs(v))
        scale = max(np.median(vals), 1.0) if len(vals) else 1.0

        # Create a stable, order-preserving squashing for R -> R, arcsinh behaves ~linear near 0 and ~log at large |x|
        def squash(y: float) -> float:
            if not np.isfinite(y):
                # finite penalty to avoid Nevergrad's internal clipping
                y = 1e12
            return float(np.arcsinh(y / scale))

        def safe_obj(x):
            raw = prob.objective(x)
            return squash(raw)

        opt = ng.optimizers.NgIohTuned(parametrization=instr, budget=self.budget)
        rec = opt.minimize(safe_obj)

        # Return metrics in the original scale to keep your benchmarks comparable
        x_star = np.asarray(rec.value)
        f_raw = float(prob.objective(x_star))

        return dict(
            x=x_star,
            f=f_raw,
            n_eval=opt.num_ask,
            history=prob.history.copy(),
            best=prob.best_so_far.copy(),
        )

class PymooGA(BaseSolver):
    name = "pymoo-GA"
    def solve(self, prob: Problem) -> Dict:
        xl = np.array([b[0] for b in prob.bounds])
        xu = np.array([b[1] for b in prob.bounds])

        class _MyProb(PymooProblem):
            def __init__(self):
                super().__init__(n_var=prob.dim, n_obj=1, xl=xl, xu=xu, elementwise=True)
            def _evaluate(self, x, out):
                out["F"] = prob.objective(x)

        alg = GA(pop_size=100, eliminate_duplicates=True)
        res = minimize(_MyProb(), alg, termination=("n_gen", self.budget // (50 * 2)), verbose=False, seed=prob.instance)
        
        if res.F is None or res.X is None:
            return dict(error="No solution found", f=float("inf"), n_eval=0, history=[], best=[])

        return dict(x=res.X, f=res.F[0], n_eval=res.algorithm.evaluator.n_eval, history=prob.history.copy(), best=prob.best_so_far.copy())

class SMTEGO(BaseSolver):
    name = "SMT-EGO"
    def solve(self, prob: Problem) -> Dict:
        xlimits = np.array(prob.bounds)
        design_space = DesignSpace(xlimits)
        init_points = 150 # Mirroring Dakota's EGO default setting for initial number of training points 
        n_parallel = 1
        n_iter = 50 # Increasing this to exhaust the problem's budget would take an incredible amount of time
        
        # Building the sample initial DoE
        sampling = LHS(xlimits=xlimits, criterion="ese", random_state=prob.instance)
        x_doe = sampling(init_points)
        y_doe = np.array([prob.objective(xi) for xi in x_doe]).reshape(-1, 1)
        surrogate = KRG(design_space=design_space)  # KRG surrogate model

        # Building the EGO model
        ego = EGO(
            n_iter=n_iter,
            xdoe=x_doe,
            ydoe=y_doe,
            surrogate=surrogate,
            criterion="EI",
            n_parallel=n_parallel,
            random_state=prob.instance,
        )

        x_opt, y_opt, _, x_data, y_data = ego.optimize(fun=prob.objective)

        return dict(
            x=x_opt,
            f=y_opt.item(),
            n_eval=len(x_data),
            history=prob.history.copy(),
            best=prob.best_so_far.copy(),
        )

class AxBO(BaseSolver):
    name = "Ax-BO"
    def solve(self, prob: Problem) -> Dict:            
        parameters = [{"name": f"x{i}", "type": "range", "bounds": list(prob.bounds[i])} for i in range(prob.dim)]

        def _eval(params):
            x = np.array([params[f"x{i}"] for i in range(prob.dim)], dtype=float)
            return {"obj": prob.objective(x)}

        kwargs = {
            "parameters": parameters,
            "evaluation_function": _eval,
            "minimize": True,
            "objective_name": "obj",
            "total_trials": 150,
            "random_seed": prob.instance
        }

        try:
            bp, bv, exp, _ = optimize(**kwargs) 
            return dict(
                x=np.array([bp[f"x{i}"] for i in range(prob.dim)]),
                f=bv[0]["obj"],
                n_eval=len(exp.trials),
                history=prob.history.copy(),
                best=prob.best_so_far.copy())
        except Exception as e:
            return dict(
                error=f"Optimization failed: {str(e)}",
                f=float('inf'),
                n_eval=0,
                history=[],
                best=[]
            )

class OMADSSolver(BaseSolver):
    name = "OMADS"
    def solve(self, prob: Problem) -> Dict:
        lb = np.array([b[0] for b in prob.bounds], dtype=float)
        ub = np.array([b[1] for b in prob.bounds], dtype=float)
        span = ub - lb

        def to_x(z):
            z = np.asarray(z, float).flatten()
            return lb + z * span

        # Robust scale
        rng = np.random.default_rng(prob.instance)
        n_probe = min(16, 2 * prob.dim + 4)
        probes = rng.random((n_probe, prob.dim))
        mags = []
        for z in probes:
            v = prob.objective(to_x(z))
            if np.isfinite(v):
                mags.append(abs(v))
        scale = max(np.median(mags), 1.0) if mags else 1.0

        def squash(y):
            if not np.isfinite(y):
                y = 1e12
            return float(np.arcsinh(y / scale))

        def omads_obj(z):
            raw = prob.objective(to_x(z))
            return squash(raw)
        # Jittered baseline so instances differ
        z0 = np.clip(0.5 + 0.05 * rng.standard_normal(prob.dim), 0.0, 1.0)

        data = {
            "evaluator": {"blackbox": omads_obj},
            "param": {
                "baseline": z0.tolist(),
                "lb": [0.0] * prob.dim,
                "ub": [1.0] * prob.dim,
                "var_names": [f"x{i+1}" for i in range(prob.dim)],
                "post_dir": "./omads_post",
                "Failure_stop": False,
            },
            "options": {
                "seed": prob.instance,
                "budget": self.budget,
                "tol": 1e-6,
                "display": False,
                "check_cache": True,
                "store_cache": True,
                "rich_direction": True,
                "opportunistic": True,
                "save_results": False,
                "isVerbose": False,
                "psize_init": 0.20,
                "precision": "low",
            },
            "sampling": {
                "method": "LHS",
                "ns": min(2 * prob.dim + 2, 40),
            },
            "search": {
                "type": "QUAD",
                "ns": min(prob.dim + 1, 20),
            },
        }

        output, _ = MADS.main(data)

        # Map values back to original & report RAW f
        z_star = np.array(output["xmin"], float).reshape(-1)
        x_star = to_x(z_star)
        f_raw = float(prob.objective(x_star))

        return dict(
            x=x_star,
            f=f_raw,
            n_eval=output.get("nbb_evals", None),
            history=prob.history.copy(),
            best=prob.best_so_far.copy(),
        )