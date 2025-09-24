# python_solvers.py, contains the definitions for the BaseSolver class and the SOO solver classes

from typing import Callable, Sequence, Tuple, Dict
import numpy as np

# pip install scipy
from scipy.optimize import differential_evolution
from scipy.optimize import dual_annealing
from scipy.optimize import direct
from scipy.optimize import NonlinearConstraint # for handling constraints in scipy

# pip install nevergrad
import nevergrad as ng

# pip install pymoo
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem as PymooProblem
from pymoo.optimize import minimize

# pip install torch botorch ax-platform
from ax.service.managed_loop import optimize

# pip install OMADS
from OMADS import MADS  # MADS is the module


# ===================================
# 1) PROBLEM
# ===================================
# each time solver creates a new instance of the Problem class, which stores the history, running minima, and instance
class Problem:
    def __init__(self, dim: int, instance: int, bounds: Sequence[Tuple[float, float]], objective: Callable[[np.ndarray], float], constraints = None):
        self.dim = dim
        self.instance = instance
        self.bounds = bounds
        self.constraints = constraints or []   # always a list
        self.history = []
        self.best_so_far = []

        # Wrap the objective to log evaluations
        orig_obj = objective

        def wrapped(x):
            x = np.asarray(x).flatten()
            val = orig_obj(x)
            
            #if not np.isfinite(val):
                #print(f"[WARN] Received non-finite objective: x={x}, val={val}")
            #else:
                #print(f"[OK] method called objective: x={x}, val={val}")
            if not self.constraints or all(c(x) >= 0.00001 for c in self.constraints):  # only log if feasible
                self.history.append(val)
                if not self.best_so_far:
                    self.best_so_far.append(val)
                else:
                    self.best_so_far.append(min(self.best_so_far[-1], val))
            else:
                val = np.inf  # or optionally return a penalized value
            return val

        self.objective = wrapped

# ===================================
# 2) BASE SOLVER
# ===================================
class BaseSolver:
    name: str = "Base"
    # instantiates the budget to 1000
    def __init__(self, budget: int = 1000):
        self.budget = budget
    def solve(self, prob: Problem) -> Dict:
        raise NotImplementedError

# ===================================
# 3) SOLVER WRAPPERS 
# ===================================

# SciPy Differential Evolution
class SciPyDE(BaseSolver):
    name = "SciPy-DE"
    def solve(self, prob: Problem) -> Dict:
        scipy_constraints = []
        if prob.constraints:
            scipy_constraints = [NonlinearConstraint(c, lb=0.0, ub=np.inf) for c in prob.constraints]

        popsize = 50 
        n_evals_per_iter = popsize * prob.dim
        maxiter = int((self.budget - n_evals_per_iter) / n_evals_per_iter)
        res = differential_evolution(prob.objective, prob.bounds, constraints=scipy_constraints, maxiter=maxiter, popsize=popsize, disp=False, polish=False, seed=prob.instance)
        return dict(x=res.x, f=res.fun, n_eval=res.nfev, history=prob.history.copy(), best=prob.best_so_far.copy())

# SciPy Dual Annealing
class SciPyDA(BaseSolver):
    name = "SciPy-DA"
    def solve(self, prob: Problem) -> Dict:
            res = dual_annealing(prob.objective, bounds=prob.bounds, maxfun=self.budget, no_local_search=True, seed=prob.instance)
            return dict(x=res.x, f=res.fun, n_eval=res.nfev, history=prob.history.copy(), best=prob.best_so_far.copy())

# SciPy DIRECT
class SciPyDIRECT(BaseSolver):
    name = "SciPy-DIRECT"
    def solve(self, prob: Problem) -> Dict:
        res = direct(prob.objective, prob.bounds, maxfun=self.budget) # no seed as it is a deterministic approach
        return dict(x=res.x, f=res.fun, n_eval=res.nfev, history=prob.history.copy(), best=prob.best_so_far.copy())

# Nevergrad NgIohTuned
class NGIoh(BaseSolver):
    name = "Nevergrad"
      
    def solve(self, prob: Problem) -> Dict:
        np.random.seed(prob.instance)
        instr = ng.p.Array(shape=(prob.dim,)).set_bounds(*zip(*prob.bounds))
        opt = ng.optimizers.NgIohTuned(parametrization=instr, budget=self.budget)
        rec = opt.minimize(prob.objective)
        return dict(x=rec.value, f=rec.loss, n_eval=opt.num_ask, history=prob.history.copy(), best=prob.best_so_far.copy())

# pymoo GA 
class PymooGA(BaseSolver):
    name = "pymoo-GA"
    def solve(self, prob: Problem) -> Dict:
        xl = np.array([b[0] for b in prob.bounds])
        xu = np.array([b[1] for b in prob.bounds])

        class _MyProb(PymooProblem):
            def __init__(self):
                super().__init__(n_var=prob.dim, n_obj=1, n_constr=len(prob.constraints) if prob.constraints else 0, xl=xl, xu=xu, elementwise=True)
            def _evaluate(self, x, out):
                out["F"] = prob.objective(x)
                
                if prob.constraints:
                    out["G"] = [-c(x) for c in prob.constraints]
        alg = GA(pop_size=100, eliminate_duplicates=True)
        res = minimize(_MyProb(), alg, termination=("n_gen", self.budget // (50 * 2)), verbose=False, seed=prob.instance)
        
        if res.F is None or res.X is None:
            return dict(error="No feasible solution found", f=float("inf"), n_eval=0, history=[], best=[])

        return dict(x=res.X, f=res.F[0], n_eval=res.algorithm.evaluator.n_eval, history=prob.history.copy(), best=prob.best_so_far.copy())
    
# Ax / BoTorch BO
class AxBO(BaseSolver):
    name = "Ax-BO"
    def solve(self, prob: Problem) -> Dict:            
        parameters = [{"name": f"x{i}", "type": "range", "bounds": list(prob.bounds[i])} for i in range(prob.dim)]

        def _eval(params):
            x = np.array([params[f"x{i}"] for i in range(prob.dim)], dtype=float)
            result = {"obj": prob.objective(x)}
            if prob.constraints:
                for i, c in enumerate(prob.constraints):
                    result[f"c{i}"] = float(c(x))
            return result

        kwargs = {
            "parameters": parameters,
            "evaluation_function": _eval,
            "minimize": True,
            "objective_name": "obj",
            "total_trials": 150,
            "random_seed": prob.instance
        }

        if prob.constraints:
            kwargs["outcome_constraints"] = [(f"c{i} <= 0.0") for i in range(len(prob.constraints))]
        try:
            bp, bv, exp, _ = optimize(**kwargs) 
            return dict(
                x=np.array([bp[f"x{i}"] for i in range(prob.dim)]),
                f=bv[0]["obj"],
                n_eval=len(exp.trials),
                history=prob.history.copy(),
                best=prob.best_so_far.copy())
        
        # gracefully handling errors
        except Exception as e:
            if 'exp' in locals():  # Check if experiment exists
                try:
                    # Get all evaluated points
                    df = exp.fetch_data().df
                    
                    # Find points that violate the fewest constraints
                    best_infeasible_idx = df['obj'].idxmin()
                    
                    return dict(
                        x=np.array([df.loc[best_infeasible_idx][f"x{i}"] for i in range(prob.dim)]),
                        f=float(df.loc[best_infeasible_idx]['obj']),
                        n_eval=len(exp.trials),
                        history=prob.history.copy(),
                        best=prob.best_so_far.copy(),
                        warning="Using best infeasible solution"
                    )
                except:
                    pass
                    
            # Ultimate fallback if everything fails
            return dict(
                error=f"Optimization failed: {str(e)}",
                f=float('inf'),
                n_eval=0,
                history=[],
                best=[]
            )
# OMADS
class OMADSSolver(BaseSolver):
    name = "OMADS"  
    def solve(self, prob: Problem) -> Dict:
        # Construct OMADS-compatible input
        data = {
            "evaluator": {"blackbox": prob.objective},
            "param": {
                "baseline": [(b[0] + b[1]) / 2 for b in prob.bounds],
                "lb": [b[0] for b in prob.bounds],
                "ub": [b[1] for b in prob.bounds],
                "var_names": [f"x{i+1}" for i in range(prob.dim)],
                "post_dir": "./omads_post",
                "Failure_stop": True
            },
            # all using default settings
            "options": { 
                "seed": prob.instance,
                "budget": self.budget,
                "tol": 0,
                "display": False,
                "check_cache": True,
                "store_cache": True,
                "rich_direction": True,
                "opportunistic": False,
                "save_results": False,
                "isVerbose": False,
                "psize_init": 1.0,          
                "precision": "medium",
            },
            "sampling": {
                "method": "ORTHO",
                "ns": 	2 * prob.dim + 2
            },
            "search": {
                "type": "QUAD",
                "ns": prob.dim + 1
            }
        }

        output, _ = MADS.main(data)  # ✅ Call the main function

        return dict(
            x=np.array(output["xmin"]),
            f=output["fmin"],
            n_eval=output["nbb_evals"],
            history=prob.history.copy(),
            best=prob.best_so_far.copy()
        )