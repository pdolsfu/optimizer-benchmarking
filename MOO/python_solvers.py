# python_solvers.py, contains the definitions for the BaseSolver class and the MOO solver classes

from typing import Callable, Sequence, Tuple, Dict, Optional
import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem as PymooProblem
from pymoo.util.ref_dirs import get_reference_directions
    
# Problem class definition
class Problem:
    def __init__(
        self,
        dim: int,
        instance: int,
        bounds: Sequence[Tuple[float, float]],
        objective: Callable[[np.ndarray], [np.ndarray]],
        constraints: Optional[Sequence[Callable[[np.ndarray], float]]] = None,
    ):
        self.dim = dim
        self.instance = instance
        self.bounds = bounds
        # --- normalize constraints ---
        if constraints is None:
            self.constraints = []                       # always a list
        elif callable(constraints):
            # vector-valued constraints → wrap into a list of scalar callables
            lb = np.array([b[0] for b in bounds], dtype=float)
            ub = np.array([b[1] for b in bounds], dtype=float)
            x_mid = (lb + ub) / 2.0

            def _vec(x):
                return np.atleast_1d(np.array(constraints(np.asarray(x, dtype=float)), dtype=float))

            m = int(_vec(x_mid).size)
            self.constraints = [(lambda k: (lambda x, k=k: _vec(x)[k]))(k) for k in range(m)]
        else:
            self.constraints = list(constraints)        # already a list

        # after self.constraints is constructed
        self._constraints_fn = None                  # keep None; we normalized to a list already
        self._constraints_list = self.constraints    # solvers read from this
        self.n_constr = len(self.constraints)        # solvers pass this to Pymoo

        self.pareto_front = []

        # Wrap the objective to log evaluations and gate infeasible points
        orig_obj = objective
        def wrapped(x):
            x = np.asarray(x).flatten()
            val = orig_obj(x)
            if self.constraints:
                if not all(c(x) >= 1e-6 for c in self.constraints):   # your convention: feasible if ≥ 0
                    return np.array([np.inf] * len(val))
            self.pareto_front.append(val)
            return val

        self.objective = wrapped

# Solver definitions
class BaseSolver:
    name: str = "Base"
    def __init__(self, budget):
        self.budget = budget
    
    def solve(self, prob: Problem) -> Dict:
        raise NotImplementedError

class PymooNSGA2(BaseSolver):
    name = "NSGA2"
    def __init__(self, budget):
        super().__init__(budget)
        self.pop_size = 50
    
    def solve(self, prob):
        n_gen = self.budget // self.pop_size

        # inside each _Prob(...) in NSGA2/MOEAD/SPEA2:

        class _Prob(PymooProblem):
            def __init__(self):
                super().__init__(
                    n_var=prob.dim,
                    n_obj=2,
                    n_constr=prob.n_constr,
                    xl=np.array([b[0] for b in prob.bounds]),
                    xu=np.array([b[1] for b in prob.bounds]),
                    elementwise=True
                )

            def _evaluate(self, x, out):
                out["F"] = prob.objective(x)
                if prob.n_constr > 0:
                    # Pymoo expects G(x) <= 0; your constraints are c(x) >= 0 → negate.
                    if prob._constraints_fn is not None:
                        out["G"] = -np.atleast_1d(np.array(prob._constraints_fn(x), dtype=float))
                    else:
                        out["G"] = -np.array([c(x) for c in prob._constraints_list], dtype=float)

        algo = NSGA2(pop_size=self.pop_size)
        res = minimize(_Prob(), algo, termination=("n_gen", n_gen), seed=prob.instance, verbose=False)
        n_eval = getattr(res, "n_eval", len(prob.pareto_front))
        return dict(X=res.X, F=res.F, n_eval=n_eval)

class PymooMOEAD(PymooNSGA2):
    name = "MOEAD"
    def solve(self, prob):
        n_gen = self.budget // self.pop_size
        ref_dirs = get_reference_directions("uniform", 2, n_points=self.pop_size)
        
        class _Prob(PymooProblem):
            def __init__(self):
                super().__init__(
                    n_var=prob.dim,
                    n_obj=2,
                    n_constr=prob.n_constr,
                    xl=np.array([b[0] for b in prob.bounds]),
                    xu=np.array([b[1] for b in prob.bounds]),
                    elementwise=True
                )

            def _evaluate(self, x, out):
                out["F"] = prob.objective(x)
                if prob.n_constr > 0:
                    # Pymoo expects G(x) <= 0; your constraints are c(x) >= 0 → negate.
                    if prob._constraints_fn is not None:
                        out["G"] = -np.atleast_1d(np.array(prob._constraints_fn(x), dtype=float))
                    else:
                        out["G"] = -np.array([c(x) for c in prob._constraints_list], dtype=float)
        algo = MOEAD(ref_dirs=ref_dirs)
        res = minimize(_Prob(), algo, termination=("n_gen", n_gen), seed=prob.instance, verbose=False)
        n_eval = getattr(res, "n_eval", len(prob.pareto_front))
        return dict(X=res.X, F=res.F, n_eval=n_eval)

class PymooSPEA2(PymooNSGA2):
    name = "SPEA2"
    def solve(self, prob):
        n_gen = self.budget // self.pop_size
        class _Prob(PymooProblem):
            def __init__(self):
                super().__init__(
                    n_var=prob.dim,
                    n_obj=2,
                    n_constr=prob.n_constr,
                    xl=np.array([b[0] for b in prob.bounds]),
                    xu=np.array([b[1] for b in prob.bounds]),
                    elementwise=True
                )

            def _evaluate(self, x, out):
                out["F"] = prob.objective(x)
                if prob.n_constr > 0:
                    # Pymoo expects G(x) <= 0; your constraints are c(x) >= 0 → negate.
                    if prob._constraints_fn is not None:
                        out["G"] = -np.atleast_1d(np.array(prob._constraints_fn(x), dtype=float))
                    else:
                        out["G"] = -np.array([c(x) for c in prob._constraints_list], dtype=float)
        algo = SPEA2(pop_size=self.pop_size)
        res = minimize(_Prob(), algo, termination=("n_gen", n_gen), seed=prob.instance, verbose=False)
        n_eval = getattr(res, "n_eval", len(prob.pareto_front))
        return dict(X=res.X, F=res.F, n_eval=n_eval)