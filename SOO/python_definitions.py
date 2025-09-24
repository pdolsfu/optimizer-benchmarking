# python_definitions.py, defines the configurations for optimizing on all eight unconstrained SOO problems

import numpy as np
from python_functions import rosenbrock, shubert, michalewicz

objective_configs = {
    "rosenbrock10": {
        "func": rosenbrock,
        "ndim": 10,
        "budget": 1000,
        "bounds": [(-10,10)],
    },

    "rosenbrock50": {
    "func": rosenbrock,
    "ndim": 50,
    "budget": 1000,
    "bounds": [(-10,10)],
},

    "schubert10": {
    "func": shubert,
    "ndim": 10,
    "budget": 1000,
    "bounds": [(-5.12,5.12)],
},

    "schubert30": {
    "func": shubert,
    "ndim": 30,
    "budget": 1000,
    "bounds": [(-5.12,5.12)],
},

    "schubert60": {
    "func": shubert,
    "ndim": 60,
    "budget": 1000,
    "bounds": [(-5.12,5.12)],
},

    "michalewicz10": {
    "func": michalewicz,
    "ndim": 10,
    "budget": 1000,
    "bounds": [(0, np.pi)],
},

    "michalewicz30": {
    "func": michalewicz,
    "ndim": 30,
    "budget": 1000,
    "bounds": [(0, np.pi)],
},

    "michalewicz60": {
    "func": michalewicz,
    "ndim": 30,
    "budget": 1000,
    "bounds": [(0, np.pi)],
},
}