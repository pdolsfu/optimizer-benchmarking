# python_definitions, the script that defines the configurations for each of the MOO problems

from python_functions_and_constraints import zdt1, zdt2, zdt3, zdt6, osy, osy_constraints, geartrain

objective_configs = {
    "zdt1": {
        "func": zdt1,
        "ndim": 30,
        "budget": 250,
        "bounds": [(0, 1)],
        "has_constraints": False,
        "constraints": None
    },

    "zdt2": {
    "func": zdt2,
    "ndim": 30,
    "budget": 250,
    "bounds": [(0, 1)],
    "has_constraints": False,
    "constraints": None
    },

    "zdt3": {
    "func": zdt3,
    "ndim": 30,
    "budget": 250,
    "bounds": [(0, 1)],
    "has_constraints": False,
    "constraints": None
    },

    "zdt6": {
    "func": zdt6,
    "ndim": 10,
    "budget": 250,
    "bounds": [(0, 1)],
    "has_constraints": False,
    "constraints": None
    },

    "osy": {
    "func": osy,
    "ndim": 6,
    "budget": 500,
    "bounds": [(0, 10), (0, 10), (1, 5), (0, 6), (1, 5), (0, 10)],
    "has_constraints": True,
    "constraints": osy_constraints
    },
    "geartrain": {
    "func": geartrain,
    "ndim": 4,
    "budget": 1000,
    "bounds": [(12, 60)]*4,
    "has_constraints": False,
    "constraints": None
    }
}