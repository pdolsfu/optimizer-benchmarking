# python_functions.py, defines the problems and the constraints for the MOO functions

import numpy as np

def zdt1(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    f1 = x[0]

    n = len(x)
    g = 1 + 9 / (n - 1) * np.sum(x[1:])  # x_2 to x_n
    h = 1 - np.sqrt(f1 / g)
    f2 = g * h

    return [float(f1), float(f2)]

def zdt2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    f1 = x[0]

    n = len(x)
    g = 1 + 9 / (n - 1) * np.sum(x[1:])
    h = 1 - (f1 / g) ** 2
    f2 = g * h

    return [float(f1), float(f2)]

def zdt3(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    f1 = x[0]

    n = len(x)
    g = 1 + 9 / (n - 1) * np.sum(x[1:])
    h = 1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1)
    f2 = g * h

    return [float(f1), float(f2)]


def zdt6(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    f1 = 1 - np.exp(-4 * x[0]) * (np.sin(6 * np.pi * x[0]) ** 6)

    n = len(x)
    g = 1 + 9 * ((np.sum(x[1:]) / (n - 1)) ** 0.25)
    h = 1 - (f1 / g) ** 2
    f2 = g * h

    return [float(f1), float(f2)]

def osy(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    # Objective functions
    f1 = -(
        25 * (x[0] - 2) ** 2
        + (x[1] - 2) ** 2
        + (x[2] - 1) ** 2
        + (x[3] - 4) ** 2
        + (x[4] - 1) ** 2
    )
    f2 = np.sum(x**2)

    return [float(f1), float(f2)]

def osy_constraints(x: np.ndarray) -> list:
    x = np.asarray(x)
    return [
        x[0] + x[1] - 2,                          # C1(x) ≥ 0
        6 - x[0] - x[1],                          # C2(x) ≥ 0
        2 - x[1] + x[0],                          # C3(x) ≥ 0
        2 - x[0] + 3 * x[1],                      # C4(x) ≥ 0
        4 - (x[2] - 3) ** 2 - x[3],               # C5(x) ≥ 0
        (x[4] - 3) ** 2 + x[5] - 4                # C6(x) ≥ 0
    ]

def geartrain(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    # x = [x1, x2, x3, x4] all in [12, 60], integer values expected in real design
    f1 = (1 / 6.931 - (x[2] * x[1]) / (x[0] * x[3])) ** 2
    f2 = max(x)  # Minimizing the maximum number of teeth among gears

    return [float(f1), float(f2)]