
from scipy.optimize import minimize

def hybrid_optimize(objective_function, initial_guess):
    result = minimize(objective_function, initial_guess, method='BFGS')
    return result.x, result.fun
