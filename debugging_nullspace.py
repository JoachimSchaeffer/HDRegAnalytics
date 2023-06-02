# %%
import numpy as np
import scipy as sp
from scipy.optimize import minimize, rosen

# %%
x0 = [0.1, 0.7, 0.1, 6]
res = minimize(rosen, x0, method="Nelder-Mead", tol=1e-6)
print(res)
# %%
cons = {"type": "ineq", "fun": lambda x: x[0] - 2 * x[1] + 2}
# %%
