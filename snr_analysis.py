# Script to anlyze the Singal-to-Noise ratio of the Battery data
# 1. Fit a smooth function to the data
# Analyse the average and standard deviation of the SNR assoicated with each voltage!

# %%
import sys
import os

abspath = r"./src/"
sys.path.append(os.path.abspath(abspath))
import numpy as np  # noqa
import pandas as pd  # noqa
from scipy.interpolate import splrep, BSpline  # noqa
from symfit import parameters, variables, sin, cos, Fit, Piecewise, Model, Eq  # noqa
import matplotlib.pyplot as plt  # noqa
from plotting_utils import plot_X  # noqa


def fourier_series(x, f, n=0):
    """
    Returns a symbolic fourier series of order `n`.

    :param n: Order of the fourier series.
    :param x: Independent variable
    :param f: Frequency of the fourier series
    """
    # Make the parameter objects for all the terms
    a0, *cos_a = parameters(",".join(["a{}".format(i) for i in range(0, n + 1)]))
    sin_b = parameters(",".join(["b{}".format(i) for i in range(1, n + 1)]))
    # Construct the series
    series = a0 + sum(
        ai * cos(i * f * x) + bi * sin(i * f * x)
        for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1)
    )
    return series


# %%
# Cool, now lets try this next with the battery data
# Load the LFP Dataset
data_path = "./data/"
lfp_df = pd.read_csv(data_path + "lfp_slim.csv", index_col=0)

X_lfp = np.array(lfp_df.iloc[:, 0:1000])
X_lfp = X_lfp[:, ::-1]
y_lfp_true = np.array(lfp_df.iloc[:, 1000])
x_lfp = np.linspace(2.0, 3.5, 1000)

X_lfp_train = np.array(X_lfp[lfp_df.iloc[:, 1002] == 0, :])
y_lfp_train_true = np.array(y_lfp_true[lfp_df.iloc[:, 1002] == 0])
X_lfp_test = np.array(X_lfp[lfp_df.iloc[:, 1002] == 1, :])
y_lfp_test_true = np.array(y_lfp_true[lfp_df.iloc[:, 1002] == 1])
X_lfp_test2 = np.array(X_lfp[lfp_df.iloc[:, 1002] == 2, :])
y_lfp_test2_true = np.array(y_lfp_true[lfp_df.iloc[:, 1002] == 2])

fig, ax = plot_X(X_lfp_train, x_lfp)

# %%
# Cool, lets move on and calculate the SNR for each voltage and for each battery.

# First, lets fit a smooth function to the data
y_fitting = X_lfp_train[0, :]
for i in range(5):
    plt.plot(x_lfp, X_lfp_train[i, :])
plt.show()


def standardize(X):
    stdx = np.std(X, axis=0)
    meanx = np.mean(X, axis=0)
    X_ = X - meanx
    X_std = X_ / stdx
    return X_std


for i in range(5):
    plt.plot(x_lfp, standardize(X_lfp_train)[i, :])
plt.show()

# %% Now lets fit

x, y = variables("x, y")
(w,) = parameters("w")
model_dict = {y: fourier_series(x, f=w, n=10)}
print(model_dict)


# Define a Fit object for this model and data
fit = Fit(model_dict, x=x_lfp, y=y_fitting)
fit_result = fit.execute()
print(fit_result)

# %% Plot the result
# plt.plot(x_lfp, y_fitting)
plt.plot(x_lfp, fit.model(x=x_lfp, **fit_result.params).y, ls=":")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# --> A fourier basis function is not a good fit for this data, as the data has section that are flat and not periodic.
# It would be required to fit a piecewise function to the data.

# %% Or to fit a simple polynomial of order n to the data.
z = np.polyfit(x_lfp, y_fitting, 5)
p = np.poly1d(z)
plt.plot(x_lfp, y_fitting)
plt.plot(x_lfp, p(x_lfp), "r--")
plt.show()
# Same issues here, the strugture of the data requires a piecewise function.


# %% Idea: Split the data t 3.1V and fit a polynomial to the data in both regions.
def polynomial(x, var_letter, n=5):
    """
    Returns a symbolic polynomial function of order `n`.

    :param n: Order of the polynomial.
    :param x: Independent variable
    """
    # Make the parameter objects for all the terms
    poly_coefs = parameters(",".join([f"{var_letter}{i}" for i in range(0, n + 1)]))

    polynomial = sum(coef * x**i for i, coef in enumerate(poly_coefs))

    return polynomial


x, y = variables("x, y")
model_dict = {y: polynomial(x, var_letter="a", n=5)}
fit = Fit(model_dict, x=x_lfp, y=y_fitting)
fit_result = fit.execute()
plt.plot(x_lfp, y_fitting)
plt.plot(x_lfp, fit.model(x=x_lfp, **fit_result.params).y, "r--")
plt.show()
# %%

(x_switch,) = parameters("x_switch")
# For some reason the polynomials must have a different order.
y1 = polynomial(x, var_letter="a", n=15)
y2 = polynomial(x, var_letter="b", n=3)

model = Model({y: Piecewise((y1, x <= x_switch), (y2, x > x_switch))})
# constraints = [Eq(y1.subs({x: x_switch}), y2.subs({x: x_switch}))]

# Flexible switch point
x_switch.min = 3.2
x_switch.max = 3.2

fit = Fit(model, x=x_lfp, y=y_fitting)  # , constraints=constraints)
fit_result = fit.execute()
print(fit_result)


plt.plot(x_lfp, y_fitting)
plt.plot(x_lfp, fit.model(x=x_lfp, **fit_result.params).y, "r--")
plt.show()

# This toolbox  doesnt seem to do the job...

# %% BSplines are an alternative!

tck = splrep(x=x_lfp, y=y_fitting, s=0.000001, k=5)

y_spline = BSpline(*tck)(x_lfp)

# plt.plot(x_lfp, y_fitting)
plt.plot(x_lfp, y_spline, "r--")
plt.show()

plt.plot(x_lfp, (y_fitting - y_spline) / y_fitting, "r--")
plt.show()


# %% Now do it for the whole dataset
# create en empty array X to store the spline smoothed data
X_lfp_spline = np.empty(X_lfp.shape)
# X_lfp_spline_std = np.empty(X_lfp.shape)

# Loop over all the batteries and fit a spline to the data
for i in range(X_lfp.shape[0]):
    tck = splrep(x=x_lfp, y=X_lfp[i, :], s=0.000001, k=5)
    X_lfp_spline[i, :] = BSpline(*tck)(x_lfp)

# Ideas for methods from: https://github.com/hrtlacek/SNR/blob/main/SNR.ipynb
# For this case it really doensn't matter at all!

Power_signal = np.mean(X_lfp_spline, axis=0) ** 2
Power_signal_with_noise = np.mean(X_lfp, axis=0) ** 2
Power_noise = np.mean((X_lfp - X_lfp_spline) ** 2, axis=0)

SNR_method_a = (Power_signal_with_noise - Power_noise) / Power_noise
SNR_method_b = Power_signal / Power_noise
SNR_method_c = np.mean(X_lfp_spline, axis=0) / np.std(X_lfp_spline, axis=0)

plt.plot(x_lfp, 10 * np.log10(SNR_method_a))
plt.plot(x_lfp, 10 * np.log10(SNR_method_b))
plt.plot(x_lfp, 20 * np.log10(SNR_method_c))

# -> Great! This Shows nicely that the noise is most likely heteroscedastic, violates the usual assumption!
# negative SNR means that there is more noise than signal! (only when measured in dB)

# %%

# Apply them on the Standardized data and the non-standardized data, come up with stats for the SNR


# %%
# Neat, now lets use this inforamtion as a prior in the Bayesian linear regression.

# Example 1: Synthetic response: Here this will not yield benefits, as the heteroscadasticity is not present in the X y realtionship

# Example 2: Cycle life response, here things might become interesting.
