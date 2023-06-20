# The following code is partly copied, extended on build on:
# https://github.com/lawrennd/mlai/
# based on commit: bb4b776a21ec17001bf20ee6406324217f902944
# expand it to the different basis funcitons in the source.
# Author: Joachim Schaeffer, 2023, joachim.schaeffer@posteo.de
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
from abc import ABC, abstractmethod
from typing import Callable


class BasisDataConstructor:
    def __init__(
        self,
        basis_function: BasisFunction,
        basis_weights: np.ndarray,
        targetfun: Callable[[np.ndarray], float],
        d: np.ndarray,
        per_range: list = None,
        **kwargs,
    ):
        self.basis_function = basis_function
        self.d = d
        self.kwargs = kwargs
        self.construct_X_data(basis_weights)
        self.construct_y_data(targetfun, per_range)

    def construct_X_data(self, basis_weights: np.ndarray) -> None:
        if self.basis_function.num_basis != basis_weights.shape[1]:
            raise ValueError(
                "Number of basis weights per observation must equal the number defined for this object!"
            )
        phi_vals = self.basis_function(self.d)
        self.X = np.zeros((basis_weights.shape[0], phi_vals.shape[0]))
        for i in range(basis_weights.shape[0]):
            # Iterate through the rows.
            self.X[i, :] = np.dot(phi_vals, basis_weights[i, :].T)

    def construct_y_data(
        self, targetfun: Callable[[np.ndarray], float], per_range: list = None
    ) -> None:
        """Caller to construct responsese, y. It is necessary to run construct_X_data first"""
        self.y = construct_y_data(self.X, targetfun, per_range)


class BasisFunction(ABC):
    def __init__(self, num_basis: int, data_limits: list = None, **kwargs):
        self.num_basis = num_basis
        if data_limits is None:
            self.data_limits = [-1.0, 1.0]
        else:
            self.data_limits = data_limits
        self.kwargs = kwargs

    @abstractmethod
    def __call__(self, d: np.ndarray) -> np.ndarray:
        pass


class PolynomBasis(BasisFunction):
    def __init__(self, num_basis: int, data_limits: list = None, center : float = None, **kwargs):
        super().__init__(num_basis, data_limits, **kwargs)
        self.center = center

    def __call__(self, d: np.ndarray) -> np.ndarray:
        # centre = self.data_limits[0] / 2.0 + self.data_limits[1] / 2.0
        span = self.data_limits[1] - self.data_limits[0]
        z = d - self.center
        z = 2 * z / span
        Phi = np.zeros((d.shape[0], self.num_basis))
        for i in range(self.num_basis):
            Phi[:, i : i + 1] = z**i
        return Phi


class RadialBasis(BasisFunction):
    def __init__(
        self, num_basis: int, data_limits: list = None, width: float = None, **kwargs
    ):
        super().__init__(num_basis, data_limits, **kwargs)
        self.width = width

    def __call__(self, d: np.ndarray) -> np.ndarray:
        if self.num_basis > 1:
            centres = np.linspace(
                self.data_limits[0], self.data_limits[1], self.num_basis
            )
            if self.width is None:
                self.width = (centres[1] - centres[0]) / 2.0
        else:
            centres = np.asarray([self.data_limits[0] / 2.0 + self.data_limits[1] / 2.0])
            if self.width is None:
                self.width = (self.data_limits[1] - self.data_limits[0]) / 2.0

        Phi = np.zeros((d.shape[0], self.num_basis))
        for i in range(self.num_basis):
            Phi[:, i : i + 1] = np.exp(
                -0.5 * ((np.asarray(d, dtype=float) - centres[i]) / self.width) ** 2
            )
        return Phi


class FourierBasis(BasisFunction):
    def __init__(self, num_basis: int, data_limits: list = None, **kwargs):
        super().__init__(num_basis, data_limits, **kwargs)

    def __call__(self, d: np.ndarray) -> np.ndarray:
        tau = 2 * np.pi
        Phi = np.ones((d.shape[0], self.num_basis))
        for i in range(1, self.num_basis):
            if i % 2:
                Phi[:, i : i + 1] = np.sin((i + 1) * tau * np.asarray(d, dtype=float))
            else:
                Phi[:, i : i + 1] = np.cos((i + 1) * tau * np.asarray(d, dtype=float))
        return Phi


def construct_data(
    basis_function: BasisFunction,
    target_function: Callable[[np.ndarray], float],
    mean_params: np.ndarray,
    stdv_params: np.ndarray,
    num_datapoints: int = 50,
    draws: int = 10,
    plot_results: bool = False,
) -> BasisDataConstructor:
    """Build an object of the basis class based on the passed parameters and return the basis object.

    Parameters
    ----------
    basis_function : callable (basis.function)
        Basis function defined in basis.py that it used for generating the data.
    target_function : callable
        Underlying relationship between X and y. This can be any function from R^n -> R^1
    mean_params : ndarray of shape (n_params for basis function)
        Array of means of random paramters that are used by the basis functions.
        The random parameters are drawn from normal distribution.
    stdv_params : ndarray of shape (n_params for basis function)
        Array of standarddeviations of random paramters that are used by the basis functions.
    num_datapoints : int, default=50
        Number of linearly spaced datapoints that will range from x_min to x_max
    draws : int, default=10
        Number of draws from basis functions.
    plot_results : bool, default=False
        If True, plot the data matrix.
        IF False, do not plot.

    Returns
    -------
    obj : BasisDataConstructor
        Object of the class BasisDataConstructor that contains the data matrix X and the target vector y.
    """
    d_min = basis_function.data_limits[0]
    d_max = basis_function.data_limits[1]
    d = np.linspace(d_min, d_max, num_datapoints)[:, None]

    # Draw the parameters for the matrix from a multidimensional normal distribution
    param_vals = np.zeros((draws, len(mean_params)))
    for i, (j, k) in enumerate(zip(mean_params, stdv_params)):
        param_vals[:, i] = np.array(
            [np.random.normal(loc=j, scale=k) for _ in range(draws)]
        )

    obj = BasisDataConstructor(basis_function, param_vals, target_function, d)

    if plot_results:
        plt.plot(d, obj.X.T)
        plt.title("Data Generated from Basis Function")
        plt.show()

    return obj


def construct_y_data(
    X: np.ndarray, targetfun: Callable[[np.ndarray], float], per_range: list = None
) -> np.ndarray:
    """Construct responsese, y. It is necessary to run construct_X_data first.

    Parameters
    ----------
    targetfun : callable
        Underlying relationship between X and y. This can be any function from R^n -> R^1
        This is also the ideal feature for predicting y and thus the information we would like to discover by applying the lionearization methodology.
    percentage_range : ndrray, default=[0,1]
        1D array with two elements, the first one being strictly smaller than the second value, both being strcitly between 0 and 1,
        defines the range of input data that shall be used to generate the target function
        The reson behind this is that in process data analytics often a sitation can arise where only a part of the data is relevant to predict the target y
    """
    if per_range is None:
        per_range = [0, 1]

    columns = X.shape[1]
    low_ind = int(per_range[0] * columns)
    high_ind = int(per_range[1] * columns)
    # self.y = targetfun(self.X[:, low_ind:high_ind])

    rows = X.shape[0]
    y = np.zeros([rows])
    for i in range(rows):
        row_i = X[i, :]
        y[i] = targetfun(row_i[low_ind:high_ind])

    return y
